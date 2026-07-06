"""
By codex

Tests for the dataset complexity metrics discussed in arXiv:2509.16410.

Implemented metrics are tested with executable examples. Metrics that are still
missing or currently buggy are captured as xfail specification tests so the
expected behavior is explicit before implementation lands.
"""

import merlin as ml
import numpy as np
import pytest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


from dataset_complexity.utils import (
    _all_photon_mode_configurations,
    _get_all_bipartitions,
    average_bipartite_entanglement_entropy,
    correlation_order,
    distributional_entropy,
    embbed_density_into_complete_fock_space,
    effective_kernel_rank,
    entanglement_entropy,
    locality_vs_expressibility,
    get_kernel_matrix,
    hilbert_space_support_dim,
    kernel_spectrum_flatness,
    kolmogorov_complexity,
    multipartite_total_correlation,
    nonclassicality,
    partial_trace_from_density,
    quantum_entropy,
    quantum_fisher_information,
    quantum_fisher_information_spread,
    topological_complexity,
    topological_invariants_of_embedding,
    topological_quantum_complexity,
    kl_div,
)


class DummyCircuit:
    def __init__(self, m: int):
        self.m = m


class DummyDualRailEmbedder:
    def __init__(self, states: torch.Tensor, m: int = 4, num_photons: int = None):
        self.states = states
        self.output_keys = [(1, 0), (0, 1)]
        self.computation_space = ml.ComputationSpace.DUAL_RAIL
        self.circuit = DummyCircuit(m)
        self.num_modes = m
        # Allow override for dual-rail tests (n = m // 2)
        self.num_photons = num_photons if num_photons is not None else 1
        self.output_size = states.shape[1] if states.ndim > 1 else states.shape[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return self.states[int(x.item())]
        # 1-D input = single data point from a per-row loop; use first element as index
        if x.ndim == 1:
            return self.states[int(x[0].item())]
        # 2-D input = full batch; index by first column
        return self.states[x[:, 0].to(dtype=torch.long)]


def _density_from_state(state: torch.Tensor) -> torch.Tensor:
    return torch.outer(state, torch.conj(state))


@pytest.fixture
def bell_state():
    return torch.tensor(
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)], dtype=torch.complex64
    )


@pytest.fixture
def product_state():
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)


def test_distributional_entropy_matches_shannon_entropy_for_duplicates():
    samples = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    assert distributional_entropy(samples) == pytest.approx(np.log(2))


def test_distributional_entropy_is_log_n_for_unique_rows():
    samples = torch.tensor([[0.0], [1.0], [2.0], [3.0]])

    assert distributional_entropy(samples) == pytest.approx(np.log(4))


def test_correlation_order_independent_features_should_be_minimal():
    samples = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert correlation_order(samples) == pytest.approx(0.0, abs=1e-6)


def test_correlation_dependant_features():
    samples = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )

    assert correlation_order(samples) == pytest.approx(np.log(4), abs=1e-6)


def test_kolmogorov_complexity_constant_dataset_should_be_low():
    constant = torch.zeros((8, 4))
    varied = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    assert kolmogorov_complexity(constant) < kolmogorov_complexity(varied)


def test_topological_complexity_two_clusters_should_exceed_single_cluster():
    single_cluster = torch.zeros((8, 2))
    two_clusters = torch.tensor([[0.0, 0.0]] * 4 + [[10.0, 10.0]] * 4)

    assert topological_complexity(two_clusters) > topological_complexity(single_cluster)


def test_hilbert_space_support_dim_counts_support_rank_for_orthogonal_states():
    states = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.complex64,
    )
    embedder = DummyDualRailEmbedder(states, m=2)
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    assert hilbert_space_support_dim(x, embedder) == pytest.approx(2.0, rel=1e-5)


def test_quantum_fisher_information_spread_averages_traces(monkeypatch):
    matrices = [
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
    ]

    def fake_get_quantum_fisher_matrices(model, inputs):
        return matrices

    monkeypatch.setattr(
        "dataset_complexity.utils.get_quantum_fisher_matrices",
        fake_get_quantum_fisher_matrices,
    )

    value = quantum_fisher_information_spread(
        torch.tensor([0.0, 1.0]),
        DummyDualRailEmbedder(torch.eye(2, dtype=torch.complex64)),
    )

    assert value == pytest.approx(5.0)


def test_entanglement_entropy_is_zero_for_product_states(product_state):
    embedder = DummyDualRailEmbedder(
        product_state.unsqueeze(0).repeat(2, 1), m=4, num_photons=2
    )
    x = torch.tensor([0, 1], dtype=torch.float32)

    assert entanglement_entropy(x, embedder) == pytest.approx(0.0, abs=1e-6)


def test_entanglement_entropy_is_log_two_for_bell_states(bell_state):
    embedder = DummyDualRailEmbedder(
        bell_state.unsqueeze(0).repeat(2, 1), m=4, num_photons=2
    )
    x = torch.tensor([0, 1], dtype=torch.float32)

    assert entanglement_entropy(x, embedder) == pytest.approx(np.log(2), rel=1e-5)


def test_kernel_spectrum_flatness_matches_participation_ratio(monkeypatch):
    # Dummy embedder as nn.Module that maps input index to orthogonal quantum state
    class DummyEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.states = torch.eye(2, dtype=torch.complex64)

        def forward(self, x):
            # x is a single feature vector (1D tensor)
            idx = int(x[0])  # Assume first element is the index
            return self.states[idx]

        def compute_kernel_matrix(self, x):
            # x is (N, d), call forward per row
            psi = torch.stack([self.forward(xi) for xi in x], dim=0)
            return torch.abs(psi @ psi.conj().T) ** 2

    embedder = DummyEmbedder()
    x = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    value = kernel_spectrum_flatness(x, embedder)
    assert value == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kl, entropy, expected_high",
    [
        # Metric: M = (exp(-kl_div) - 2*S/log(N))^2
        # DummyDualRailEmbedder(m=4): N=4, S_max=log(2)
        # e_hat = exp(-kl),  s_hat = S / S_max = 2*S / log(N)
        #
        # Expressive + local  (BAD):  e_hat≈1, s_hat≈0 → M≈1
        pytest.param(0.01, 0.0, True, id="expressive_local_bad"),
        # Non-expressive + global (BAD): e_hat≈0, s_hat≈1 → M≈1
        pytest.param(10.0, np.log(2), True, id="non_expressive_global_bad"),
        # Expressive + global (GOOD):  e_hat≈1, s_hat≈1 → M≈0
        pytest.param(0.01, np.log(2), False, id="expressive_global_good"),
        # Non-expressive + local (GOOD): e_hat≈0, s_hat≈0 → M≈0
        pytest.param(10.0, 0.0, False, id="non_expressive_local_good"),
    ],
)
def test_locality_vs_expressibility_mismatch_metric(
    monkeypatch, kl, entropy, expected_high
):
    embedder = DummyDualRailEmbedder(torch.eye(2, dtype=torch.complex64))
    x = torch.rand(4, 2)

    monkeypatch.setattr("dataset_complexity.utils.kl_div", lambda *a, **kw: kl)
    monkeypatch.setattr(
        "dataset_complexity.utils.entanglement_entropy", lambda *a, **kw: entropy
    )

    score = locality_vs_expressibility(x, embedder)

    if expected_high:
        assert score > 0.5, f"Expected M > 0.5 (miscalibrated), got {score}"
    else:
        assert score < 0.05, f"Expected M < 0.05 (well calibrated), got {score}"


def test_kl_div_is_nonnegative(monkeypatch):
    n = 20
    x = torch.stack([torch.arange(n), torch.arange(n)], dim=1)

    class DummyEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.states = torch.eye(n, dtype=torch.complex64)
            self.output_size = n

        def forward(self, x):
            idx = int(x[0])
            return self.states[idx]

        def compute_kernel_matrix(self, x):
            psi = torch.stack([self.forward(xi) for xi in x], dim=0)
            return torch.abs(psi @ psi.conj().T) ** 2

        @property
        def quantum_embedding_layer(self):
            class QEL:
                output_size = n

            return QEL()

    embedder = DummyEmbedder()
    assert kl_div(x, embedder, n_samples=n, n_bins=10) >= 0.0


def test_kl_div_identical_states_yields_large_divergence(monkeypatch):
    n = 30
    x = torch.stack([torch.arange(n), torch.arange(n)], dim=1)

    class DummyEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # All states are identical
            self.states = torch.ones((n, 1), dtype=torch.complex64)
            self.output_size = n

        def forward(self, x):
            idx = int(x[0])
            return self.states[idx]

        def compute_kernel_matrix(self, x):
            psi = torch.stack([self.forward(xi) for xi in x], dim=0)
            return torch.abs(psi @ psi.conj().T) ** 2

        @property
        def quantum_embedding_layer(self):
            class QEL:
                output_size = n

            return QEL()

    embedder = DummyEmbedder()
    assert kl_div(x, embedder, n_samples=n, n_bins=10) > 1.0


def test_kl_div_haar_like_fidelities_yields_small_divergence(monkeypatch):
    D = 4
    rng = np.random.default_rng(42)
    n = 150
    n_pairs = n * (n - 1) // 2
    fidelity_samples = rng.beta(1, D - 1, size=n_pairs).astype(np.float32)
    kernel = torch.eye(n, dtype=torch.float32)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            kernel[i, j] = kernel[j, i] = float(fidelity_samples[k])
            k += 1

    class DummyEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.states = torch.eye(n, dtype=torch.complex64)
            self.output_size = n

        def forward(self, x):
            idx = int(x[0])
            return self.states[idx]

        def compute_kernel_matrix(self, x):
            # Use the precomputed kernel for this test
            return kernel

        @property
        def quantum_embedding_layer(self):
            class QEL:
                output_size = D

            return QEL()

    embedder = DummyEmbedder()
    x = torch.stack([torch.arange(n), torch.arange(n)], dim=1)
    assert kl_div(x, embedder, n_samples=n, n_bins=20) < 0.5


@pytest.mark.xfail(reason="Metric is not implemented yet.")
def test_topological_invariants_of_embedding_nontrivial_loop_should_not_vanish():
    circle = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    assert topological_invariants_of_embedding(circle, None) > 0


def test_average_bipartite_entanglement_entropy_is_zero_for_product_density(
    product_state,
):
    # For dual-rail, use 4x4 density matrices (2 qubits, n_modes=4)
    density = torch.zeros((4, 4), dtype=torch.complex64)
    density[0, 0] = 1.0  # |00><00|
    densities = density.unsqueeze(0).repeat(2, 1, 1)
    value = average_bipartite_entanglement_entropy(
        densities,
        ml.ComputationSpace.DUAL_RAIL,
        state_keys=[(1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1)],
        n_modes=4,
    )
    assert value == pytest.approx(0.0, abs=1e-6)


def test_average_bipartite_entanglement_entropy_is_log_two_for_bell_density(bell_state):
    # For dual-rail, use 4x4 Bell state density matrix (2 qubits, n_modes=4)
    bell = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / np.sqrt(2)
    density = torch.outer(bell, bell.conj())
    densities = density.unsqueeze(0).repeat(2, 1, 1)
    value = average_bipartite_entanglement_entropy(
        densities,
        ml.ComputationSpace.DUAL_RAIL,
        state_keys=[(1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1)],
        n_modes=4,
    )
    assert value == pytest.approx(np.log(2), rel=1e-5)


def test_multipartite_total_correlation_product_states_should_be_zero():
    product_density = _density_from_state(
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    ).unsqueeze(0)

    assert multipartite_total_correlation(
        product_density, num_subsystem=2, fock_space=False
    ) == pytest.approx(0.0)


def test_effective_kernel_rank_matches_participation_ratio(monkeypatch):
    kernel = torch.tensor([[2.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr("dataset_complexity.utils.get_kernel_matrix", lambda x: kernel)

    rhos = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])

    value = effective_kernel_rank(rhos)

    assert value == pytest.approx(9.0 / 5.0)


@pytest.mark.xfail(reason="Metric is not implemented yet.")
def test_nonclassicality_classical_basis_state_should_be_zero():
    rho = _density_from_state(
        torch.tensor([1.0, 0.0], dtype=torch.complex64)
    ).unsqueeze(0)

    assert nonclassicality(rho) == pytest.approx(0.0)


@pytest.mark.xfail(reason="Metric is not implemented yet.")
def test_quantum_fisher_information_constant_encoding_should_be_zero():
    rho = (
        _density_from_state(torch.tensor([1.0, 0.0], dtype=torch.complex64))
        .unsqueeze(0)
        .repeat(2, 1, 1)
    )

    assert quantum_fisher_information(rho) == pytest.approx(0.0)


@pytest.mark.xfail(reason="Metric is not implemented yet.")
def test_topological_quantum_complexity_product_states_should_be_lower_than_entangled_states(
    bell_state, product_state
):
    product_density = _density_from_state(product_state).unsqueeze(0)
    bell_density = _density_from_state(bell_state).unsqueeze(0)

    assert topological_quantum_complexity(
        bell_density
    ) > topological_quantum_complexity(product_density)


def test_quantum_entropy_is_zero_for_pure_state():
    rho = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)

    assert quantum_entropy(rho) == pytest.approx(0.0, abs=1e-6)


def test_quantum_entropy_is_log_two_for_maximally_mixed_qubit():
    rho = 0.5 * torch.eye(2, dtype=torch.complex64)

    assert quantum_entropy(rho) == pytest.approx(np.log(2), rel=1e-5)


@pytest.mark.xfail(reason="Current implementation fills the kernel matrix incorrectly.")
def test_get_kernel_matrix_returns_identity_for_orthogonal_pure_states():
    zero = _density_from_state(torch.tensor([1.0, 0.0], dtype=torch.complex64))
    one = _density_from_state(torch.tensor([0.0, 1.0], dtype=torch.complex64))
    rhos = torch.stack([zero, one])

    expected = torch.eye(2)
    assert torch.allclose(get_kernel_matrix(rhos), expected, atol=1e-6)


@pytest.mark.xfail(
    reason="Current implementation does not embed into the larger Fock basis."
)
def test_embbed_density_into_complete_fock_space_places_entries_on_requested_basis_states():
    rho = torch.tensor([[0.7, 0.2], [0.2, 0.3]])
    state_keys = [(1, 0), (0, 1)]

    embedded = embbed_density_into_complete_fock_space(
        rho,
        n_modes=2,
        n_photons=1,
        state_keys=state_keys,
    )

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.2, 0.0],
            [0.0, 0.2, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(embedded, expected)


def test_partial_trace_from_density_reduces_bell_state_to_maximally_mixed_qubit(
    bell_state,
):
    rho = _density_from_state(bell_state)

    reduced = partial_trace_from_density(rho, states_to_trace=[0], dim_per_state=2)

    assert torch.allclose(reduced, 0.5 * torch.eye(2, dtype=reduced.dtype), atol=1e-6)


def test_all_photon_mode_configurations_returns_cartesian_product():
    assert _all_photon_mode_configurations(2, 1) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_get_all_bipartitions_returns_unique_nontrivial_splits():
    bipartitions = _get_all_bipartitions(3)
    # Convert sets to sorted tuples for hashability and sort both sides for robust comparison
    bipartition_tuples = {
        tuple(sorted(tuple(sorted(x)) for x in partition)) for partition in bipartitions
    }
    expected = {
        ((0,), (1, 2)),
        ((1,), (0, 2)),
        ((2,), (0, 1)),
    }

    # Sort both sides for robust comparison
    def sort_partition(t):
        return tuple(sorted(t, key=lambda x: (len(x), x)))

    actual_sorted = {sort_partition(t) for t in bipartition_tuples}
    expected_sorted = {sort_partition(t) for t in expected}
    assert actual_sorted == expected_sorted


# --- Additional tests for utils.py ---

from dataset_complexity.utils import _fock_state_to_full_index


def test_fock_state_to_full_index_basic():
    # For n_photons=1, m=2, states: (0,0),(0,1),(1,0),(1,1)
    assert _fock_state_to_full_index((0, 0), 1) == 0
    assert _fock_state_to_full_index((0, 1), 1) == 1
    assert _fock_state_to_full_index((1, 0), 1) == 2
    assert _fock_state_to_full_index((1, 1), 1) == 3


def test_fock_state_to_full_index_large():
    # For n_photons=2, m=3, states: (0,0,0)...(2,2,2)
    idx = _fock_state_to_full_index((2, 1, 0), 2)
    # Should match manual calculation: 2*(2+1)^2 + 1*(2+1) + 0 = 2*9 + 1*3 + 0 = 18+3=21
    assert idx == 21


def test_embbed_density_into_complete_fock_space_correct_embedding():
    # 2 modes, 1 photon, state_keys = [(1,0),(0,1)]
    rho = torch.tensor([[0.7, 0.2], [0.2, 0.3]])
    n_modes = 2
    n_photons = 1
    state_keys = [(1, 0), (0, 1)]
    embedded = embbed_density_into_complete_fock_space(
        rho, n_modes, n_photons, state_keys
    )
    # Should embed into 4x4, with entries at (2,2),(2,1),(1,2),(1,1)
    expected = torch.zeros((4, 4))
    expected[2, 2] = 0.7
    expected[2, 1] = 0.2
    expected[1, 2] = 0.2
    expected[1, 1] = 0.3
    assert torch.allclose(embedded, expected)


def test_entanglement_entropy_random_pure_state_edge_case():
    # Random pure state for 2 qubits (dual rail)
    state = torch.randn(4) + 1j * torch.randn(4)
    state = state / torch.norm(state)
    embedder = DummyDualRailEmbedder(
        state.unsqueeze(0).repeat(2, 1), m=4, num_photons=2
    )
    x = torch.tensor([0, 1], dtype=torch.float32)
    entropy = entanglement_entropy(x, embedder)
    assert entropy >= 0.0
    assert np.isfinite(entropy)


# --- Additional tests for entanglement_entropy in FOCK (non-dual-rail) mode ---
class DummyFockEmbedder:
    def __init__(
        self, states: torch.Tensor, m: int, n: int, state_keys: list[tuple[int]]
    ):
        self.states = states
        self.computation_space = ml.ComputationSpace.FOCK
        self.num_modes = m
        self.num_photons = n
        self.output_keys = state_keys
        self.circuit = DummyCircuit(m)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return self.states[int(x.item())]
        flat_indices = x.reshape(-1).to(dtype=torch.long)
        return self.states[flat_indices]


def _fock_basis_states(m, n):
    # All photon configurations for m modes, n photons
    from itertools import product

    return [s for s in product(range(n + 1), repeat=m) if sum(s) == n]


def test_entanglement_entropy_fock_product_state():
    # 2 modes, 2 photons, product state (all photons in mode 0)
    m, n = 2, 2
    state_keys = _fock_basis_states(m, n)
    # |2,0> in occupation basis
    psi = torch.zeros(len(state_keys), dtype=torch.complex64)
    psi[state_keys.index((2, 0))] = 1.0
    embedder = DummyFockEmbedder(psi.unsqueeze(0).repeat(2, 1), m, n, state_keys)
    x = torch.tensor([0, 1], dtype=torch.float32)
    entropy = entanglement_entropy(x, embedder)
    assert entropy == pytest.approx(0.0, abs=1e-6)


def test_entanglement_entropy_fock_maximally_entangled_state():
    # 2 modes, 2 photons, maximally entangled state: (|2,0> + |0,2>)/sqrt(2)
    m, n = 2, 2
    state_keys = _fock_basis_states(m, n)
    psi = torch.zeros(len(state_keys), dtype=torch.complex64)
    psi[state_keys.index((2, 0))] = 1 / np.sqrt(2)
    psi[state_keys.index((0, 2))] = 1 / np.sqrt(2)
    embedder = DummyFockEmbedder(psi.unsqueeze(0).repeat(2, 1), m, n, state_keys)
    x = torch.tensor([0, 1], dtype=torch.float32)
    entropy = entanglement_entropy(x, embedder)
    # For a 2-level system, entropy should be log(2)
    assert entropy == pytest.approx(np.log(2), rel=1e-5)


def test_entanglement_entropy_fock_random_pure_state():
    # 3 modes, 2 photons, random pure state
    m, n = 3, 2
    state_keys = _fock_basis_states(m, n)
    psi = torch.randn(len(state_keys)) + 1j * torch.randn(len(state_keys))
    psi = psi / torch.norm(psi)
    embedder = DummyFockEmbedder(psi.unsqueeze(0).repeat(2, 1), m, n, state_keys)
    x = torch.tensor([0, 1], dtype=torch.float32)
    entropy = entanglement_entropy(x, embedder)
    assert entropy >= 0.0
    assert np.isfinite(entropy)


# ── Tests derived from arXiv:2509.16410 metric definitions ───────────────────
#
# The paper (Pere, 2025) formalises classical data complexity (Eq. 7) as:
#   C_data = λ1·S(D) + λ2·I_corr(D) + λ3·K(D) + λ4·C_top(D)
# and induced quantum complexity (Eq. 11) as a weighted sum of M1-M6.
# Each block below targets one metric.


# ── §2.1.1  Intrinsic / Effective Dimension (Eq. 1) ──────────────────────────


def test_hilbert_space_support_dim_single_repeated_state_is_one():
    """Paper §2.2 / Eq. 12: when all data maps to the *same* quantum state the
    effective Hilbert-space support dimension should collapse to 1."""
    state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    # Both data points map to the same state.
    states = state.unsqueeze(0).repeat(4, 1)
    embedder = DummyDualRailEmbedder(states, m=2)
    # 2D so the batch branch (x[:, 0]) is used and all 4 states are fetched
    x = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    dim = hilbert_space_support_dim(x, embedder)
    # rho = |0><0|, single eigenvalue 1.  effective_dim = 1/(1+eps) ≈ 1.
    assert dim == pytest.approx(1.0, abs=1e-4)


def test_hilbert_space_support_dim_monotone_with_rank():
    """Paper §2.2: adding more orthogonal states must strictly increase the
    effective dimension (monotonicity in Hilbert-space support)."""

    def _dim_for_n_orthogonal(n: int) -> float:
        states = torch.eye(n, dtype=torch.complex64)
        embedder = DummyDualRailEmbedder(states, m=n)
        x = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        return float(hilbert_space_support_dim(x, embedder))

    d2 = _dim_for_n_orthogonal(2)
    d3 = _dim_for_n_orthogonal(3)
    d4 = _dim_for_n_orthogonal(4)
    assert d2 < d3 < d4


# ── §2.1.2  Correlation / Interaction Order (Eq. 2) ─────────────────────────


def test_correlation_order_perfectly_anticorrelated_features():
    """Paper §2.1.2: features x2 = -x1 are perfectly dependent, so multivariate
    MI is positive and the metric should exceed the independent case."""
    independent = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    anticorrelated = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, -2.0]])
    assert correlation_order(anticorrelated) > correlation_order(independent)


def test_correlation_order_scales_with_interaction_strength():
    """Paper §2.1.2: stronger higher-order coupling → larger I_corr."""
    weak = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    strong = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert correlation_order(strong) >= correlation_order(weak)


# ── §2.1.3  Kolmogorov Complexity / Compressibility (Eq. 3) ─────────────────


def test_kolmogorov_complexity_is_at_most_one():
    """Paper §2.1.3: C(D) = compressed / raw ≤ 1 for any lossless compressor."""
    for tensor in [
        torch.zeros((16, 4)),
        torch.ones((16, 4)),
        torch.arange(64, dtype=torch.float32).reshape(16, 4),
        torch.randn(16, 4),
    ]:
        assert kolmogorov_complexity(tensor) <= 1.0 + 1e-9


def test_kolmogorov_complexity_linearly_dependent_rows_are_compressible():
    """Paper §2.1.3: a dataset whose rows are copies of a single row is
    maximally compressible compared to a random dataset of the same shape."""
    constant = torch.ones((32, 8)) * 3.14
    random = torch.randn(32, 8)
    assert kolmogorov_complexity(constant) < kolmogorov_complexity(random)


# ── §2.1.5  Topological Complexity (Eq. 3 / §2.1.5) ────────────────────────


def test_topological_complexity_ring_has_higher_complexity_than_disk():
    """Paper §2.1.5: a ring (non-trivial β1) should have higher topological
    complexity than a filled disk (trivial topology)."""
    theta = torch.linspace(0, 2 * np.pi, 100)[:-1]
    ring = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    rng = torch.Generator().manual_seed(0)
    disk = (torch.rand(20, 2, generator=rng) - 0.5) * 0.2

    # Compare only H1 (loop) contribution: random dense clouds can create many
    # short-lived higher-dimensional/noise features in the unweighted sum.
    assert topological_complexity(
        ring, weights=[0.0, 1.0, 0.0]
    ) > topological_complexity(disk, weights=[0.0, 1.0, 0.0])


def test_topological_complexity_respects_weights():
    """Paper §2.1.5 / Eq. 3: weights w_k scale the contribution of each
    homological dimension.  Zeroing w_1 should lower the complexity of a
    dataset that has a prominent 1-cycle."""
    theta = torch.linspace(0, 2 * np.pi, 20)[:-1]
    ring = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    c_uniform = topological_complexity(ring, weights=[1.0, 1.0, 1.0])
    c_no_loops = topological_complexity(ring, weights=[1.0, 0.0, 0.0])
    assert c_uniform >= c_no_loops


# ── §2.2.2  Entanglement Entropy (Eq. 5) ────────────────────────────────────


def test_entanglement_entropy_bounded_by_log_dimension():
    """Paper §2.2.2 / Eq. 5: S(ρ_A) ≤ log(dim H_A).  For a 2-qubit system
    (m=4 dual-rail) the max per-bipartition entropy is log(2), so the average
    over all bipartitions should not exceed log(2)."""
    # Maximally entangled Bell state
    bell = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / np.sqrt(2)
    embedder = DummyDualRailEmbedder(bell.unsqueeze(0).repeat(2, 1), m=4, num_photons=2)
    x = torch.tensor([0, 1], dtype=torch.float32)
    ee = entanglement_entropy(x, embedder)
    assert ee <= np.log(2) + 1e-6


def test_entanglement_entropy_increases_toward_bell_state():
    """Paper §2.2.2: entanglement entropy is a monotone of entanglement.
    A state with partial entanglement should yield entropy between 0 and log(2)."""
    alpha = np.cos(np.pi / 8)  # cos(22.5°)
    beta = np.sin(np.pi / 8)
    partial = torch.tensor([alpha, 0.0, 0.0, beta], dtype=torch.complex64)
    embedder = DummyDualRailEmbedder(
        partial.unsqueeze(0).repeat(2, 1), m=4, num_photons=2
    )
    x = torch.tensor([0, 1], dtype=torch.float32)
    ee = entanglement_entropy(x, embedder)
    assert 0.0 < ee < np.log(2)


# ── §2.2.5  Expressibility / KL divergence ───────────────────────────────────


def test_kl_div_returns_nonneg_for_orthogonal_states():
    """Paper §2.2.5: D_KL(P_U || P_Haar) ≥ 0 always (non-negativity of KL
    divergence).  With orthonormal states the off-diagonal fidelities are all 0
    — a valid (if extreme) embedding — so the function must return a
    non-negative finite float."""
    n = 4
    states = torch.eye(n, dtype=torch.complex64)
    embedder = DummyDualRailEmbedder(states, m=n, num_photons=1)
    x = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    result = kl_div(x, embedder, n_samples=n, n_bins=4)
    assert isinstance(result, float)
    assert result >= 0.0


# ── §2.2.3  Kernel spectrum flatness / effective rank (Eq. 12–13) ────────────


def test_kernel_spectrum_flatness_flat_spectrum_gives_maximal_rank():
    """Paper §2.4 / Eq. 12: for K = λI (flat spectrum) the effective rank
    equals N (all N eigenvalues equal).  r_eff = (N·λ)²/(N·λ²) = N."""
    n = 4
    # Build embedder that gives orthonormal states → K = I_n
    states = torch.eye(n, dtype=torch.complex64)

    class OrthEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._s = states

        def forward(self, x):
            idx = int(x[0].item())
            return self._s[idx]

    x = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    ksf = kernel_spectrum_flatness(x, OrthEmbedder())
    assert ksf == pytest.approx(float(n), rel=1e-4)


def test_kernel_spectrum_flatness_rank_one_kernel_gives_one():
    """Paper §2.4 / Eq. 12: if all states are identical K = ones·ones^T / n,
    giving a rank-1 matrix with a single non-zero eigenvalue, r_eff = 1."""
    n = 4
    state = torch.tensor([1.0, 0.0], dtype=torch.complex64)

    class SameEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return state

    x = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    ksf = kernel_spectrum_flatness(x, SameEmbedder())
    assert ksf == pytest.approx(1.0, rel=1e-4)


# ── §2.1.4 / §2.4  Spectral connection to sample complexity (Eq. 12) ────────


def test_quantum_fisher_information_spread_vanishes_for_constant_encoder():
    """Paper §2.4: QFI spread measures sensitivity to *input* perturbations.
    A constant encoder (output independent of input) has zero Jacobian, so
    all FIM matrices are zero and the spread should be 0."""

    class ConstantEncoder(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([1.0, 0.0], dtype=torch.complex128)

    x = torch.randn(8, 3)
    spread = quantum_fisher_information_spread(x, ConstantEncoder())
    assert spread == pytest.approx(0.0, abs=1e-8)


def test_quantum_fisher_information_spread_positive_for_sensitive_encoder():
    """Paper §2.4: an encoder whose output rotates with input should yield a
    positive QFI spread."""

    class RotationEncoder(torch.nn.Module):
        """Maps scalar x to [cos(x), sin(x)] — a pure-state rotation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            theta = x.flatten()[0]
            return torch.stack([torch.cos(theta), torch.sin(theta)]).to(
                torch.complex128
            )

    x = torch.linspace(0, np.pi, 10).unsqueeze(1)
    spread = quantum_fisher_information_spread(x, RotationEncoder())
    assert spread > 0.0


# ── §2.4  Composite classical complexity (Eq. 7) ─────────────────────────────


def test_classical_complexity_returns_all_metric_keys():
    """Paper Eq. 7: the returned dict must contain each named sub-metric plus
    the weighted total."""
    from dataset_complexity.complexity_metrics import classical_complexity

    X = torch.randn(20, 4)
    Y = torch.cat((torch.zeros(10), torch.ones(10))).long()
    result = classical_complexity(X, Y)
    for key in (
        "distributional_entropy",
        "correlation_order",
        "kolmogorov_complexity",
        "topological_complexity",
        "wasserstein distance",
        "total",
    ):
        assert key in result, f"Missing key: {key}"


def test_classical_complexity_total_equals_weighted_sum():
    """Paper Eq. 7: total = λ1·S + λ2·I_corr + λ3·K + λ4·C_top."""
    from dataset_complexity.complexity_metrics import classical_complexity

    weights = [2.0, 0.5, 1.0, 3.0, 0.75]
    X = torch.randn(20, 3)
    Y = torch.cat((torch.zeros(10), torch.ones(10))).long()
    result = classical_complexity(X, Y, hyper_parameters=weights)
    expected = (
        result["distributional_entropy"]
        + result["correlation_order"]
        + result["kolmogorov_complexity"]
        + result["topological_complexity"]
        + result["wasserstein distance"]
    )
    assert result["total"] == pytest.approx(expected, rel=1e-6)


def test_classical_complexity_low_entropy_dataset_has_lower_total():
    """Paper §2.4: a constant (zero-entropy, zero-correlation) dataset must
    have strictly lower classical complexity than a random dataset."""
    from dataset_complexity.complexity_metrics import classical_complexity

    constant = torch.zeros((30, 4))
    random = torch.randn(30, 4)
    labels = torch.cat((torch.zeros(15), torch.ones(15))).long()
    assert (
        classical_complexity(constant, labels)["total"]
        < classical_complexity(random, labels)["total"]
    )


# ── §2.4  Composite induced quantum complexity (Eq. 11) ──────────────────────


def test_induced_quantum_complexity_returns_all_metric_keys():
    """Paper Eq. 11: the dict must contain each M_j sub-metric plus the total."""
    from dataset_complexity.complexity_metrics import induced_quantum_complexity

    states = torch.eye(4, dtype=torch.complex64)
    embedder = DummyDualRailEmbedder(states, m=4, num_photons=2)
    # Shape [4, 1]: rows iterate as 1-D tensors so DummyDualRailEmbedder uses x[0] as index
    X = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    result = induced_quantum_complexity(
        X,
        Y,
        embedder,
        max_samples=4,
        n_samples_loc_vs_express=4,
        n_bins_loc_vs_express=4,
    )
    for key in (
        "hilbert_space_support_dim",
        "quantum_fisher_information_spread",
        "entanglement_entropy",
        "kernel_spectrum_flatness",
        "locality_vs_expressibility",
        "topological_invariants_of_embedding",
        "total",
    ):
        assert key in result, f"Missing key: {key}"


def test_induced_quantum_complexity_total_equals_weighted_sum():
    """Paper Eq. 11: total = Σ β_j · M_j."""
    from dataset_complexity.complexity_metrics import induced_quantum_complexity

    states = torch.eye(4, dtype=torch.complex64)
    embedder = DummyDualRailEmbedder(states, m=4, num_photons=2)
    X = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    result = induced_quantum_complexity(
        X,
        Y,
        embedder,
        max_samples=4,
        n_samples_loc_vs_express=4,
        n_bins_loc_vs_express=4,
    )
    expected = (
        result["hilbert_space_support_dim"]
        + result["quantum_fisher_information_spread"]
        + result["entanglement_entropy"]
        + result["kernel_spectrum_flatness"]
        + result["locality_vs_expressibility"]
        + result["topological_invariants_of_embedding"]
    )
    assert result["total"] == pytest.approx(expected, rel=1e-6)
