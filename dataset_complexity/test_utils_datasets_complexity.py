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
    expressibility_vs_locality_ratio,
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
)


class DummyCircuit:
    def __init__(self, m: int):
        self.m = m


class DummyDualRailEmbedder:
    def __init__(self, states: torch.Tensor, m: int = 4):
        self.states = states
        self.output_keys = [(1, 0), (0, 1)]
        self.computation_space = ml.ComputationSpace.DUAL_RAIL
        self.circuit = DummyCircuit(m)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return self.states[int(x.item())]
        flat_indices = x.reshape(-1).to(dtype=torch.long)
        return self.states[flat_indices]


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


@pytest.mark.xfail(reason="Metric is not implemented yet.")
def test_correlation_order_independent_features_should_be_minimal():
    samples = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert correlation_order(samples) == pytest.approx(1.0)


@pytest.mark.xfail(reason="Metric is not implemented yet.")
def test_kolmogorov_complexity_constant_dataset_should_be_low():
    constant = torch.zeros((8, 4))
    varied = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    assert kolmogorov_complexity(constant) < kolmogorov_complexity(varied)


@pytest.mark.xfail(reason="Metric is not implemented yet.")
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
    embedder = DummyDualRailEmbedder(product_state.unsqueeze(0).repeat(2, 1))
    x = torch.tensor([0, 1], dtype=torch.float32)

    assert entanglement_entropy(x, embedder) == pytest.approx(0.0, abs=1e-6)


def test_entanglement_entropy_is_log_two_for_bell_states(bell_state):
    embedder = DummyDualRailEmbedder(bell_state.unsqueeze(0).repeat(2, 1))
    x = torch.tensor([0, 1], dtype=torch.float32)

    assert entanglement_entropy(x, embedder) == pytest.approx(np.log(2), rel=1e-5)


def test_kernel_spectrum_flatness_matches_participation_ratio(monkeypatch):
    kernel = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    class DummyKernel:
        def __init__(self, *args, **kwargs):
            pass

        def compute_kernel_matrix(self, x):
            return kernel

    monkeypatch.setattr("dataset_complexity.utils.NeuralEmbeddingMerLinKernel", DummyKernel)

    value = kernel_spectrum_flatness(
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        DummyDualRailEmbedder(torch.eye(2, dtype=torch.complex64)),
    )

    assert value == pytest.approx(2.0)


@pytest.mark.xfail(reason="Metric is not implemented yet.")
def test_expressibility_vs_locality_ratio_more_expressive_embedding_should_score_higher():
    local_score = expressibility_vs_locality_ratio(torch.zeros(2, 1), None)
    expressive_score = expressibility_vs_locality_ratio(torch.ones(2, 1), None)

    assert expressive_score > local_score


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
    densities = _density_from_state(product_state).unsqueeze(0).repeat(2, 1, 1)

    value = average_bipartite_entanglement_entropy(
        densities,
        ml.ComputationSpace.DUAL_RAIL,
        state_keys=[(1, 0), (0, 1)],
        n_modes=4,
    )

    assert value == pytest.approx(0.0, abs=1e-6)


def test_average_bipartite_entanglement_entropy_is_log_two_for_bell_density(bell_state):
    densities = _density_from_state(bell_state).unsqueeze(0).repeat(2, 1, 1)

    value = average_bipartite_entanglement_entropy(
        densities,
        ml.ComputationSpace.DUAL_RAIL,
        state_keys=[(1, 0), (0, 1)],
        n_modes=4,
    )

    assert value == pytest.approx(np.log(2), rel=1e-5)


@pytest.mark.xfail(reason="Metric is not implemented yet.")
def test_multipartite_total_correlation_product_states_should_be_zero():
    product_density = _density_from_state(
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    ).unsqueeze(0)

    assert multipartite_total_correlation(product_density) == pytest.approx(0.0)


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

    assert {tuple(partition) for partition in bipartitions} == {
        (1, 2),
        (0, 1),
        (0, 2),
    }
