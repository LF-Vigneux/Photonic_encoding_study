"""
Not implemented techniques are identified by #####TODO


REMOVE TQDM TO REMOVE THE PROGRESS BAR
"""

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import merlin as ml
import scipy as sp
from itertools import product, combinations
import quimb.tensor as qtn
import zlib
from ripser import ripser
from copy import deepcopy

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


from nn_embedding.lib.merlin_based_model import (
    NeuralEmbeddingMerLinKernel,
)
from nn_embedding.utils.merlin_model_utils import assign_params  # noqa: E402
from nn_embedding.utils.utils import state_vector_to_density_matrix, TransparentModel
from encodings_merlin.utils import compute_kernel_matrix_without_nqe
from scipy.optimize import linear_sum_assignment


# Classical
############################################################################################################
def distributional_entropy(X: torch.Tensor) -> float:
    N = X.size(0)
    _, frequencies = torch.unique(X, dim=0, return_counts=True)
    if frequencies.size(0) == N:
        return np.log(N)

    probs = frequencies / N
    entropy = 0
    for prob in probs:
        entropy += prob * np.log(prob)
    return (-1) * entropy


def correlation_order(
    X: torch.Tensor,
    max_order: int | None = None,
) -> float:
    N, d = X.shape
    if max_order is None:
        max_order = d
    max_order = min(max_order, d)

    X_np = X.detach().numpy()

    def joint_entropy(indices: list[int]) -> float:
        X_sub = X_np[:, indices]
        _, counts = np.unique(X_sub, axis=0, return_counts=True)
        probs = counts / N
        return float(-np.sum(probs * np.log(probs)))

    def multivariate_mi(S: tuple) -> float:
        mi = 0.0
        for j in range(1, len(S) + 1):
            sign = (-1) ** (j - 1)
            for T in combinations(S, j):
                mi += sign * joint_entropy(list(T))
        return mi

    total = 0.0
    for k in range(2, max_order + 1):
        subsets = list(combinations(range(d), k))
        total += sum(multivariate_mi(S) for S in subsets) / len(subsets)

    return total


def kolmogorov_complexity(X: torch.Tensor) -> float:
    raw_bytes = X.numpy().tobytes()
    compressed_bytes = zlib.compress(raw_bytes, level=9)
    # Clamp to 1: zlib adds a header so compressed can exceed raw for small/random data
    return min(len(compressed_bytes), len(raw_bytes)) / len(raw_bytes)


# With LLM, TODO verify in detail, but seems to work at first glance
def topological_complexity(
    X: torch.Tensor,
    max_dim: int = 2,
    weights: list[float] | None = None,
    max_samples: int | None = 1000,
) -> float:
    if weights is None:
        weights = [1.0] * (max_dim + 1)
    if len(weights) != max_dim + 1:
        raise ValueError(f"weights must have length max_dim+1={max_dim + 1}")

    points = X.detach().numpy()
    if max_samples is not None and len(points) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=max_samples, replace=False)
        points = points[idx]

    # Geometric normalisation: scale points so dataset diameter is 1.
    # This makes persistence lifetimes directly comparable across datasets.
    if len(points) > 1:
        max_dist = float(sp.spatial.distance.pdist(points).max())
        if max_dist > 0.0:
            points = points / max_dist

    diagrams = ripser(points, maxdim=max_dim)["dgms"]

    all_lifetimes = []
    for dgm in diagrams:
        finite_mask = np.isfinite(dgm[:, 1])
        all_lifetimes.append(dgm[finite_mask, 1] - dgm[finite_mask, 0])

    total = 0.0
    for k, lifetimes in enumerate(all_lifetimes):
        total += weights[k] * float(np.sum(lifetimes))

    return total


# New metric wasserstein distance
def dataset_wasserstein(X: np.ndarray, y: np.ndarray, **kw) -> float:
    """Compute average pairwise 1-Wasserstein distance between all class pairs.

    For binary: returns W1(pos, neg).
    For multiclass: returns mean W1 over all (class_i, class_j) pairs.
    """
    classes = np.unique(y)

    # Binary case (backward compatible)
    if len(classes) == 2:
        y_pos, y_neg = classes[0], classes[1]
        return wasserstein1_l1(X[y == y_pos], X[y == y_neg], **kw)

    # Multiclass: average pairwise distances
    w1_distances = []
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1 :]:
            w1 = wasserstein1_l1(X[y == c1], X[y == c2], **kw)
            w1_distances.append(w1)

    return float(np.mean(w1_distances))


# Induced, MerLin gives states not density matrix (easier to compute for some metrics)
############################################################################################################


def _safe_hermitian_eigvals(rho: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(rho).all():
        raise ValueError("Density matrix contains non-finite entries")
    rho = (rho + rho.conj().mT) / 2
    try:
        return torch.linalg.eigvalsh(rho)
    except RuntimeError:
        try:
            rho_cpu = rho.detach().cpu().to(torch.complex128)
            return torch.linalg.eigvalsh(rho_cpu).to(rho.dtype)
        except RuntimeError:
            jitter = 1e-10 * torch.eye(rho.size(-1), dtype=rho.dtype, device=rho.device)
            return torch.linalg.eigvalsh(rho + jitter)


def hilbert_space_support_dim(
    x: torch.Tensor,
    embedder: nn.Module | NeuralEmbeddingMerLinKernel,
    eps: float = 1e-8,
) -> int:
    """
    Per the kernel effective dim of p.8 eq 12
    """
    if isinstance(embedder, NeuralEmbeddingMerLinKernel):

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.classical_model = embedder.classical_encoder

            def forward(self, x: torch.Tensor):
                params = self.classical_model(x)
                with torch.no_grad():
                    output = assign_params(embedder.quantum_embedding_layer, params)
                return output

        encoder = Encoder()
    else:
        encoder = embedder
    psis = encoder(x)
    rhos = state_vector_to_density_matrix(psis)
    rho = torch.sum(rhos, dim=0) / x.size(0)
    if not torch.isfinite(rho).all():
        return float(min(x.size(0), rho.size(0)))
    try:
        eigvals = _safe_hermitian_eigvals(rho)
    except RuntimeError:
        purity = torch.real(torch.trace(rho @ rho))
        if torch.isfinite(purity) and purity > 0:
            return float(1.0 / purity)
        return float(min(x.size(0), rho.size(0)))

    effective_dim = 0.0
    for val in eigvals:
        effective_dim += float(val) / (float(val) + eps)
    return effective_dim


def quantum_fisher_information_spread(
    x: torch.Tensor, embedder: nn.Module | NeuralEmbeddingMerLinKernel
) -> float:
    # Assumed QFI is the same as QFI spread
    if isinstance(embedder, NeuralEmbeddingMerLinKernel):

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.classical_model = embedder.classical_encoder

            def forward(self, x: torch.Tensor):
                params = self.classical_model(x)
                with torch.no_grad():
                    output = assign_params(embedder.quantum_embedding_layer, params)
                return output

        encoder = Encoder()
    else:
        encoder = embedder
    print("Getting the quantum fisher matrice")
    # Consume the generator lazily — only one FIM matrix is in memory at a time
    return sum(
        torch.trace(mat) for mat in get_quantum_fisher_matrices(encoder, x)
    ) / x.size(0)


def entanglement_entropy(
    x: torch.Tensor, embedder: nn.Module | NeuralEmbeddingMerLinKernel
) -> float:

    disable_tqdm = not sys.stdout.isatty()

    if isinstance(embedder, NeuralEmbeddingMerLinKernel):
        is_dual_rail = (
            embedder.quantum_embedding_layer.computation_space
            is ml.ComputationSpace.DUAL_RAIL
        )
        num_modes = embedder.quantum_embedding_layer.circuit.m
        num_photons = embedder.quantum_embedding_layer.n_photons
        state_keys = embedder.quantum_embedding_layer.output_keys

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.classical_model = embedder.classical_encoder

            def forward(self, x: torch.Tensor):
                params = self.classical_model(x)
                with torch.no_grad():
                    return assign_params(embedder.quantum_embedding_layer, params)

        encoder = Encoder()

    else:
        comp_space_to_str = {
            ml.ComputationSpace.DUAL_RAIL: "dual_rail",
            ml.ComputationSpace.FOCK: "fock",
            ml.ComputationSpace.UNBUNCHED: "unbunched",
        }

        is_dual_rail = embedder.computation_space is ml.ComputationSpace.DUAL_RAIL
        num_modes = embedder.num_modes
        num_photons = embedder.num_photons

        state_keys = ml.Combinadics(
            scheme=comp_space_to_str[embedder.computation_space],
            n=num_photons,
            m=num_modes,
        ).enumerate_states()

        encoder = embedder

    # Hilbert space structure
    if is_dual_rail:
        dim_per_state = 2
        num_states = num_modes // 2
    else:
        dim_per_state = num_photons + 1
        num_states = num_modes

    bipartitions = _get_all_bipartitions(num_states)
    num_bipartitions = len(bipartitions)

    total_entropy = 0.0

    for point in tqdm(x, desc="[entanglement_entropy] points", disable=disable_tqdm):

        psi = encoder(point)

        point_entropy = 0.0

        if is_dual_rail:
            rho = state_vector_to_density_matrix(psi)

            for bipartition in tqdm(
                bipartitions,
                desc="[entanglement_entropy] bipartitions",
                leave=False,
                disable=disable_tqdm,
            ):

                bip_to_use = (
                    bipartition[0]
                    if len(bipartition[0]) > len(bipartition[1])
                    else bipartition[1]
                )

                point_entropy += quantum_entropy(
                    partial_trace_from_density(
                        rho,
                        states_to_trace=bip_to_use,
                        dim_per_state=dim_per_state,
                    )
                )

        else:
            for bipartition in tqdm(
                bipartitions,
                desc="[entanglement_entropy] bipartitions",
                leave=False,
                disable=disable_tqdm,
            ):

                M_to_SVD = create_correlation_matrix_bipartition(
                    psi,
                    bipartition,
                    n_photons=num_photons,
                    state_keys=state_keys,
                )

                # Dense LAPACK SVD is faster for these matrix sizes
                if sp.sparse.issparse(M_to_SVD):
                    M_to_SVD = M_to_SVD.toarray()

                schmidt_values = np.linalg.svd(
                    M_to_SVD,
                    compute_uv=False,
                )

                entropy_values = schmidt_values**2

                # Remove numerical zeros
                entropy_values = entropy_values[entropy_values > 1e-12]

                point_entropy += -np.sum(entropy_values * np.log(entropy_values))

        total_entropy += point_entropy / num_bipartitions

    return total_entropy / x.size(0)


def kernel_spectrum_flatness(
    x: torch.Tensor, embedder: nn.Module | NeuralEmbeddingMerLinKernel
) -> float:
    if isinstance(embedder, NeuralEmbeddingMerLinKernel):
        kernel_matrix = embedder.compute_kernel_matrix(x)
    else:
        kernel_matrix = compute_kernel_matrix_without_nqe(x, embedder)

    eigvals = torch.linalg.eigvalsh(kernel_matrix)
    eigvals_square = eigvals**2
    result = (torch.sum(eigvals) ** 2) / torch.sum(eigvals_square)
    for _ in tqdm(range(1), desc="[kernel_spectrum_flatness] eigvals"):
        pass
    return result


def locality_vs_expressibility(
    x: torch.Tensor,
    embedder: nn.Module | NeuralEmbeddingMerLinKernel,
    n_samples: int = 1000,
    n_bins: int = 50,
    ee: float | None = None,
) -> float:
    """
    The x must already be pretreated or be pretreated in the embedder
    """

    # A more expressive embedding is more complex, so invert the score
    print("[locality_vs_expressibility] Computing kl_div...")
    expressibility_score = kl_div(x, embedder, n_samples=n_samples, n_bins=n_bins)

    # TODO Check with the author if the locality is as expected
    print("[locality_vs_expressibility] Computing entanglement_entropy...")
    if ee is None:
        avg_entropy = entanglement_entropy(x, embedder)
    else:
        avg_entropy = ee

    # Find the dimension N
    if isinstance(embedder, NeuralEmbeddingMerLinKernel):
        if (
            embedder.quantum_embedding_layer.computation_space
            is ml.ComputationSpace.DUAL_RAIL
        ):
            N = 2 ** (embedder.quantum_embedding_layer.circuit.m // 2)
        else:
            N = (
                embedder.quantum_embedding_layer.n_photons + 1
            ) ** embedder.quantum_embedding_layer.circuit.m
    else:
        n_photons = (
            embedder.n_photons
            if hasattr(embedder, "n_photons")
            else embedder.num_photons
        )
        n_modes = (
            embedder.circuit.m if hasattr(embedder, "circuit") else embedder.num_modes
        )
        if embedder.computation_space is ml.ComputationSpace.DUAL_RAIL:
            N = 2 ** (n_modes // 2)
        else:
            N = (n_photons + 1) ** n_modes

    for _ in tqdm(range(1), desc="[locality_vs_expressibility] final computation"):
        pass
    return (np.exp(-expressibility_score) - (2 * avg_entropy / np.log(N))) ** 2


def topological_invariants_of_embedding(
    x: torch.Tensor,
    embedder: nn.Module | NeuralEmbeddingMerLinKernel,
    max_dim: int = 2,
    weights: list[float] | None = None,
    max_samples: int | None = 1000,
) -> float:
    # TODO Verify my choice with author, use the kernel distance to calculate the persistent homology
    if weights is None:
        weights = [1.0] * (max_dim + 1)
    if len(weights) != max_dim + 1:
        raise ValueError(f"weights must have length max_dim+1={max_dim + 1}")

    if max_samples is not None and x.size(0) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(x.size(0), size=max_samples, replace=False)
        x = x[idx]

    if isinstance(embedder, NeuralEmbeddingMerLinKernel):
        kernel_matrix = embedder.compute_kernel_matrix(x).detach().cpu().numpy()
    else:
        kernel_matrix = (
            compute_kernel_matrix_without_nqe(x, embedder).detach().cpu().numpy()
        )

    print("[topological_invariants_of_embedding] Computing persistent diagrams...")
    # Convert similarity matrix to distance matrix: d[i,j] = 1 - k[i,j]
    # (fidelity=1 → distance=0 for identical states; fidelity=0 → distance=1 for orthogonal states)
    distance_matrix = 1.0 - kernel_matrix
    del kernel_matrix
    diagrams = ripser(distance_matrix, maxdim=max_dim, distance_matrix=True)["dgms"]
    del distance_matrix

    total = 0.0
    for k, dgm in enumerate(
        tqdm(diagrams, desc="[topological_invariants_of_embedding] diagrams")
    ):
        finite_mask = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        total += weights[k] * float(np.sum(lifetimes))

    return total


def encoded_states_classes_overlap(
    x: torch.Tensor,
    y: torch.Tensor,
    embedder: nn.Module | NeuralEmbeddingMerLinKernel,
    distance: str = "Trace",
) -> float:

    if isinstance(embedder, NeuralEmbeddingMerLinKernel):

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.classical_model = embedder.classical_encoder

            def forward(self, x: torch.Tensor):
                params = self.classical_model(x)
                with torch.no_grad():
                    output = assign_params(embedder.quantum_embedding_layer, params)
                return output

        encoder = Encoder()

    else:
        encoder = embedder

    num_classes = int(torch.max(y).item()) + 1
    possible_combinations = list(combinations(list(i for i in range(num_classes)), 2))
    rhos = []

    X_splits = []
    for class_index in tqdm(
        range(num_classes), desc="[encoded_states_classes_overlap] splitting classes"
    ):
        X_splits.append(
            torch.stack([x[i] for i in range(len(x)) if y[i] == class_index])
        )

    for class_index in tqdm(
        range(num_classes), desc="[encoded_states_classes_overlap] computing rhos"
    ):
        # Training states
        states = encoder(X_splits[class_index])
        total_rhos = state_vector_to_density_matrix(states)
        rhos.append(torch.sum(total_rhos, dim=0) / len(X_splits[class_index]))

    overlaps = 0
    for comb in tqdm(
        possible_combinations,
        desc="[encoded_states_classes_overlap] computing overlaps",
    ):
        overlaps += 1 - calculate_distance(
            rhos[comb[0]],
            rhos[comb[1]],
            distance=distance,
        )
    return overlaps / len(possible_combinations)


# Quantum
############################################################################################################


# TODO Optimize just like the induced
def average_bipartite_entanglement_entropy(
    x: torch.Tensor,
    computation_space: ml.ComputationSpace,
    state_keys: list[tuple[int]],
    n_modes: int,
    n_photons: int | None = None,
) -> float:

    # Get data for the partial trace
    if computation_space is ml.ComputationSpace.DUAL_RAIL:
        dim_per_state = 2
        num_states = n_modes // 2
    else:
        if n_photons is None:
            raise ValueError(
                "The number of photons must be given if the computation space is not DUAL_RAIL"
            )
        dim_per_state = n_photons + 1
        num_states = n_modes
    bipartitions = _get_all_bipartitions(num_states)
    num_bipartitions = len(bipartitions)

    total_entropy = 0

    for point in x:
        # Getting the density matrix in the right encoding
        if computation_space is ml.ComputationSpace.DUAL_RAIL:
            rho = point
        else:
            rho = embbed_density_into_complete_fock_space(
                point,
                n_modes,
                n_photons=n_photons,
                state_keys=state_keys,
            )

        # Computing the entropy per bipartition
        point_entropy = 0
        for bipartition in bipartitions:
            bip_to_use = (
                bipartition[0]
                if len(bipartition[0]) > len(bipartition[1])
                else bipartition[1]
            )
            reduced = partial_trace_from_density(
                rho, states_to_trace=bip_to_use, dim_per_state=dim_per_state
            )
            # Ensure shape is tuple of ints
            shape = tuple(int(s) for s in reduced.shape)
            reduced = reduced.reshape(shape)
            point_entropy += quantum_entropy(reduced)
        total_entropy += point_entropy / num_bipartitions

    return total_entropy / x.size(0)


def multipartite_total_correlation(
    x: torch.Tensor,
    num_subsystem: int,
    dim_per_state: int = 2,
    fock_space: bool = True,
    num_photons: int | None = None,
    state_keys: list[tuple[int]] | None = None,
) -> float:
    rho = torch.sum(x, dim=0) / x.size(0)
    if fock_space:
        if (num_photons is None) or (state_keys is None):
            raise ValueError(
                "When fock_space is True, the number of photons and the state keys must be provided"
            )
        rho = embbed_density_into_complete_fock_space(
            rho, n_modes=num_subsystem, n_photons=num_photons, state_keys=state_keys
        )
    subsystems = list(i for i in range(num_subsystem))
    correlation = 0
    for i in subsystems:
        if i == subsystems[0]:
            to_trace = subsystems[1:]
        elif i == subsystems[-1]:
            to_trace = subsystems[:-1]
        else:
            to_trace = subsystems[:i]
            b = subsystems[i + 1 :]
            to_trace.extend(b)
        traced_density = partial_trace_from_density(
            deepcopy(rho), to_trace, dim_per_state=dim_per_state
        )
        correlation += quantum_entropy(traced_density)

    return correlation - quantum_entropy(rho)


def effective_kernel_rank(x: torch.Tensor) -> float:
    kernel_matrix = get_kernel_matrix(x)
    eigvals = torch.linalg.eigvalsh(kernel_matrix)
    eigvals = eigvals.flatten()  # Ensure 1D
    eigvals_square = eigvals**2
    return (torch.sum(eigvals) ** 2) / torch.sum(eigvals_square)


##### TODO
def nonclassicality(x: torch.Tensor) -> float:
    # TODO what are Clifford stabilizers in photonics
    pass


##### TODO
def quantum_fisher_information(x: torch.Tensor) -> float:
    # TODO what are the parameters
    pass


##### TODO
def topological_quantum_complexity(
    x: torch.Tensor,
    max_dim: int = 2,
    weights: list[float] | None = None,
    gamma_1: float = 0.33,
    gamma_2: float = 0.33,
    gamma_3: float = 0.33,
) -> float:
    ### S_topo
    s_topo = None

    #### Euler_carac
    euler_carac = 0

    #### Pers
    # TODO Verify my choice with author, use the kernel distance to calculate the persistent homology
    if weights is None:
        weights = [1.0] * (max_dim + 1)
    if len(weights) != max_dim + 1:
        raise ValueError(f"weights must have length max_dim+1={max_dim + 1}")

    kernel_matrix = get_kernel_matrix(x)

    diagrams = ripser(kernel_matrix, maxdim=max_dim, distance_matrix=True)["dgms"]

    pers = 0.0
    for k, dgm in enumerate(diagrams):
        # Exclude infinite death values (unpaired features)
        # Removes features that are not dead
        finite_mask = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        pers += weights[k] * float(np.sum(lifetimes))

    return (gamma_1 * s_topo) + (gamma_2 * euler_carac) + (gamma_3 * pers)


# general utils
############################################################################################################
def calculate_distance(
    rho0: torch.Tensor, rho1: torch.Tensor, distance: str = "Trace"
) -> float:
    rho_diff = rho1 - rho0
    if distance == "Trace":
        eigvals = torch.linalg.eigvals(rho_diff)
        return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
    elif distance == "Hilbert-Schmidt":
        return 0.5 * torch.trace(rho_diff @ rho_diff)
    else:
        raise ValueError("No distance with that name")


def quantum_entropy(rho: torch.Tensor) -> float:
    """
    To check but I think its legit
    """
    eigvals = torch.linalg.eigvalsh(rho)
    entropy = 0
    for val in eigvals:
        if val > 1e-8:
            entropy += val * np.log(val)
    return (-1) * entropy


def get_kernel_matrix(rhos: torch.Tensor) -> torch.Tensor:
    """
    Not efficent but legit for any density. matrix state, if we stick to pure states, it is much easier: Tr(rho_a*rho_b)
    """
    N = rhos.size(0)
    kernel_matrix = torch.empty((N, N))
    for i in range(N):
        kernel_matrix[i] = 1.0
        sqrt_A = sp.linalg.sqrtm(rhos[i])
        for j in range(i + 1, N):
            to_trace = sp.linalg.sqrtm(
                torch.matmul(sqrt_A, torch.matmul(rhos[j], sqrt_A))
            )
            kernel_matrix[i, j] = torch.trace(to_trace) ** 2
    return kernel_matrix


def embbed_density_into_complete_fock_space(
    rho: torch.Tensor,
    n_modes: int,
    n_photons: int,
    state_keys: list[tuple[int]],
):
    # Get the basis states
    basis_states = _all_photon_mode_configurations(m=n_modes, n=n_photons)
    space_size = len(basis_states)

    # Embbed the density matrix in the more physically accurate
    larger_density = torch.zeros((space_size, space_size))
    for i in range(len(state_keys)):
        for j in range(len(state_keys)):
            larger_matrix_i = basis_states.index(state_keys[i])
            larger_matrix_j = basis_states.index(state_keys[j])
            larger_density[larger_matrix_i, larger_matrix_j] = rho[i, j]

    return larger_density


def create_correlation_matrix_bipartition(
    state: torch.Tensor,
    bipartitions: tuple[list[int]],
    n_photons: int,
    state_keys: list[tuple[int, ...]],
) -> sp.sparse.csr_matrix:

    rows = [
        _fock_state_to_full_index(tuple(sk[i] for i in bipartitions[0]), n_photons)
        for sk in state_keys
    ]
    cols = [
        _fock_state_to_full_index(tuple(sk[i] for i in bipartitions[1]), n_photons)
        for sk in state_keys
    ]

    d_A = (n_photons + 1) ** len(bipartitions[0])
    d_B = (n_photons + 1) ** len(bipartitions[1])

    return sp.sparse.csr_matrix((state.flatten(), (rows, cols)), shape=(d_A, d_B))


def partial_trace_from_density(
    torch_state: torch.Tensor, states_to_trace: list[int], dim_per_state: int
) -> torch.Tensor:
    mpo = qtn.MatrixProductOperator.from_dense(
        torch_state.detach().numpy(), dims=dim_per_state
    )
    num_qubits = mpo.num_tensors
    reduced_state = mpo.trace(
        left_inds=[f"b{i}" for i in states_to_trace],
        right_inds=[f"k{i}" for i in states_to_trace],
    )
    new_num_qubits = num_qubits - len(states_to_trace)

    rho_dense = reduced_state.contract().data
    rho_torch = torch.from_numpy(rho_dense)
    rho_torch = rho_torch.reshape(
        (dim_per_state**new_num_qubits, dim_per_state**new_num_qubits)
    )
    return rho_torch


def get_quantum_fisher_matrices(
    model: nn.Module, inputs: torch.Tensor, eps: float = 1e-5
):
    """Generator: yields one QFI matrix per input point to avoid accumulating
    all matrices in memory simultaneously."""
    for point in inputs:
        d = point.numel()
        with torch.no_grad():
            psi = model(point).to(torch.complex128).flatten()

        # Numerical Jacobian via central finite differences — works for all
        # encoders regardless of whether their forward pass is differentiable
        derivatives = torch.zeros((psi.numel(), d), dtype=torch.complex128)
        for k in range(d):
            delta = torch.zeros_like(point)
            delta.flatten()[k] = eps
            with torch.no_grad():
                psi_plus = model(point + delta).to(torch.complex128).flatten()
                psi_minus = model(point - delta).to(torch.complex128).flatten()
            derivatives[:, k] = (psi_plus - psi_minus) / (2 * eps)
            del psi_plus, psi_minus

        fim = torch.zeros((d, d), dtype=torch.float64)
        # Per https://arxiv.org/pdf/1907.08037 p.10
        for i in range(d):
            dpsi_i = derivatives[:, i]
            for j in range(d):
                dpsi_j = derivatives[:, j]
                fim[i, j] = 4 * torch.real(
                    torch.vdot(dpsi_i, dpsi_j)
                    - (torch.vdot(dpsi_i, psi) * torch.vdot(psi, dpsi_j))
                )
        del derivatives, psi
        yield fim


# CO-WRITTEN with LLM to understand what is computed
def kl_div(
    x: torch.Tensor,
    embedder: ml.QuantumLayer | NeuralEmbeddingMerLinKernel,
    n_samples: int = 1000,
    n_bins: int = 50,
):
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values

    # Sample random inputs uniformly over the observed input range
    rand_inputs = torch.rand(n_samples, x.size(1)) * (x_max - x_min) + x_min

    # Get the fidelities with the Kernel matrix
    if isinstance(embedder, NeuralEmbeddingMerLinKernel):
        kernel_matrix = embedder.compute_kernel_matrix(rand_inputs)
        D = embedder.quantum_embedding_layer.output_size
    else:
        kernel_matrix = compute_kernel_matrix_without_nqe(x, embedder)
        D = embedder.output_size

    # Ensure kernel_matrix is 2D and extract upper triangle as 1D
    idx = torch.triu_indices(n_samples, n_samples, offset=1)
    fidelities = kernel_matrix[idx[0], idx[1]].detach().float().numpy().flatten()

    # Get discrete probability distribution over n_bins
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    p = hist / (hist.sum() + 1e-12)

    # Calculating the probabilities of each bin for the Haar distributed states
    haar_pdf = (D - 1) * (1 - bin_centers) ** (D - 2)
    q = haar_pdf * bin_width
    q = q / (q.sum() + 1e-12)

    # Floor q to avoid silently ignoring bins where p>0 but q→0
    # (e.g. high-fidelity bins for Haar with D>2: (D-1)(1-f)^(D-2) → 0 near f=1).
    # Without this, a non-expressive encoder that concentrates fidelities near 1
    # gets an artificially low KL divergence because those bins are masked out.
    # One "virtual count" worth of probability is a natural resolution floor.
    eps_q = 1.0 / (n_bins * n_samples)
    q_safe = np.maximum(q, eps_q)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe[mask])))


######################################
"""
Here onwards are quick utility function that copilot generated,
TODO Test them in details
"""


def _all_photon_mode_configurations(m: int, n: int):
    return list(product(range(n + 1), repeat=m))


def _fock_state_to_full_index(state: tuple[int, ...], n_photons: int) -> int:
    """Convert a mode-occupation tuple to its lexicographic index in
    product(range(n_photons+1), repeat=m) without materialising that list.
    Treats occupations as digits in base (n_photons+1)."""
    idx = 0
    for occ in state:
        idx = idx * (n_photons + 1) + occ
    return idx


def _get_all_bipartitions(m: int) -> list[tuple[list[int]]]:
    states = set(range(m))
    others = list(range(1, m))

    result = []

    for r in range(m):
        for combo in combinations(others, r):
            A = set(combo) | {0}
            B = states - A

            if not B:
                continue
            result.append(tuple([A, B]))
    return result


def wasserstein1_l1(
    X_pos: np.ndarray, X_neg: np.ndarray, max_per_class: int = 500, seed: int = 0
) -> float:
    """Empirical 1-Wasserstein distance with L1 ground metric between two point clouds.

    Uses scipy's linear_sum_assignment for exact EMD (Hungarian algorithm). This is
    a polynomial algorithm. It is ok since all the points have the same points but different distances.
    The problem becomes an assignement problem.
    No threading issues, pure Python implementation.
    """
    rng = np.random.default_rng(seed)
    if len(X_pos) > max_per_class:
        X_pos = X_pos[rng.choice(len(X_pos), max_per_class, replace=False)]
    if len(X_neg) > max_per_class:
        X_neg = X_neg[rng.choice(len(X_neg), max_per_class, replace=False)]

    # Compute L1 cost matrix
    M = np.abs(X_pos[:, np.newaxis, :] - X_neg[np.newaxis, :, :]).sum(axis=2)

    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(M)

    # Compute transport cost
    transport_cost = M[row_ind, col_ind].sum()

    # Normalize by number of samples (average cost per sample)
    return float(transport_cost / max(len(X_pos), len(X_neg)))
