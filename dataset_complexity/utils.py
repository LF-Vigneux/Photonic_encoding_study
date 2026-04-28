"""
Not implemented techniques are identified by #####TODO
"""

import numpy as np
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
        print(prob * np.log(prob))
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
    return len(compressed_bytes) / len(raw_bytes)


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
    diagrams = ripser(points, maxdim=max_dim)["dgms"]

    total = 0.0
    for k, dgm in enumerate(diagrams):
        # Exclude infinite death values (unpaired features)
        # Removes features that are not dead
        finite_mask = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        total += weights[k] * float(np.sum(lifetimes))

    return total


# Induced, MerLin gives states not density matrix (easier to compute for some metrics)
############################################################################################################


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
                self.classical_model = embedder.classical_model

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
    eigvals = torch.linalg.eigvalsh(rho)
    effective_dim = 0
    for val in eigvals:
        effective_dim += val / (val + eps)
    return effective_dim


def quantum_fisher_information_spread(
    x: torch.Tensor, embedder: nn.Module | NeuralEmbeddingMerLinKernel
) -> float:
    # Assumed QFI is the same as QFI spread
    if isinstance(embedder, NeuralEmbeddingMerLinKernel):

        class Encoder(nn.Module):
            def __init__(self):
                self.classical_model = embedder.classical_model

            def forward(self, x: torch.Tensor):
                params = self.classical_model(x)
                with torch.no_grad():
                    output = assign_params(embedder.quantum_embedding_layer, params)
                return output

        encoder = Encoder()
    else:
        encoder = embedder

    matricies = get_quantum_fisher_matrices(encoder, x)
    return sum(torch.trace(mat) for mat in matricies) / x.size(0)


def entanglement_entropy(
    x: torch.Tensor, embedder: nn.Module | NeuralEmbeddingMerLinKernel
) -> float:

    if isinstance(embedder, NeuralEmbeddingMerLinKernel):
        layer_object = embedder.quantum_embedding_layer
    else:
        layer_object = embedder

    # The same as average_bipartite_entanglement_entropy, I think
    state_keys = layer_object.output_keys

    # Get data for the partial trace
    if layer_object.computation_space is ml.ComputationSpace.DUAL_RAIL:
        dim_per_state = 2
        num_states = layer_object.circuit.m // 2
    else:
        dim_per_state = layer_object.n_photons + 1
        num_states = layer_object.circuit.m
    bipartitions = _get_all_bipartitions(num_states)
    num_bipartitions = len(bipartitions)

    total_entropy = 0

    for point in x:
        # Getting the density matrix in the right encoding
        if layer_object.computation_space is ml.ComputationSpace.DUAL_RAIL:
            psi = embedder(point)
            rho = state_vector_to_density_matrix(psi)

        else:
            psi = embbed_density_into_complete_fock_space(
                embedder(point),
                layer_object.circuit.m,
                n_photons=layer_object.n_photons,
                state_keys=state_keys,
            )
            rho = state_vector_to_density_matrix(psi)

        # Computing the entropy per bipartition
        point_entropy = 0
        for bipartition in bipartitions:
            point_entropy += quantum_entropy(
                partial_trace_from_density(
                    rho, states_to_trace=bipartition, dim_per_state=dim_per_state
                )
            )
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
    return (torch.sum(eigvals) ** 2) / torch.sum(eigvals_square)


def locality_vs_expressibility(
    x: torch.Tensor,
    embedder: nn.Module | NeuralEmbeddingMerLinKernel,
    n_samples: int = 1000,
    n_bins: int = 50,
) -> float:
    """
    The x must already be pretreated or be pretreated in the embedder
    """

    # A more expressive embedding is more complex, so invert the score
    expressibility_score = kl_div(x, embedder, n_samples=n_samples, n_bins=n_bins)

    # TODO Check with the author, locality proxy, the entanglement entropy
    avg_entropy = entanglement_entropy(x, embedder)

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
        if embedder.computation_space is ml.ComputationSpace.DUAL_RAIL:
            N = 2 ** (embedder.circuit.m // 2)
        else:
            N = (embedder.n_photons + 1) ** embedder.circuit.m

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
        kernel_matrix = embedder.compute_kernel_matrix(x)
    else:
        kernel_matrix = compute_kernel_matrix_without_nqe(x, embedder)

    diagrams = ripser(kernel_matrix, maxdim=max_dim, distance_matrix=True)["dgms"]

    total = 0.0
    for k, dgm in enumerate(diagrams):
        # Exclude infinite death values (unpaired features)
        # Removes features that are not dead
        finite_mask = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        total += weights[k] * float(np.sum(lifetimes))

    return total


# Quantum
############################################################################################################


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
            point_entropy += quantum_entropy(
                partial_trace_from_density(
                    rho, states_to_trace=bipartition, dim_per_state=dim_per_state
                )
            )
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
    for i in enumerate(len(state_keys)):
        for j in enumerate(len(state_keys)):
            larger_matrix_i = basis_states.index(state_keys[i])
            larger_matrix_j = basis_states.index(state_keys[j])
            larger_density[larger_matrix_i, larger_matrix_j] = rho[i, j]

    return rho


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
    model: nn.Module, inputs: torch.Tensor
) -> list[torch.Tensor]:
    d = sum(i.numel() for i in model.parameters())
    matricies = []
    for point in inputs:
        psi = model(point)
        model
        fim = torch.empty((d, d))
        derivatives = torch.autograd.functional.jacobian(
            model, point, create_graph=False
        )
        # Per https://arxiv.org/pdf/1907.08037 p.10
        for i in range(d):
            dpsi_i = derivatives[:, i]
            for j in range(d):
                dpsi_j = derivatives[:, j]
                fim[i, j] = 4 * torch.real(
                    torch.vdot(dpsi_i, dpsi_j)
                    - (torch.vdot(dpsi_i, psi) * torch.vdot(psi, dpsi_j))
                )
        matricies.append(fim)

    return matricies


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

    idx = torch.triu_indices(n_samples, n_samples, offset=1)
    fidelities = kernel_matrix[idx[0], idx[1]].detach().float().numpy()

    # Get discrete probability distribution over n_bins
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    p = hist / (hist.sum() + 1e-12)

    # Calculating the probabilities of each bin for the Haar distributed states
    haar_pdf = (D - 1) * (1 - bin_centers) ** (D - 2)
    q = haar_pdf * bin_width
    q = q / (q.sum() + 1e-12)

    # make sure that no negative probability is being used
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


######################################
"""
Here onwards are quick utility function that copilot generated,
TODO Test them in details
"""


def _all_photon_mode_configurations(m: int, n: int):
    return list(product(range(n + 1), repeat=m))


def _get_all_bipartitions(m: int) -> list[list[int]]:
    states = set(range(m))
    others = list(range(1, m))

    result = []

    for r in range(m):
        for combo in combinations(others, r):
            A = set(combo) | {0}
            B = states - A

            if not B:
                continue

            # keep only the larger subset
            if len(A) >= len(B):
                result.append(sorted(A))
            else:
                result.append(sorted(B))

    return result
