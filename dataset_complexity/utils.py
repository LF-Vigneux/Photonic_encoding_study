import numpy as np
import torch
import torch.nn as nn
import merlin as ml
import scipy as sp
from itertools import product, combinations
import quimb.tensor as qtn
from nn_embedding.lib.merlin_based_model import (
    NeuralEmbeddingMerLinKernel,
    NeuralEmbeddingMerLinModel,
)
from nn_embedding.utils.utils import (
    TransparentModel,
    assign_params,
    state_vector_to_density_matrix,
)


# Classical
############################################################################################################
def distributional_entropy(X: torch.Tensor) -> float:
    N = X.size(0)
    unique_tensors = torch.unique(X, dim=0)
    if unique_tensors.size(0) == N:
        return np.log(N)

    frequencies = np.zeros(unique_tensors.size(0))
    for tensor in X:
        print(tensor)
        for index, tensor_u in enumerate(unique_tensors):
            if torch.allclose(tensor_u, tensor, rtol=1e-09):
                frequencies[index] += 1
                print(index)
                break

    if not np.sum(frequencies) == N:
        raise ValueError(
            "The sum of freqencies did not match the total amount of elements in X"
        )

    probs = frequencies / N
    entropy = 0
    for prob in probs:
        entropy += prob * np.log(prob)
        print(prob * np.log(prob))
    return (-1) * entropy


def correlation_order(X: torch.Tensor) -> float:
    # How to implement it efficiently, factorial (entropy per possibile subset of features)
    pass


def kolmogorov_complexity(X: torch.Tensor) -> float:
    # How to find the legnt of representation in bits and find the optimal lossless compression
    pass


def topological_complexity(X: torch.Tensor) -> float:
    pass


# Induced, MerLin gives states not density matrix (easier to compute for some metrics)
############################################################################################################


def hilbert_space_support_dim(
    x: torch.Tensor,
    embedder: ml.QuantumLayer | NeuralEmbeddingMerLinModel,
    eps: float = 1e-8,
) -> int:
    """
    Per the kernel effective dim of p.8 eq 12
    """
    psis = embedder(x)
    rhos = state_vector_to_density_matrix(psis)
    rho = torch.sum(rhos, dim=0) / x.size(0)
    eigvals = torch.linalg.eigvalsh(rho)
    effective_dim = 0
    for val in eigvals:
        effective_dim += val / (val + eps)
    return effective_dim


def quantum_fisher_information_spread(
    x: torch.Tensor, embedder: ml.QuantumLayer | NeuralEmbeddingMerLinModel
) -> float:
    if isinstance(embedder, NeuralEmbeddingMerLinModel):

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
    x: torch.Tensor, embedder: ml.QuantumLayer | NeuralEmbeddingMerLinModel
) -> float:

    if isinstance(embedder, NeuralEmbeddingMerLinModel):
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
    x: torch.Tensor, embedder: ml.QuantumLayer | NeuralEmbeddingMerLinModel
) -> float:
    if isinstance(embedder, NeuralEmbeddingMerLinModel):
        kernel_object = NeuralEmbeddingMerLinKernel(
            embedder.classical_encoder, embedder.quantum_embedding_layer
        )
    else:
        # TODO Will be ok once I change the method for input parameters instead of trainable ones
        kernel_object = NeuralEmbeddingMerLinKernel(TransparentModel(), embedder)

    kernel_matrix = kernel_object.compute_kernel_matrix(x)
    eigvals = torch.linalg.eigvalsh(kernel_matrix)
    eigvals_square = eigvals**2
    return (torch.sum(eigvals) ** 2) / torch.sum(eigvals_square)


def expressibility_vs_locality_ratio(
    x: torch.Tensor, embedder: ml.QuantumLayer | NeuralEmbeddingMerLinModel
) -> float:
    pass


def topological_invariants_of_embedding(
    x: torch.Tensor, embedder: ml.QuantumLayer | NeuralEmbeddingMerLinModel
) -> float:
    pass


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


def multipartite_total_correlation(x: torch.Tensor) -> float:
    pass


def effective_kernel_rank(x: torch.Tensor) -> float:
    kernel_matrix = get_kernel_matrix(x)
    eigvals = torch.linalg.eigvalsh(kernel_matrix)
    eigvals_square = eigvals**2
    return (torch.sum(eigvals) ** 2) / torch.sum(eigvals_square)


def nonclassity(x: torch.Tensor) -> float:
    pass


def quantum_fisher_information(x: torch.Tensor) -> float:
    # TODO what are the parameters
    pass


def topological_quantum_complexity(x: torch.Tensor) -> float:
    pass


# general utils
############################################################################################################
def quantum_entropy(rho: torch.Tensor) -> float:
    """
    To check but I think its legit
    """
    eigvals = torch.linalg.eigvalsh(rho)
    entropy = 0
    for val in eigvals:
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
