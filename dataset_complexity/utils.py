import numpy as np
import torch
import merlin as ml
import scipy as sp
from itertools import product, combinations
import quimb.tensor as qtn


# Classical
############################################################################################################
def distributional_entropy(X: torch.Tensor) -> float:
    N = X.size(0)
    unique_tensors = torch.unique(X, dim=0)
    if unique_tensors.size(0) == N:
        return np.log(N)

    frequencies = np.zeros(unique_tensors.size(0))
    print(unique_tensors)
    print()
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


# Induced
############################################################################################################


def hilbert_space_support_dim(
    x: torch.Tensor,
    embedder: ml.QuantumLayer,
    eps: float = 1e-8,
) -> int:
    """
    Per the kernel effective dim of p.8 eq 12
    """
    rhos = embedder(x)
    rho = torch.sum(rhos, dim=0) / x.size(0)
    eigvals = torch.linalg.eigvalsh(rho)
    effective_dim = 0
    for val in eigvals:
        effective_dim += val / (val + eps)
    return effective_dim


def quantum_fisher_information_spread(
    x: torch.Tensor, embedder: ml.QuantumLayer
) -> float:
    pass


def entanglement_entropy(x: torch.Tensor, embedder: ml.QuantumLayer) -> float:
    # The same as average_bipartite_entanglement_entropy, I think
    state_keys = embedder.output_keys

    # Get data for the partial trace
    if embedder.computation_space is ml.ComputationSpace.DUAL_RAIL:
        dim_per_state = 2
        num_states = embedder.circuit.m // 2
    else:
        dim_per_state = embedder.n_photons + 1
        num_states = embedder.circuit.m
    bipartitions = _get_all_bipartitions(num_states)
    num_bipartitions = len(bipartitions)

    total_entropy = 0

    for point in x:
        # Getting the density matrix in the right encoding
        if embedder.computation_space is ml.ComputationSpace.DUAL_RAIL:
            rho = embedder(point)

        else:
            rho = embbed_density_into_complete_fock_space(
                embedder(point),
                embedder.circuit.m,
                n_photons=embedder.n_photons,
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


def kernel_spectrum_flatness(x: torch.Tensor, embedder: ml.QuantumLayer) -> float:
    pass


def expressibility_vs_locality_ratio(
    x: torch.Tensor, embedder: ml.QuantumLayer
) -> float:
    pass


def topological_invariants_of_embedding(
    x: torch.Tensor, embedder: ml.QuantumLayer
) -> float:
    pass


# Quantum
############################################################################################################


def average_bipartite_entanglement_entropy(x: torch.Tensor) -> float:
    # Comment je peux faire une bipartition en photonique
    pass


def multipartite_total_correlation(x: torch.Tensor) -> float:
    pass


def effective_kernel_rank(x: torch.Tensor) -> float:
    pass


def nonclassity(x: torch.Tensor) -> float:
    pass


def quantum_fisher_information(x: torch.Tensor) -> float:
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


# def partial_trace(
#     rho: torch.Tensor,
#     modes_indexes: list[int],
#     computation_space: ml.ComputationSpace,
#     state_keys: list[tuple[int]],
#     n_photons: int | None = None,
#     n_modes: int | None = None,
# ) -> torch.Tensor:
#     """If it is qubits, write the qubit index

#     Here we assume litte-endian formalism: q0 tensor q1 tensor q2 ...
#     """
#     if (computation_space is not ml.ComputationSpace.DUAL_RAIL) and (
#         (n_photons is None) or (n_modes is None)
#     ):
#         assert ValueError(
#             "n_photons must be given if the computation space is not Dual Rail"
#         )
#     if computation_space is ml.ComputationSpace.DUAL_RAIL:
#         return compute_partial_trace_from_density(
#             rho, states_to_trace=modes_indexes, dim_per_state=2
#         )
#     else:
#         basic_states = _all_photon_mode_configurations(m=n_modes, n=n_photons)
#         space_size = len(basic_states)

#         # Embbed the density matrix in the more physically accurate
#         larger_density = torch.zeros((space_size, space_size))
#         for i in enumerate(len(state_keys)):
#             for j in enumerate(len(state_keys)):
#                 larger_matrix_i = basic_states.index(state_keys[i])
#                 larger_matrix_j = basic_states.index(state_keys[j])
#                 larger_density[larger_matrix_i, larger_matrix_j] = rho[i, j]

#         return compute_partial_trace_from_density(
#             larger_density, states_to_trace=None, dim_per_state=n_photons + 1
#         )


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
