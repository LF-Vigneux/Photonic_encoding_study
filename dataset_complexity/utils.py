import numpy as np
import torch
import merlin as ml
import scipy as sp


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


def hilbert_space_support_dim(x: torch.Tensor, embedder: ml.QuantumLayer) -> float:
    pass


def quantum_fisher_information_spread(
    x: torch.Tensor, embedder: ml.QuantumLayer
) -> float:
    pass


def entanglement_entropy(x: torch.Tensor, embedder: ml.QuantumLayer) -> float:
    # The same as average_bipartite_entanglement_entropy, I think
    pass


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
    To check but I thibnk its legit
    """
    eigvals = torch.linalg.eigvals(rho)
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


# X = torch.Tensor(
#     np.array(
#         [
#             [1, 2, 3, 4],
#             [5, 6, 7, 8],
#             [9, 2, 1, 0],
#             [1, 2, 3, 4],
#             [4, 3, 2, 1],
#             [11, 1, 1, 1],
#         ]
#     )
# )

# output = distributional_entropy(X)

# print(output)
