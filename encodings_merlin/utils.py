import torch
import numpy as np
import perceval as pcvl
import torch.nn as nn
from math import comb
import warnings


MZI = (
    pcvl.components.BS()
    // (0, pcvl.components.PS(pcvl.Parameter("phi1")))
    // pcvl.components.BS()
    // (0, pcvl.components.PS(pcvl.Parameter("phi2")))
)


def find_upper_even_square(x: int) -> int:
    x_sqrt = np.sqrt(x)
    if int(x_sqrt) ** 2 == x_sqrt:
        return int(x_sqrt)
    new_sqrt = int(np.ceil(x_sqrt))
    new_sqrt = new_sqrt + 1 if new_sqrt % 2 == 1 else new_sqrt
    return new_sqrt**2


def vector_to_matrix_evo(
    x: torch.Tensor,
    matrix_size: int | None = None,
    symetric: bool = True,
) -> torch.Tensor:

    # Format the ourput matrix size
    num_features = x.numel()
    if matrix_size is None:
        matrix_size = num_features // 2 + 1 if symetric else num_features
    if (2 * matrix_size) - 1 < num_features:
        raise ValueError(
            "Matrix size shoud respect (2 * matrix_size) - 1 >= num_features"
        )
    if num_features > matrix_size and not symetric:
        raise ValueError(
            "For non symetric encoding, Matrix size shoud respect matrix_size >= num_features"
        )
    output_matrix = torch.zeros((matrix_size, matrix_size), dtype=x.dtype)

    # Format the diagonal values
    x_to_apply = torch.zeros(2 * matrix_size - 1, dtype=x.dtype)
    if num_features < 2 * matrix_size - 1 and symetric:
        x_to_apply[
            (x_to_apply.numel() - num_features)
            // 2 : ((x_to_apply.numel() - num_features) // 2)
            + num_features
        ] = x
    elif not symetric:
        x_to_apply = torch.zeros(2 * matrix_size - 1, dtype=x.dtype)
        x_to_apply[matrix_size - 1 : matrix_size + num_features - 1] = x
    else:
        x_to_apply = x

    # Create the matrix
    for i in range(matrix_size):
        for j in range(matrix_size):
            tag = j - i + matrix_size - 1
            output_matrix[i, j] = x_to_apply[tag]

    return output_matrix


def compute_kernel_matrix_without_nqe(
    X_data: torch.Tensor, encoding_strategy: nn.Module
):
    """Compute the symmetric kernel matrix for a dataset.

    Parameters
    ----------
    X_data : torch.Tensor
        Input dataset for which to evaluate all pairwise kernel values.
    batch_size : int, optional
        Number of upper-triangular pairs evaluated per forward pass
        (default: ``256``).

    Returns
    -------
    torch.Tensor
        Symmetric kernel matrix of shape ``(n_samples, n_samples)``.
    """
    n = X_data.size(0)
    idx_i, idx_j = torch.triu_indices(n, n)
    pairs_a = X_data[idx_i]
    pairs_b = X_data[idx_j]

    values_list = []
    for a, b in zip(pairs_a, pairs_b):
        values_list.append(torch.vdot(encoding_strategy(a), encoding_strategy(b)))

    values = torch.cat(values_list)

    output = torch.empty(n, n)
    output[idx_i, idx_j] = values
    output[idx_j, idx_i] = values
    return output


def find_mode_photon_config(
    num_features: int,
    max_modes: int = 20,
) -> tuple[int, int]:
    """
    Find (n_modes, n_photons) with smallest n_modes such that
    C(n_modes, n_photons) >= num_features and
    n_photons <= n_modes // 2.

    Parameters
    ----------
    num_features : int
        Required feature dimension.
    max_modes : int, optional
        Maximum number of modes to search.

    Returns
    -------
    tuple[int, int]
        (n_modes, n_photons) configuration.
    """
    if num_features <= 0:
        raise ValueError("num_features must be positive.")

    best = None
    for n_modes in range(1, max_modes + 1):
        for n_photons in range(1, (n_modes // 2) + 1):
            dim = comb(n_modes, n_photons)
            if dim >= num_features:
                candidate = (n_modes, n_photons)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                break
        if best is not None and best[0] == n_modes:
            break

    if best is None:
        warnings.warn(
            "System too large for simulation: no valid (n_modes, n_photons) "
            "found with max_modes=20.",
            RuntimeWarning,
            stacklevel=2,
        )
        raise ValueError("System too large for simulation with max_modes=20.")
    return best
