import torch
import numpy as np
import perceval as pcvl


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
