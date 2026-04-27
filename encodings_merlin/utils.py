import torch
import numpy as np
import perceval as pcvl
import torch.nn as nn


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
    self, X_data: torch.Tensor, encoding_strategy: nn.Module, batch_size: int = 256
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
    self.kernel_function.eval()
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


# class OneHotEncoder(nn.Module):
#     """
#     One Hot Encoder

#     Converts an image `x` to density matrix in the One Hot Amplitude
#     basis. For a given d by d image, the density matrix will be of
#     size d^2 by d^2.
#     """

#     def __init__(self, image_size: int, return_sv: bool = True):
#         super().__init__()
#         self.output_size = image_size**2
#         self.return_sv = return_sv
#         self.num_photons = 1
#         self.num_modes = 2 * image_size
#         self.computation_space = ml.ComputationSpace.UNBUNCHED

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 3:
#             x = x.unsqueeze(1)

#         norm = torch.sqrt(torch.square(torch.abs(x)).sum(dim=(1, 2, 3)))
#         x = x / norm.view(-1, 1, 1, 1)

#         # Flatten each image and multiply by transpose to get density matrix
#         x_flat = x.reshape(x.shape[0], -1)
#         if self.return_sv:
#             return x_flat
#         rho = x_flat.unsqueeze(2) @ x_flat.unsqueeze(1)
#         rho = rho.to(torch.complex128)

#         return rho

#     def __repr__(self):
#         return "OneHotEncoder()"


# class AngleEncoder(nn.Module):
#     def __init__(
#         self,
#         num_features: int,
#         num_photons: int,
#         computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
#         num_modes: int | None = None,
#         return_sv: bool = True,
#         change_output_size_even_square: bool = False,
#     ):
#         super().__init__()
#         self.num_features = num_features
#         self.num_photons = num_photons
#         self.num_modes = num_features if num_modes is None else num_modes
#         self.computation_space = computation_space

#         self.return_sv = return_sv
#         circuit = ml.CircuitBuilder(n_modes=self.num_modes)
#         features_assigned = 0
#         while features_assigned < self.num_features:
#             circuit.add_entangling_layer(trainable=True)
#             if features_assigned + circuit.n_modes < self.num_features:
#                 circuit.add_angle_encoding()
#                 features_assigned += circuit.n_modes
#             else:
#                 circuit.add_angle_encoding(
#                     modes=[i for i in range(self.num_features - features_assigned)]
#                 )
#                 features_assigned += self.num_features - features_assigned

#         self.qlayer = ml.QuantumLayer(
#             builder=circuit,
#             n_photons=num_photons,
#             measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
#             computation_space=computation_space,
#             dtype=torch.float64,
#             amplitude_encoding=False,
#         )
#         if change_output_size_even_square:
#             self.output_size = find_upper_even_square(self.qlayer.output_size)
#             self.encoder = nn.Sequential(
#                 self.qlayer, ml.LexGrouping(self.qlayer.output_size, self.output_size)
#             )
#         else:
#             self.output_size = self.qlayer.output_size
#             self.encoder = self.qlayer

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 1:
#             x = torch.unsqueeze(x, 0)
#         if x.dim() > 2:
#             x = torch.reshape(x, (x.shape[0], np.prod(x.shape[1:])))

#         amplitudes_output = self.encoder(x)

#         if self.return_sv:
#             return amplitudes_output.to(torch.complex128)
#         else:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size, self.output_size),
#                 dtype=torch.complex128,
#             )
#             for i, amplitude in enumerate(amplitudes_output):
#                 output_tensors[i, :, :] = torch.outer(amplitude, amplitude.conj())

#             return output_tensors

#     def __repr__(self):
#         return "AngleEncoder()"


# class AmplitudeEncoder(nn.Module):
#     def __init__(
#         self,
#         num_modes: int,
#         num_photons: int = 0,
#         computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
#         shuffle_amplitude: bool = False,
#         return_sv: bool = True,
#         change_output_size_even_square: bool = False,
#     ):
#         super().__init__()
#         self.num_modes = num_modes
#         self.num_photons = num_photons
#         self.computation_space = computation_space
#         self.shuffle_amplitude = shuffle_amplitude
#         self.return_sv = return_sv

#         if self.computation_space is ml.ComputationSpace.UNBUNCHED:
#             self.output_size = math.comb(self.num_modes, self.num_photons)
#         elif self.computation_space is ml.ComputationSpace.FOCK:
#             self.output_size = math.comb(
#                 self.num_modes + self.num_photons - 1, self.num_photons
#             )
#         elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
#             self.output_size = 2 ** (num_modes / 2)
#         else:
#             raise ValueError("Wrong computation space")

#         if self.shuffle_amplitude:
#             self.indexes_perm = np.arange(self.output_size)
#             np.random.shuffle(self.indexes_perm)
#         else:
#             self.indexes_perm = None

#         if change_output_size_even_square:
#             corrected_output_size = find_upper_even_square(self.output_size)
#             self.encoder = ml.LexGrouping(self.output_size, corrected_output_size)
#             self.output_size = corrected_output_size
#         else:
#             self.encoder = ml.LexGrouping(self.output_size, self.output_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         if x.dim() > 2:
#             x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

#         if self.return_sv:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size),
#                 dtype=torch.complex128,
#             )

#             for i, tensor in enumerate(x):
#                 output_tensors[i, :] = self.encoder(
#                     torch.tensor(
#                         amplitude_encoding(
#                             tensor,
#                             self.num_modes,
#                             computation_space=self.computation_space,
#                             num_photons=self.num_photons,
#                             indexes_perm=self.indexes_perm,
#                         )
#                     )
#                 )
#         else:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size, self.output_size),
#                 dtype=torch.complex128,
#             )

#             for i, tensor in enumerate(x):
#                 state = self.encoder(
#                     torch.tensor(
#                         amplitude_encoding(
#                             tensor,
#                             self.num_modes,
#                             computation_space=self.computation_space,
#                             num_photons=self.num_photons,
#                             indexes_perm=self.indexes_perm,
#                         )
#                     )
#                 )
#                 output_tensors[i, :, :] = torch.outer(state, state.conj())

#         return output_tensors

#     def __repr__(self):
#         return "AmplitudeEncoder()"


# class DenseAngleEncoder(nn.Module):
#     def __init__(
#         self,
#         num_features: int,
#         num_modes: int | None = None,
#         return_sv: bool = True,
#         change_output_size_even_square: bool = False,
#     ):
#         super().__init__()
#         width = len(str(num_features - 1))

#         self.num_features = num_features
#         self.num_modes = num_features if num_modes is None else num_modes
#         self.num_photons = int(np.ceil(self.num_modes / 2))
#         self.computation_space = ml.ComputationSpace.DUAL_RAIL
#         self.return_sv = return_sv
#         perceval_circuit = dense_angle_encoding_circuit(
#             num_features=num_features,
#             num_modes=self.num_modes,
#         )

#         input_state = [
#             1 if i % 2 == 0 else 0 for i in range(int(np.ceil(num_features / 2)) * 2)
#         ]

#         self.qlayer = ml.QuantumLayer(
#             circuit=perceval_circuit,
#             input_state=input_state,
#             measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
#             computation_space=ml.ComputationSpace.DUAL_RAIL,
#             input_parameters=[f"phi{i:0{width}d}" for i in range(num_features)],
#         )
#         if change_output_size_even_square:
#             self.output_size = find_upper_even_square(self.qlayer.output_size)
#             self.encoder = nn.Sequential(
#                 self.qlayer, ml.LexGrouping(self.qlayer.output_size, self.output_size)
#             )
#         else:
#             self.output_size = self.qlayer.output_size
#             self.encoder = self.qlayer

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         if x.dim() > 2:
#             x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

#         amplitudes_output = self.encoder(x)

#         if self.return_sv:
#             return amplitudes_output.to(torch.complex128)
#         else:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size, self.output_size),
#                 dtype=torch.complex128,
#             )
#             for i, amplitude in enumerate(amplitudes_output):
#                 output_tensors[i, :, :] = torch.outer(amplitude, amplitude.conj())

#             return output_tensors

#     def __repr__(self):
#         return "DenseAngleEncoder()"


# class DenseAmplitudeEncoder(nn.Module):
#     def __init__(
#         self,
#         num_modes: int,
#         num_photons: int = 0,
#         computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
#         shuffle_amplitude: bool = False,
#         return_sv: bool = True,
#         change_output_size_even_square: bool = False,
#     ):
#         super().__init__()
#         self.num_modes = num_modes
#         self.num_photons = num_photons
#         self.computation_space = computation_space
#         self.shuffle_amplitude = shuffle_amplitude
#         self.return_sv = return_sv

#         if self.computation_space is ml.ComputationSpace.UNBUNCHED:
#             self.output_size = math.comb(self.num_modes, self.num_photons)
#         elif self.computation_space is ml.ComputationSpace.FOCK:
#             self.output_size = math.comb(
#                 self.num_modes + self.num_photons - 1, self.num_photons
#             )
#         elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
#             self.output_size = 2 ** (num_modes / 2)
#         else:
#             raise ValueError("Wrong computation space")

#         if self.shuffle_amplitude:
#             self.indexes_perm = np.arange(self.output_size)
#             np.random.shuffle(self.indexes_perm)
#         else:
#             self.indexes_perm = None

#         if change_output_size_even_square:
#             corrected_output_size = find_upper_even_square(self.output_size)
#             self.encoder = ml.LexGrouping(self.output_size, corrected_output_size)
#             self.output_size = corrected_output_size
#         else:
#             self.encoder = ml.LexGrouping(self.output_size, self.output_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         if x.dim() > 2:
#             x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

#         if self.return_sv:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size),
#                 dtype=torch.complex128,
#             )

#             for i, tensor in enumerate(x):
#                 output_tensors[i, :] = self.encoder(
#                     torch.tensor(
#                         dense_encoding_of_features(
#                             tensor,
#                             self.num_modes,
#                             computation_space=self.computation_space,
#                             num_photons=self.num_photons,
#                             indexes_perm=self.indexes_perm,
#                         )
#                     )
#                 )
#         else:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size, self.output_size),
#                 dtype=torch.complex128,
#             )

#             for i, tensor in enumerate(x):
#                 state = self.encoder(
#                     torch.tensor(
#                         dense_encoding_of_features(
#                             tensor,
#                             self.num_modes,
#                             computation_space=self.computation_space,
#                             num_photons=self.num_photons,
#                             indexes_perm=self.indexes_perm,
#                         )
#                     )
#                 )
#                 output_tensors[i, :, :] = torch.outer(state, state.conj())

#         return output_tensors

#     def __repr__(self):
#         return "DenseAmplitudeEncoder()"


# class TimeEvolutionEncoder(nn.Module):
#     def __init__(
#         self,
#         image_size: int,
#         num_photons: int,
#         time: float = 0.1,
#         computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
#         return_sv: bool = True,
#         change_output_size_even_square: bool = False,
#         input_are_images: bool = True,
#     ):
#         """
#         image_size is one size of the image
#         """
#         super().__init__()
#         self.time = time
#         self.num_photons = num_photons
#         self.image_size = image_size
#         self.computation_space = computation_space
#         self.return_sv = return_sv
#         self.input_are_images = input_are_images
#         self.num_modes = (
#             2 * image_size if input_are_images else 2 * (image_size // 2 + 1)
#         )

#         base_circuit = ml.CircuitBuilder(n_modes=self.num_modes)
#         base_circuit.add_entangling_layer(trainable=True)
#         self.base_perceval = base_circuit.to_pcvl_circuit()
#         self.num_modes = self.base_perceval.m

#         if self.computation_space is ml.ComputationSpace.UNBUNCHED:
#             if input_are_images:
#                 self.output_size = math.comb(self.image_size * 2, self.num_photons)
#             else:
#                 self.output_size = math.comb(self.num_modes, self.num_photons)
#         elif self.computation_space is ml.ComputationSpace.FOCK:
#             if input_are_images:
#                 self.output_size = math.comb(
#                     self.num_modes + self.num_photons - 1, self.num_photons
#                 )
#             else:
#                 self.output_size = math.comb(
#                     self.num_modes + self.num_photons - 1,
#                     self.num_photons,
#                 )
#         elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
#             if input_are_images:
#                 self.output_size = 2**image_size
#             else:
#                 self.output_size = 2 ** (self.image_size // 2 + 1)
#         else:
#             raise ValueError("Wrong computation space")

#         if change_output_size_even_square:
#             corrected_output_size = find_upper_even_square(self.output_size)
#             self.encoder = ml.LexGrouping(self.output_size, corrected_output_size)
#             self.output_size = corrected_output_size
#         else:
#             self.encoder = ml.LexGrouping(self.output_size, self.output_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         if x.dim() == 2:
#             if not self.input_are_images:
#                 output_tensor = torch.empty(
#                     (x.shape[0], x.shape[1] // 2 + 1, x.shape[1] // 2 + 1)
#                 )
#                 for i, tensor in enumerate(x):
#                     output_tensor[i, :, :] = vector_to_matrix_evo(tensor)
#                 x = output_tensor
#             else:
#                 x = x.unsqueeze(0)

#         if self.return_sv:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size),
#                 dtype=torch.complex128,
#             )

#             for i, tensor in enumerate(x):
#                 total_circuit = self.base_perceval.copy()
#                 total_circuit.add(
#                     list(range(self.num_modes)),
#                     unitary_evolution(tensor, self.time),
#                 )
#                 qlayer = ml.QuantumLayer(
#                     circuit=total_circuit,
#                     n_photons=self.num_photons,
#                     measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
#                     computation_space=self.computation_space,
#                     trainable_parameters=["el_"],
#                 )
#                 output_tensors[i, :] = self.encoder(qlayer().flatten())
#         else:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size, self.output_size),
#                 dtype=torch.complex128,
#             )

#             for i, tensor in enumerate(x):
#                 total_circuit = self.base_perceval.copy()
#                 total_circuit.add(
#                     list(range(self.num_modes)),
#                     unitary_evolution(tensor, self.time),
#                 )
#                 qlayer = ml.QuantumLayer(
#                     circuit=total_circuit,
#                     n_photons=self.num_photons,
#                     measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
#                     computation_space=self.computation_space,
#                     trainable_parameters=["el_"],
#                 )
#                 state = self.encoder(qlayer().flatten())

#                 output_tensors[i, :, :] = torch.outer(state, state.conj())

#         return output_tensors

#     def __repr__(self):
#         return "TimeEvolutionEncoder()"


# class FourierEncoder(nn.Module):
#     def __init__(
#         self,
#         num_features: int,
#         n_photon_per_feature: int,
#         return_sv: bool = True,
#         change_output_size_even_square: bool = False,
#     ):
#         """
#         n_modes is one size of the image
#         """
#         super().__init__()
#         self.num_features = num_features
#         self.n_photon_per_feature = n_photon_per_feature
#         self.num_photons = num_features * n_photon_per_feature
#         self.num_modes = self.num_photons * 2

#         width = len(str((num_features) - 1))
#         self.qlayer = ml.QuantumLayer(
#             circuit=fourier_basis(
#                 num_features=num_features, num_qubits_per_feature=n_photon_per_feature
#             ),
#             n_photons=self.num_photons,
#             measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
#             computation_space=ml.ComputationSpace.DUAL_RAIL,
#             input_parameters=[f"phi{i:0{width}d}" for i in range(self.num_photons)],
#         )

#         self.computation_space = ml.ComputationSpace.DUAL_RAIL
#         self.return_sv = return_sv
#         self.output_size = 2**self.num_photons

#         if change_output_size_even_square:
#             corrected_output_size = find_upper_even_square(self.output_size)
#             self.encoder = nn.Sequential(
#                 self.qlayer, ml.LexGrouping(self.output_size, corrected_output_size)
#             )
#             self.output_size = corrected_output_size
#         else:
#             self.encoder = self.qlayer

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         if x.dim() > 2:
#             x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

#         # Repeating the input tensor to acount the scaling
#         input_tensor = torch.empty((x.shape[0], x.shape[1] * self.n_photon_per_feature))
#         k = torch.arange(self.n_photon_per_feature)
#         scales = math.pi * 2.0 ** (-k)

#         for i, input in enumerate(x):
#             out = input[:, None] * scales
#             out = out.reshape(-1)
#             input_tensor[i, :] = out

#         amplitudes_output = self.encoder(input_tensor)
#         # amplitudes_output = self.encoder(x)

#         if self.return_sv:
#             return amplitudes_output.to(torch.complex128)
#         else:
#             output_tensors = torch.empty(
#                 (x.shape[0], self.output_size, self.output_size),
#                 dtype=torch.complex128,
#             )
#             for i, amplitude in enumerate(amplitudes_output):
#                 output_tensors[i, :, :] = torch.outer(amplitude, amplitude.conj())

#             return output_tensors

#     def __repr__(self):
#         return "FourierEncoder()"


# def choose_encoding(
#     encoding_name: str,
#     return_sv: bool = True,
#     change_output_size_even_square: bool = False,
#     num_photons: int | None = None,
#     num_modes: int | None = None,
#     num_features: int | None = None,
#     time: float = 0.01,
#     computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
#     shuffle_amplitude: bool = False,
#     input_are_images: bool = True,
# ) -> tuple[
#     AngleEncoder
#     | DenseAngleEncoder
#     | AmplitudeEncoder
#     | DenseAmplitudeEncoder
#     | TimeEvolutionEncoder
#     | FourierEncoder
#     | OneHotEncoder,
#     int,
# ]:
#     if encoding_name == "OneHot":
#         return OneHotEncoder(int(np.sqrt(num_features)), return_sv=return_sv)
#     elif encoding_name == "Angle":
#         return AngleEncoder(
#             num_features=num_features,
#             num_photons=num_photons,
#             computation_space=computation_space,
#             num_modes=num_modes,
#             return_sv=return_sv,
#             change_output_size_even_square=change_output_size_even_square,
#         )
#     elif encoding_name == "DenseAngle":
#         return DenseAngleEncoder(
#             num_features=num_features,
#             num_modes=num_modes,
#             return_sv=return_sv,
#             change_output_size_even_square=change_output_size_even_square,
#         )
#     elif encoding_name == "Amplitude":
#         return AmplitudeEncoder(
#             num_photons=num_photons,
#             num_modes=num_modes,
#             computation_space=computation_space,
#             shuffle_amplitude=shuffle_amplitude,
#             return_sv=return_sv,
#             change_output_size_even_square=change_output_size_even_square,
#         )
#     elif encoding_name == "DenseAmplitude":
#         return DenseAmplitudeEncoder(
#             num_photons=num_photons,
#             num_modes=num_modes,
#             computation_space=computation_space,
#             shuffle_amplitude=shuffle_amplitude,
#             return_sv=return_sv,
#             change_output_size_even_square=change_output_size_even_square,
#         )
#     elif encoding_name == "TimeEvolution":
#         return TimeEvolutionEncoder(
#             image_size=num_features,
#             num_photons=num_photons,
#             time=time,
#             computation_space=computation_space,
#             return_sv=return_sv,
#             change_output_size_even_square=change_output_size_even_square,
#             input_are_images=input_are_images,
#         )
#     elif encoding_name == "Fourier":
#         return FourierEncoder(
#             num_features=num_features,
#             n_photon_per_feature=num_photons,
#             return_sv=return_sv,
#             change_output_size_even_square=change_output_size_even_square,
#         )
#     else:
#         raise ValueError("No encoding method associates with that name")
