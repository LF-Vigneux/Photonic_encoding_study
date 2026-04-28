import numpy as np
import merlin as ml
import perceval as pcvl
import math
import torch
import math
import sys
from pathlib import Path
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from encodings_merlin.utils import MZI, vector_to_matrix_evo


########################################################
# Easily integrable with the NQE framework
########################################################
def angle_encoding_layer(
    num_features: int,
    num_modes: int | None = None,
    num_photons: int = 1,
    computation_space=ml.ComputationSpace.UNBUNCHED,
    input_size_0: bool = True,
) -> ml.QuantumLayer:
    if num_modes is None:
        num_modes = num_features + 1
    else:
        num_modes = max(num_modes, num_features + 1)

    circuit = ml.CircuitBuilder(n_modes=num_modes)
    circuit.add_entangling_layer(trainable=False)

    if input_size_0:
        for i in range(num_features):
            circuit.add_rotations(modes=i, trainable=True)
        return ml.QuantumLayer(
            input_size=0,
            builder=circuit,
            computation_space=computation_space,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            n_photons=num_photons,
        )
    else:
        circuit.add_angle_encoding(modes=list(i for i in range(num_features)))
        return ml.QuantumLayer(
            input_size=num_features,
            builder=circuit,
            computation_space=computation_space,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            n_photons=num_photons,
        )


def dense_angle_encoding_layer(
    num_features: int,
    param_prefix: str = "phi",
    num_modes: int | None = None,
    input_size_0: bool = True,
) -> ml.QuantumLayer:
    width = len(str(num_features - 1))
    params = [
        pcvl.Parameter(f"{param_prefix}{i:0{width}d}") for i in range(num_features)
    ]
    circuit = pcvl.Circuit(m=max(int(np.ceil(num_features / 2)) * 2, num_modes))

    mode_index = 0
    bs_angle_done = False
    for param in params:
        if not bs_angle_done:
            circuit.add([mode_index, mode_index + 1], pcvl.BS(theta=param))
            bs_angle_done = True
        else:
            circuit.add(mode_index, pcvl.PS(phi=param))
            bs_angle_done = False
            mode_index += 2
            if num_modes is not None:
                if mode_index == num_modes:
                    mode_index = 0

    if input_size_0:
        return ml.QuantumLayer(
            input_size=0,
            circuit=circuit,
            computation_space=ml.ComputationSpace.DUAL_RAIL,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            trainable_parameters=params,
            n_photons=int(np.ceil(num_features / 2)),
        )
    else:
        return ml.QuantumLayer(
            input_size=num_features,
            circuit=circuit,
            computation_space=ml.ComputationSpace.DUAL_RAIL,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            input_parameters=params,
            n_photons=int(np.ceil(num_features / 2)),
        )


def fourier_basis_layer(
    num_features: int,
    num_qubits_per_feature: int,
    input_size_0: bool = True,
) -> pcvl.Circuit:
    main_circuit = pcvl.Circuit(m=num_features * num_qubits_per_feature * 2)

    width = len(str((num_features) - 1))
    params = [
        pcvl.Parameter(f"phi{i:0{width}d}")
        for i in range(num_features * num_qubits_per_feature)
    ]
    param_index = 0
    mode_index = 0
    for _ in range(num_features):
        for _ in range(num_qubits_per_feature):
            main_circuit.add(
                [mode_index, mode_index + 1], pcvl.BS.H(phi_br=params[param_index])
            )
            param_index += 1
            mode_index += 2

    if input_size_0:
        return ml.QuantumLayer(
            input_size=0,
            circuit=main_circuit,
            computation_space=ml.ComputationSpace.DUAL_RAIL,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            trainable_parameters=params,
            n_photons=num_features * num_qubits_per_feature,
        )
    else:
        return ml.QuantumLayer(
            input_size=num_features,
            circuit=main_circuit,
            computation_space=ml.ComputationSpace.DUAL_RAIL,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            input_parameters=params,
            n_photons=num_features * num_qubits_per_feature,
        )


########################################################
# Not easily integrable with the NQE framework


# Use training without nqe
# use the kernel_without_nqe_function
########################################################
def amplitude_encoding(
    features: list[float],
    num_modes: int,
    num_photons: int = 0,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
    indexes_perm: list[int] | None = None,
) -> pcvl.Circuit:
    if computation_space == ml.ComputationSpace.UNBUNCHED:
        state = np.zeros(math.comb(num_modes, num_photons), dtype=np.complex128)
    elif computation_space == ml.ComputationSpace.FOCK:
        state = np.zeros(
            math.comb(num_modes + num_photons - 1, num_photons), dtype=np.complex128
        )
    elif computation_space == ml.ComputationSpace.DUAL_RAIL:
        state = np.zeros(2**num_modes, dtype=np.complex128)
    else:
        raise ValueError("Invalid computation space")
    state[: len(features)] = features
    state /= np.linalg.norm(state)
    if indexes_perm is not None:
        state = state[indexes_perm]
    return state


def dense_encoding_of_features(
    features: list[float],
    num_modes: int,
    num_photons: int = 0,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
    indexes_perm: list[int] | None = None,
) -> pcvl.Circuit:
    if computation_space == ml.ComputationSpace.UNBUNCHED:
        state = np.zeros(math.comb(num_modes, num_photons), dtype=np.complex128)
    elif computation_space == ml.ComputationSpace.FOCK:
        state = np.zeros(
            math.comb(num_modes + num_photons - 1, num_photons), dtype=np.complex128
        )
    elif computation_space == ml.ComputationSpace.DUAL_RAIL:
        state = np.zeros(2**num_modes, dtype=np.complex128)
    else:
        raise ValueError("Invalid computation space")

    feature_shuffler = np.zeros(len(state) * 2)
    feature_shuffler[: len(features)] = features

    if indexes_perm is not None:
        feature_shuffler = feature_shuffler[indexes_perm]

    for state_index, feature_index in enumerate(range(0, len(feature_shuffler), 2)):
        state[state_index] = (
            feature_shuffler[feature_index] + 1.0j * feature_shuffler[feature_index + 1]
        )
    state /= np.linalg.norm(state)
    return state


def unitary_evolution(
    features_matrix: torch.Tensor,
    time: float,
) -> pcvl.Circuit:
    feature_size = np.shape(features_matrix)[0]

    unitary_to_apply = torch.zeros(
        [feature_size * 2, feature_size * 2], dtype=torch.complex128
    )

    unitary_to_apply[feature_size:, :feature_size] = features_matrix.conj().transpose(
        0, 1
    )
    unitary_to_apply[:feature_size, feature_size:] = features_matrix
    unitary_to_apply = pcvl.Matrix(
        torch.matrix_exp((-1) * time * 1.0j * unitary_to_apply).detach().cpu().numpy()
    )

    return pcvl.Circuit.decomposition(
        unitary_to_apply,
        MZI,
        phase_shifter_fn=pcvl.PS,
        shape=pcvl.InterferometerShape.TRIANGLE,
        allow_error=True,
    )


class AmplitudeEncoder(nn.Module):
    def __init__(
        self,
        num_modes: int,
        num_photons: int = 0,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        shuffle_amplitude: bool = False,
        return_sv: bool = True,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_photons = num_photons
        self.computation_space = computation_space
        self.shuffle_amplitude = shuffle_amplitude
        self.return_sv = return_sv

        if self.computation_space is ml.ComputationSpace.UNBUNCHED:
            self.output_size = math.comb(self.num_modes, self.num_photons)
        elif self.computation_space is ml.ComputationSpace.FOCK:
            self.output_size = math.comb(
                self.num_modes + self.num_photons - 1, self.num_photons
            )
        elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
            self.output_size = 2 ** (num_modes / 2)
        else:
            raise ValueError("Wrong computation space")

        if self.shuffle_amplitude:
            self.indexes_perm = np.arange(self.output_size)
            np.random.shuffle(self.indexes_perm)
        else:
            self.indexes_perm = None

        self.encoder = ml.LexGrouping(self.output_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        if self.return_sv:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                output_tensors[i, :] = self.encoder(
                    torch.tensor(
                        amplitude_encoding(
                            tensor,
                            self.num_modes,
                            computation_space=self.computation_space,
                            num_photons=self.num_photons,
                            indexes_perm=self.indexes_perm,
                        )
                    )
                )
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                state = self.encoder(
                    torch.tensor(
                        amplitude_encoding(
                            tensor,
                            self.num_modes,
                            computation_space=self.computation_space,
                            num_photons=self.num_photons,
                            indexes_perm=self.indexes_perm,
                        )
                    )
                )
                output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "AmplitudeEncoder()"


class DenseAmplitudeEncoder(nn.Module):
    def __init__(
        self,
        num_modes: int,
        num_photons: int = 0,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        shuffle_amplitude: bool = False,
        return_sv: bool = True,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_photons = num_photons
        self.computation_space = computation_space
        self.shuffle_amplitude = shuffle_amplitude
        self.return_sv = return_sv

        if self.computation_space is ml.ComputationSpace.UNBUNCHED:
            self.output_size = math.comb(self.num_modes, self.num_photons)
        elif self.computation_space is ml.ComputationSpace.FOCK:
            self.output_size = math.comb(
                self.num_modes + self.num_photons - 1, self.num_photons
            )
        elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
            self.output_size = 2 ** (num_modes / 2)
        else:
            raise ValueError("Wrong computation space")

        if self.shuffle_amplitude:
            self.indexes_perm = np.arange(self.output_size)
            np.random.shuffle(self.indexes_perm)
        else:
            self.indexes_perm = None

        self.encoder = ml.LexGrouping(self.output_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        if self.return_sv:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                output_tensors[i, :] = self.encoder(
                    torch.tensor(
                        dense_encoding_of_features(
                            tensor,
                            self.num_modes,
                            computation_space=self.computation_space,
                            num_photons=self.num_photons,
                            indexes_perm=self.indexes_perm,
                        )
                    )
                )
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                state = self.encoder(
                    torch.tensor(
                        dense_encoding_of_features(
                            tensor,
                            self.num_modes,
                            computation_space=self.computation_space,
                            num_photons=self.num_photons,
                            indexes_perm=self.indexes_perm,
                        )
                    )
                )
                output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "DenseAmplitudeEncoder()"


class TimeEvolutionEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_photons: int,
        time: float = 0.1,
        computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
        return_sv: bool = True,
        input_are_images: bool = True,
    ):
        """
        image_size is one size of the image
        """
        super().__init__()
        self.time = time
        self.num_photons = num_photons
        self.image_size = image_size
        self.computation_space = computation_space
        self.return_sv = return_sv
        self.input_are_images = input_are_images
        self.num_modes = (
            2 * image_size if input_are_images else 2 * (image_size // 2 + 1)
        )

        base_circuit = ml.CircuitBuilder(n_modes=self.num_modes)
        base_circuit.add_entangling_layer(trainable=True)
        self.base_perceval = base_circuit.to_pcvl_circuit()
        self.num_modes = self.base_perceval.m

        if self.computation_space is ml.ComputationSpace.UNBUNCHED:
            if input_are_images:
                self.output_size = math.comb(self.image_size * 2, self.num_photons)
            else:
                self.output_size = math.comb(self.num_modes, self.num_photons)
        elif self.computation_space is ml.ComputationSpace.FOCK:
            if input_are_images:
                self.output_size = math.comb(
                    self.num_modes + self.num_photons - 1, self.num_photons
                )
            else:
                self.output_size = math.comb(
                    self.num_modes + self.num_photons - 1,
                    self.num_photons,
                )
        elif self.computation_space is ml.ComputationSpace.DUAL_RAIL:
            if input_are_images:
                self.output_size = 2**image_size
            else:
                self.output_size = 2 ** (self.image_size // 2 + 1)
        else:
            raise ValueError("Wrong computation space")

        self.encoder = ml.LexGrouping(self.output_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            if not self.input_are_images:
                output_tensor = torch.empty(
                    (x.shape[0], x.shape[1] // 2 + 1, x.shape[1] // 2 + 1)
                )
                for i, tensor in enumerate(x):
                    output_tensor[i, :, :] = vector_to_matrix_evo(tensor)
                x = output_tensor
            else:
                x = x.unsqueeze(0)

        if self.return_sv:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                total_circuit = self.base_perceval.copy()
                total_circuit.add(
                    list(range(self.num_modes)),
                    unitary_evolution(tensor, self.time),
                )
                qlayer = ml.QuantumLayer(
                    circuit=total_circuit,
                    n_photons=self.num_photons,
                    measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
                    computation_space=self.computation_space,
                    trainable_parameters=["el_"],
                )
                output_tensors[i, :] = self.encoder(qlayer().flatten())
        else:
            output_tensors = torch.empty(
                (x.shape[0], self.output_size, self.output_size),
                dtype=torch.complex128,
            )

            for i, tensor in enumerate(x):
                total_circuit = self.base_perceval.copy()
                total_circuit.add(
                    list(range(self.num_modes)),
                    unitary_evolution(tensor, self.time),
                )
                qlayer = ml.QuantumLayer(
                    circuit=total_circuit,
                    n_photons=self.num_photons,
                    measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
                    computation_space=self.computation_space,
                    trainable_parameters=["el_"],
                )
                state = self.encoder(qlayer().flatten())

                output_tensors[i, :, :] = torch.outer(state, state.conj())

        return output_tensors

    def __repr__(self):
        return "TimeEvolutionEncoder()"
