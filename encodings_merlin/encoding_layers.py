import numpy as np
import merlin as ml
import perceval as pcvl
import math
import torch
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from encodings_merlin.utils import MZI, vector_to_matrix_evo


def angle_encoding_layer(
    num_features: int,
    param_prefix: str = "phi",
    num_modes: int | None = None,
    num_photons: int = 1,
    computation_space=ml.ComputationSpace.UNBUNCHED,
    input_size_0: bool = True,
) -> ml.QuantumLayer:
    width = len(str(num_features - 1))
    params = [
        pcvl.Parameter(f"{param_prefix}{i:0{width}d}") for i in range(num_features)
    ]
    circuit = pcvl.Circuit(num_features, num_modes)

    for i, param in enumerate(params):
        circuit.add(i, pcvl.PS(phi=param))

    if input_size_0:
        return ml.QuantumLayer(
            input_size=0,
            circuit=circuit,
            computation_space=computation_space,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            trainable_parameters=params,
            n_photons=num_photons,
        )
    else:
        return ml.QuantumLayer(
            input_size=num_features,
            circuit=circuit,
            computation_space=computation_space,
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
            input_parameters=params,
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


# TODO
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


# TODO
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


# TODO
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
