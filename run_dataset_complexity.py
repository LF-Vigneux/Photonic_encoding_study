import json
import numpy as np
import torch
import sys
from pathlib import Path
import merlin as ml
import torch.nn as nn
from copy import deepcopy

_FILE_DIR = Path(__file__).resolve().parent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import data_load_and_process
from dataset_complexity.complexity_metrics import (
    classical_complexity,
    induced_quantum_complexity,
)
from encodings_merlin.encoding_layers import (
    angle_encoding_layer,
    dense_angle_encoding_layer,
    fourier_basis_layer,
    AmplitudeEncoder,
    DenseAmplitudeEncoder,
    TimeEvolutionEncoder,
)
from encodings_merlin.utils import find_mode_photon_config
from nn_embedding.lib.merlin_based_model import NeuralEmbeddingMerLinKernel
from nn_embedding.utils.utils import TransparentModel
from dataset_complexity.plotter import plot_complexity_comparison


def dataset_complexity_induced_comparison(
    dataset_name: str,
    classes: tuple | None = None,
    feature_reduction: int | None = None,
    n_modes: int | None = None,
    n_photons: int | None = None,
    computation_space: ml.ComputationSpace = ml.ComputationSpace.UNBUNCHED,
    num_qubits_per_feature_fourier: int = 1,
    hyper_parameters_classical: list[float] = [1, 1, 1, 1],
    max_order_correlation_classical: int | None = None,
    max_dim_topology_classical: int = 2,
    weights_topology_classical: list[float] | None = None,
    max_samples_topology_classical: int | None = 1000,
    hyper_parameters_induced: list[float] = [1, 1, 1, 1, 1, 1],
    epsilon_hilbert_support_dim_induced: float = 1e-8,
    n_samples_loc_vs_express_induced: int = 1000,
    n_bins_loc_vs_express_induced: int = 50,
    max_dim_topology_induced: int = 2,
    weights_topology_induced: list[float] | None = None,
    max_samples_topology_induced: int | None = 1000,
    run_dir: Path = None,
) -> dict:
    output = {"classical": None, "induced": {}}

    x_train, x_test, y_train, _ = data_load_and_process(
        dataset=dataset_name, classes=classes, feature_reduction=feature_reduction
    )
    X = torch.cat((x_train, x_test), 0)
    n_features = int(np.prod(X.shape[1:]))

    #######################################################
    ### Classical complexity
    #######################################################

    print(f"Doing the classical complexity 1/8")
    output["classical"] = classical_complexity(
        X,
        hyper_parameters=hyper_parameters_classical,
        max_order_correlation=max_order_correlation_classical,
        max_dim_topology=max_dim_topology_classical,
        weights_topology=weights_topology_classical,
        max_samples_topology=max_samples_topology_classical,
    )
    print(f"Complexity of {output["classical"]}")

    #######################################################
    ### Induced complexity
    #######################################################

    ###########################
    ### Angle complexity
    ###########################
    print(f"Doing the angle encoding complexity 2/8")
    if n_photons is None:
        if n_modes is None:
            num_photons_encoder = n_features // 2
        else:
            num_photons_encoder = max(n_features // 2, n_modes)
    else:
        num_photons_encoder = n_photons

    encoder = angle_encoding_layer(
        num_features=n_features,
        num_modes=n_modes,
        num_photons=num_photons_encoder,
        computation_space=computation_space,
    )

    model = NeuralEmbeddingMerLinKernel(
        classical_model=TransparentModel(),
        quantum_embedding_layer=encoder,
    )
    output["induced"]["angle"] = induced_quantum_complexity(
        X,
        model,
        hyper_parameters=hyper_parameters_induced,
        epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
        n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
        n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
        max_dim_topology=max_dim_topology_induced,
        weights_topology=weights_topology_induced,
        max_samples_topology=max_samples_topology_induced,
    )
    print(f"Complexity of {output['induced']['angle']}")

    ###########################
    ### Dense Angle complexity
    ###########################
    print(f"Doing the dense angle encoding complexity 3/8")
    encoder = (
        dense_angle_encoding_layer(
            num_features=n_features,
            num_modes=n_modes,
        ),
    )

    model = NeuralEmbeddingMerLinKernel(
        classical_model=TransparentModel(),
        quantum_embedding_layer=encoder,
    )
    output["induced"]["dense_angle"] = induced_quantum_complexity(
        X,
        model,
        hyper_parameters=hyper_parameters_induced,
        epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
        n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
        n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
        max_dim_topology=max_dim_topology_induced,
        weights_topology=weights_topology_induced,
        max_samples_topology=max_samples_topology_induced,
    )
    print(f"Complexity of {output['induced']['dense_angle']}")

    ###########################
    ### Fourier complexity
    ###########################
    print(f"Doing the Fourier encoding complexity 4/8")
    encoder = (
        fourier_basis_layer(
            num_features=n_features,
            num_qubits_per_feature=num_qubits_per_feature_fourier,
        ),
    )

    model = NeuralEmbeddingMerLinKernel(
        classical_model=TransparentModel(),
        quantum_embedding_layer=encoder,
    )
    output["induced"]["fourier"] = induced_quantum_complexity(
        X,
        model,
        hyper_parameters=hyper_parameters_induced,
        epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
        n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
        n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
        max_dim_topology=max_dim_topology_induced,
        weights_topology=weights_topology_induced,
        max_samples_topology=max_samples_topology_induced,
    )
    print(f"Complexity of {output['induced']['fourier']}")

    ###########################
    ### Amplitude complexity
    ###########################
    print(f"Doing the amplitude encoding complexity 5/8")
    num_modes_encoder, num_photons_encoder = (
        find_mode_photon_config(n_features)
        if (n_modes is None or n_photons is None)
        else (n_modes, n_photons)
    )
    encoder = AmplitudeEncoder(
        num_modes=num_modes_encoder,
        num_photons=num_photons_encoder,
        computation_space=computation_space,
    )
    output["induced"]["amplitude"] = induced_quantum_complexity(
        X,
        encoder,
        hyper_parameters=hyper_parameters_induced,
        epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
        n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
        n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
        max_dim_topology=max_dim_topology_induced,
        weights_topology=weights_topology_induced,
        max_samples_topology=max_samples_topology_induced,
    )
    print(f"Complexity of {output['induced']['amplitude']}")

    ###########################
    ### Dense amplitude complexity
    ###########################
    print(f"Doing the dense angle encoding complexity 6/8")
    num_modes_encoder, num_photons_encoder = (
        find_mode_photon_config(n_features // 2 + 1)
        if (n_modes is None or n_photons is None)
        else (n_modes, n_photons)
    )
    encoder = DenseAmplitudeEncoder(
        num_modes=num_modes_encoder,
        num_photons=num_photons_encoder,
        computation_space=computation_space,
    )
    output["induced"]["dense_amplitude"] = induced_quantum_complexity(
        X,
        encoder,
        hyper_parameters=hyper_parameters_induced,
        epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
        n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
        n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
        max_dim_topology=max_dim_topology_induced,
        weights_topology=weights_topology_induced,
        max_samples_topology=max_samples_topology_induced,
    )
    print(f"Complexity of {output['induced']['dense_amplitude']}")

    ###########################
    ### Evolution complexity
    ###########################
    print(f"Doing the evolution encoding complexity 7/8")
    if len(X.shape) == 2:
        encoder = TimeEvolutionEncoder(
            image_size=n_features,
            num_photons=n_features if n_photons is None else n_photons,
            computation_space=computation_space,
            input_are_images=False,
        )
    else:
        encoder = TimeEvolutionEncoder(
            image_size=X.size(2),
            num_photons=n_features if n_photons is None else n_photons,
            computation_space=computation_space,
            input_are_images=True,
        )
        # Merging RGB if necessary
        if X.ndim == 4 and X.shape[1] == 3:
            X_copy = deepcopy(X)
            # Standard luminance conversion keeps the image square.
            weights = X_copy.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
            merged_X = (X * weights).sum(dim=1)

    output["induced"]["evolution"] = induced_quantum_complexity(
        merged_X,
        encoder,
        hyper_parameters=hyper_parameters_induced,
        epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
        n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
        n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
        max_dim_topology=max_dim_topology_induced,
        weights_topology=weights_topology_induced,
        max_samples_topology=max_samples_topology_induced,
    )
    print(f"Complexity of {output['induced']['evolution']}")

    ###########################
    ### NQE
    ###########################
    print(f"Doing the nqe complexity 8/8")
    # Same number of modes as dense amplitude encoder, using the lesser most ressources
    general_unitary = ml.CircuitBuilder(n_modes=num_modes_encoder)
    # Two deep
    general_unitary.add_entangling_layer()
    general_unitary.add_entangling_layer()

    encoder = ml.QuantumLayer(
        input_size=0,
        builder=deepcopy(general_unitary),
        n_photons=num_photons_encoder,
        computation_space=computation_space,
        measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
    )

    if len(X.shape) == 2:
        classical_model = nn.Sequential(
            nn.Linear(n_features, n_features // 2 + 10),
            nn.ReLU(),
            nn.Linear(n_features // 2 + 10, n_features // 2 + 10),
            nn.ReLU(),
            nn.Linear(
                n_features // 2 + 10, sum([i.numel() for i in encoder.parameters()])
            ),
        )
    else:
        in_channels = X.shape[1] if X.ndim == 4 else 1
        classical_model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.LazyLinear(sum([i.numel() for i in encoder.parameters()])),
        )

    model = NeuralEmbeddingMerLinKernel(
        classical_model=classical_model,
        quantum_embedding_layer=encoder,
    )
    model.train_embedding(
        x_train=x_train, y_train=y_train, batch_size=100, num_epochs=1000
    )
    output["induced"]["nqe"] = induced_quantum_complexity(
        X,
        model,
        hyper_parameters=hyper_parameters_induced,
        epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
        n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
        n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
        max_dim_topology=max_dim_topology_induced,
        weights_topology=weights_topology_induced,
        max_samples_topology=max_samples_topology_induced,
    )
    print(f"Complexity of {output['induced']['nqe']}")
    print("Saving")
    payload = {
        "results": output,
        "config": {
            "dataset_name": dataset_name,
            "classes": list(classes) if classes is not None else None,
            "feature_reduction": feature_reduction,
            "n_modes": n_modes,
            "n_photons": n_photons,
            "computation_space": computation_space.name,
            "num_qubits_per_feature_fourier": num_qubits_per_feature_fourier,
            "hyper_parameters_classical": hyper_parameters_classical,
            "max_order_correlation_classical": max_order_correlation_classical,
            "max_dim_topology_classical": max_dim_topology_classical,
            "weights_topology_classical": weights_topology_classical,
            "hyper_parameters_induced": hyper_parameters_induced,
            "epsilon_hilbert_support_dim_induced": epsilon_hilbert_support_dim_induced,
            "n_samples_loc_vs_express_induced": n_samples_loc_vs_express_induced,
            "n_bins_loc_vs_express_induced": n_bins_loc_vs_express_induced,
            "max_dim_topology_induced": max_dim_topology_induced,
            "weights_topology_induced": weights_topology_induced,
            "max_samples_topology_induced": max_samples_topology_induced,
        },
    }

    results_dir = (
        run_dir if run_dir is not None else _FILE_DIR / "results" / "dataset_complexity"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"dataset_complexity_{dataset_name}_results.json"

    def _json_default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)

    output_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"Saved results to {output_path}")

    plot_complexity_comparison(
        results=output,
        dataset_name=dataset_name,
        classes=classes,
        feature_reduction=feature_reduction,
        run_dir=results_dir,
        filename=f"dataset_complexity_{dataset_name}_plot.pdf",
    )

    return payload
