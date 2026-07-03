import json
import gc
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
from egas.lib.photonic_egas import (
    run_egas as run_photonic_egas,
    unique_sorted_candidates,
    refine_candidates,
)
from egas.lib.photonic_circuits import build_token_pool
from dataset_complexity.plotter import (
    plot_complexity_comparison,
    plot_induced_per_encoding,
    plot_normalized_summary,
    plot_complexity_comparison_normalized,
)
from dataset_complexity.umap import save_umap_plots


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
    hyper_parameters_induced: list[float] = [1, 1, 1, 1, 1, 1, 1],
    epsilon_hilbert_support_dim_induced: float = 1e-8,
    n_samples_loc_vs_express_induced: int = 1000,
    n_bins_loc_vs_express_induced: int = 50,
    max_dim_topology_induced: int = 2,
    weights_topology_induced: list[float] | None = None,
    max_samples_topology_induced: int | None = 1000,
    max_samples_induced: int | None = 5000,
    evaluate_evolution: bool = False,
    randomize_entangling: bool = True,
    generate_umap_plots: bool = True,
    generate_umap_2d: bool = True,
    generate_umap_3d: bool = True,
    umap_state: int = 42,
    umap_num_points_per_class: int = 50,
    umap_n_neighbors: int = 15,
    umap_n_epochs: int = 200,
    run_dir: Path = None,
) -> dict:
    output = {"classical": None, "induced": {}}

    import time as _time

    _t0 = _time.perf_counter()
    print("Loading dataset...", flush=True)
    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=dataset_name, classes=classes, feature_reduction=feature_reduction
    )
    print(f"Dataset loaded in {_time.perf_counter() - _t0:.1f}s", flush=True)
    X = torch.cat((x_train, x_test), 0)
    Y = torch.cat((y_train, y_test), 0)
    n_features = int(np.prod(X.shape[1:]))

    def _total(v) -> float:
        if isinstance(v, dict):
            if "total" in v and isinstance(v["total"], (int, float, np.number)):
                return float(v["total"])
            return float(
                sum(
                    float(val)
                    for k, val in v.items()
                    if k != "total" and isinstance(val, (int, float, np.number))
                )
            )
        return float(v)

    # ── Persistence setup ─────────────────────────────────────────────────────
    results_root_dir = (
        run_dir if run_dir is not None else _FILE_DIR / "results" / "dataset_complexity"
    )
    complexities_dir = results_root_dir / "complexities"
    umaps_dir = results_root_dir / "umaps"
    complexities_dir.mkdir(parents=True, exist_ok=True)
    umaps_dir.mkdir(parents=True, exist_ok=True)

    output_path = complexities_dir / f"dataset_complexity_{dataset_name}_results.json"
    legacy_output_path = (
        results_root_dir / f"dataset_complexity_{dataset_name}_results.json"
    )

    config_payload = {
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
        "max_samples_induced": max_samples_induced,
        "randomize_entangling": randomize_entangling,
        "generate_umap_plots": generate_umap_plots,
        "generate_umap_2d": generate_umap_2d,
        "generate_umap_3d": generate_umap_3d,
        "umap_state": umap_state,
        "umap_num_points_per_class": umap_num_points_per_class,
        "umap_n_neighbors": umap_n_neighbors,
        "umap_n_epochs": umap_n_epochs,
    }

    def _json_default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)

    def _save():
        payload = {"results": output, "config": config_payload}
        output_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    def _has_umap_outputs(embedding_name: str) -> bool:
        dataset_slug = "_".join(str(dataset_name).strip().lower().split()) or "dataset"
        embedding_slug = (
            "_".join(str(embedding_name).strip().lower().split()) or "embedding"
        )
        base_name = f"u_map_{embedding_slug}_{dataset_slug}"
        path_2d = umaps_dir / f"{base_name}_2d.html"
        path_3d = umaps_dir / f"{base_name}_3d.html"

        if generate_umap_2d and not path_2d.exists():
            return False
        if generate_umap_3d and not path_3d.exists():
            return False
        return True

    def _save_umap_projection(
        embedding_name: str,
        embedder=None,
        X_input: torch.Tensor | None = None,
    ) -> None:
        if not generate_umap_plots:
            return

        X_plot = X if X_input is None else X_input
        Y_plot = Y
        # Sample umap_num_points_per_class points per class (stratified)
        unique_classes = torch.unique(Y_plot)
        keep_idx = torch.cat(
            [
                idx_c[:umap_num_points_per_class]
                for c in unique_classes
                if len(idx_c := torch.where(Y_plot == c)[0]) > 0
            ]
        )
        X_plot = X_plot[keep_idx]
        Y_plot = Y_plot[keep_idx]

        saved_paths = save_umap_plots(
            X_plot,
            Y_plot,
            embedder=embedder,
            dataset_name=dataset_name,
            embedding_strategy=embedding_name,
            output_dir=umaps_dir,
            umap_state=umap_state,
            umap_n_neighbors=umap_n_neighbors,
            umap_n_epochs=umap_n_epochs,
            save_2d=generate_umap_2d,
            save_3d=generate_umap_3d,
        )
        if saved_paths:
            print(
                "Saved UMAP plots for "
                f"{embedding_name}: {', '.join(str(path.name) for path in saved_paths.values())}"
            )

    # Load any previously computed results
    results_to_load_path = output_path if output_path.exists() else legacy_output_path
    if results_to_load_path.exists():
        try:
            existing = json.loads(results_to_load_path.read_text())
            existing_results = existing.get("results", {})
            if existing_results.get("classical") is not None:
                output["classical"] = existing_results["classical"]
            for key, val in existing_results.get("induced", {}).items():
                output["induced"][key] = val
            print(f"Loaded existing results from {results_to_load_path}")
        except Exception:
            pass
    # ──────────────────────────────────────────────────────────────────────────
    print(f"Everything loaded, ready to compute")
    #######################################################
    ### Classical complexity
    #######################################################

    print(f"Doing the classical complexity 1/9")
    if output["classical"] is None:
        output["classical"] = classical_complexity(
            X,
            hyper_parameters=hyper_parameters_classical,
            max_order_correlation=max_order_correlation_classical,
            max_dim_topology=max_dim_topology_classical,
            weights_topology=weights_topology_classical,
            max_samples_topology=max_samples_topology_classical,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("classical")
    print(f"Complexity of {_total(output['classical'])}")
    if (
        output["classical"] is not None
        and generate_umap_plots
        and not _has_umap_outputs("classical")
    ):
        print("UMAP (cached classical)")
        _save_umap_projection("classical")
    print()

    #######################################################
    ### Induced complexity
    #######################################################

    ###########################
    ### Angle complexity
    ###########################
    print(f"Doing the angle encoding complexity 2/9")
    if n_photons is None:
        if n_modes is None:
            num_photons_encoder = n_features // 2
        else:
            num_photons_encoder = max(n_features // 2, n_modes)
    else:
        num_photons_encoder = n_photons

    if output["induced"].get("angle") is None:
        encoder = angle_encoding_layer(
            num_features=n_features,
            num_modes=n_modes,
            num_photons=num_photons_encoder,
            computation_space=computation_space,
            randomize_entangling=randomize_entangling,
        )
        model = NeuralEmbeddingMerLinKernel(
            classical_model=TransparentModel(),
            quantum_embedding_layer=encoder,
        )
        output["induced"]["angle"] = induced_quantum_complexity(
            X,
            Y,
            model,
            hyper_parameters=hyper_parameters_induced,
            epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
            max_dim_topology=max_dim_topology_induced,
            weights_topology=weights_topology_induced,
            max_samples_topology=max_samples_topology_induced,
            max_samples=max_samples_induced,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("angle", embedder=model)
        del encoder, model
    gc.collect()
    print(f"Complexity of {_total(output['induced']['angle'])}")
    if (
        output["induced"].get("angle") is not None
        and generate_umap_plots
        and not _has_umap_outputs("angle")
    ):
        encoder = angle_encoding_layer(
            num_features=n_features,
            num_modes=n_modes,
            num_photons=num_photons_encoder,
            computation_space=computation_space,
            randomize_entangling=randomize_entangling,
        )
        model = NeuralEmbeddingMerLinKernel(
            classical_model=TransparentModel(),
            quantum_embedding_layer=encoder,
        )
        print("UMAP (cached angle)")
        _save_umap_projection("angle", embedder=model)
        del encoder, model
        gc.collect()
    print()

    ###########################
    ### Dense Angle complexity
    ###########################
    print(f"Doing the dense angle encoding complexity 3/9")
    if output["induced"].get("dense_angle") is None:
        encoder = dense_angle_encoding_layer(
            num_features=n_features,
            num_modes=n_modes,
        )
        model = NeuralEmbeddingMerLinKernel(
            classical_model=TransparentModel(),
            quantum_embedding_layer=encoder,
        )
        output["induced"]["dense_angle"] = induced_quantum_complexity(
            X,
            Y,
            model,
            hyper_parameters=hyper_parameters_induced,
            epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
            max_dim_topology=max_dim_topology_induced,
            weights_topology=weights_topology_induced,
            max_samples_topology=max_samples_topology_induced,
            max_samples=max_samples_induced,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("dense_angle", embedder=model)
        del encoder, model
    gc.collect()
    print(f"Complexity of {_total(output['induced']['dense_angle'])}")
    if (
        output["induced"].get("dense_angle") is not None
        and generate_umap_plots
        and not _has_umap_outputs("dense_angle")
    ):
        encoder = dense_angle_encoding_layer(
            num_features=n_features,
            num_modes=n_modes,
        )
        model = NeuralEmbeddingMerLinKernel(
            classical_model=TransparentModel(),
            quantum_embedding_layer=encoder,
        )
        print("UMAP (cached dense_angle)")
        _save_umap_projection("dense_angle", embedder=model)
        del encoder, model
        gc.collect()
    print()

    ###########################
    ### Fourier complexity
    ###########################
    print(f"Doing the Fourier encoding complexity 4/9")
    if output["induced"].get("fourier") is None:
        encoder = fourier_basis_layer(
            num_features=n_features,
            num_qubits_per_feature=num_qubits_per_feature_fourier,
        )
        model = NeuralEmbeddingMerLinKernel(
            classical_model=TransparentModel(),
            quantum_embedding_layer=encoder,
        )
        output["induced"]["fourier"] = induced_quantum_complexity(
            X,
            Y,
            model,
            hyper_parameters=hyper_parameters_induced,
            epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
            max_dim_topology=max_dim_topology_induced,
            weights_topology=weights_topology_induced,
            max_samples_topology=max_samples_topology_induced,
            max_samples=max_samples_induced,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("fourier", embedder=model)
        del encoder, model
    gc.collect()
    print(f"Complexity of {_total(output['induced']['fourier'])}")
    if (
        output["induced"].get("fourier") is not None
        and generate_umap_plots
        and not _has_umap_outputs("fourier")
    ):
        encoder = fourier_basis_layer(
            num_features=n_features,
            num_qubits_per_feature=num_qubits_per_feature_fourier,
        )
        model = NeuralEmbeddingMerLinKernel(
            classical_model=TransparentModel(),
            quantum_embedding_layer=encoder,
        )
        print("UMAP (cached fourier)")
        _save_umap_projection("fourier", embedder=model)
        del encoder, model
        gc.collect()
    print()

    ###########################
    ### Amplitude complexity
    ###########################
    print(f"Doing the amplitude encoding complexity 5/9")
    num_modes_encoder, num_photons_encoder = (
        find_mode_photon_config(n_features, len(classes))
        if (n_modes is None or n_photons is None)
        else (n_modes, n_photons)
    )
    if output["induced"].get("amplitude") is None:
        encoder = AmplitudeEncoder(
            num_modes=num_modes_encoder,
            num_photons=num_photons_encoder,
            computation_space=computation_space,
        )
        output["induced"]["amplitude"] = induced_quantum_complexity(
            X,
            Y,
            encoder,
            hyper_parameters=hyper_parameters_induced,
            epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
            max_dim_topology=max_dim_topology_induced,
            weights_topology=weights_topology_induced,
            max_samples_topology=max_samples_topology_induced,
            max_samples=max_samples_induced,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("amplitude", embedder=encoder)
        del encoder
    gc.collect()
    print(f"Complexity of {_total(output['induced']['amplitude'])}")
    if (
        output["induced"].get("amplitude") is not None
        and generate_umap_plots
        and not _has_umap_outputs("amplitude")
    ):
        encoder = AmplitudeEncoder(
            num_modes=num_modes_encoder,
            num_photons=num_photons_encoder,
            computation_space=computation_space,
        )
        print("UMAP (cached amplitude)")
        _save_umap_projection("amplitude", embedder=encoder)
        del encoder
        gc.collect()
    print()

    ###########################
    ### Dense amplitude complexity
    ###########################
    print(f"Doing the dense angle encoding complexity 6/9")
    num_modes_encoder, num_photons_encoder = (
        find_mode_photon_config(max(n_features // 2 + 1, len(classes)))
        if (n_modes is None or n_photons is None)
        else (n_modes, n_photons)
    )
    if output["induced"].get("dense_amplitude") is None:
        encoder = DenseAmplitudeEncoder(
            num_modes=num_modes_encoder,
            num_photons=num_photons_encoder,
            computation_space=computation_space,
        )
        output["induced"]["dense_amplitude"] = induced_quantum_complexity(
            X,
            Y,
            encoder,
            hyper_parameters=hyper_parameters_induced,
            epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
            max_dim_topology=max_dim_topology_induced,
            weights_topology=weights_topology_induced,
            max_samples_topology=max_samples_topology_induced,
            max_samples=max_samples_induced,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("dense_amplitude", embedder=encoder)
        del encoder
    gc.collect()
    print(f"Complexity of {_total(output['induced']['dense_amplitude'])}")
    if (
        output["induced"].get("dense_amplitude") is not None
        and generate_umap_plots
        and not _has_umap_outputs("dense_amplitude")
    ):
        encoder = DenseAmplitudeEncoder(
            num_modes=num_modes_encoder,
            num_photons=num_photons_encoder,
            computation_space=computation_space,
        )
        print("UMAP (cached dense_amplitude)")
        _save_umap_projection("dense_amplitude", embedder=encoder)
        del encoder
        gc.collect()
    print()

    ###########################
    ### Evolution complexity
    ###########################
    print(f"Doing the evolution encoding complexity 7/9")
    _existing_evolution = output["induced"].get("evolution")
    if _existing_evolution is not None:
        print(f"Complexity of {_total(_existing_evolution)} (loaded from file)")
    elif not evaluate_evolution:
        print("Skipping evolution encoding (evaluate_evolution=False)")
        output["induced"]["evolution"] = None
    else:
        merged_X = X
        if len(X.shape) == 2:
            encoder = TimeEvolutionEncoder(
                image_size=n_features,
                num_photons=n_features if n_photons is None else n_photons,
                computation_space=computation_space,
                input_are_images=False,
                randomize_entangling=randomize_entangling,
            )
        else:
            encoder = TimeEvolutionEncoder(
                image_size=X.size(2),
                num_photons=n_features if n_photons is None else n_photons,
                computation_space=computation_space,
                input_are_images=True,
                randomize_entangling=randomize_entangling,
            )
            # Merging RGB if necessary
            if X.ndim == 4 and X.shape[1] == 3:
                X_copy = deepcopy(X)
                # Standard luminance conversion keeps the image square.
                weights = X_copy.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
                merged_X = (X * weights).sum(dim=1)

        output["induced"]["evolution"] = induced_quantum_complexity(
            merged_X,
            Y,
            encoder,
            hyper_parameters=hyper_parameters_induced,
            epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
            max_dim_topology=max_dim_topology_induced,
            weights_topology=weights_topology_induced,
            max_samples_topology=max_samples_topology_induced,
            max_samples=max_samples_induced,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("evolution", embedder=encoder, X_input=merged_X)
        print(f"Complexity of {_total(output['induced']['evolution'])}")
        del encoder
        if merged_X is not X:
            del merged_X
    gc.collect()
    if (
        output["induced"].get("evolution") is not None
        and generate_umap_plots
        and not _has_umap_outputs("evolution")
        and evaluate_evolution
    ):
        merged_X = X
        if len(X.shape) == 2:
            encoder = TimeEvolutionEncoder(
                image_size=n_features,
                num_photons=n_features if n_photons is None else n_photons,
                computation_space=computation_space,
                input_are_images=False,
                randomize_entangling=randomize_entangling,
            )
        else:
            encoder = TimeEvolutionEncoder(
                image_size=X.size(2),
                num_photons=n_features if n_photons is None else n_photons,
                computation_space=computation_space,
                input_are_images=True,
                randomize_entangling=randomize_entangling,
            )
            if X.ndim == 4 and X.shape[1] == 3:
                weights = X.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
                merged_X = (X * weights).sum(dim=1)
        print("UMAP (cached evolution)")
        _save_umap_projection("evolution", embedder=encoder, X_input=merged_X)
        del encoder
        if merged_X is not X:
            del merged_X
        gc.collect()
    print()

    ###########################
    ### NQE
    ###########################
    print(f"Doing the nqe complexity 8/9")
    nqe_batch_size = (
        len(classes) * 100 if classes is not None else len(torch.unique(y_train)) * 100
    )

    def _train_nqe_model() -> NeuralEmbeddingMerLinKernel:
        # Same number of modes as dense amplitude encoder, using the lesser most resources.
        general_unitary = ml.CircuitBuilder(n_modes=max(n_features, len(classes)) // 2)
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
                    n_features // 2 + 10,
                    sum([i.numel() for i in encoder.parameters()]),
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

        model_local = NeuralEmbeddingMerLinKernel(
            classical_model=classical_model,
            quantum_embedding_layer=encoder,
        )
        model_local.train_embedding(
            x_train=x_train,
            y_train=y_train,
            batch_size=nqe_batch_size,
            num_epochs=1000,
        )
        return model_local

    if output["induced"].get("nqe") is None:
        model = _train_nqe_model()
        output["induced"]["nqe"] = induced_quantum_complexity(
            X,
            Y,
            model,
            hyper_parameters=hyper_parameters_induced,
            epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
            max_dim_topology=max_dim_topology_induced,
            weights_topology=weights_topology_induced,
            max_samples_topology=max_samples_topology_induced,
            max_samples=max_samples_induced,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("nqe", embedder=model)
        del model
    gc.collect()
    print(f"Complexity of {_total(output['induced']['nqe'])}")
    if (
        output["induced"].get("nqe") is not None
        and generate_umap_plots
        and not _has_umap_outputs("nqe")
    ):
        print("UMAP (cached nqe)")
        model = _train_nqe_model()
        _save_umap_projection("nqe", embedder=model)
        del model
        gc.collect()
    print()
    print(f"Saved results to {output_path}")

    ###########################
    ### EGAS
    ###########################
    print(f"Doing the EGAS complexity 9/9")
    nqe_batch_size = (
        len(classes) * 100 if classes is not None else len(torch.unique(y_train)) * 100
    )

    def _train_egas_model() -> nn.Module:
        def _flatten_for_egas(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(tensor.size(0), -1) if tensor.ndim > 2 else tensor

        X_flat = _flatten_for_egas(X)
        x_train_flat = _flatten_for_egas(x_train)
        y_train_flat = y_train

        n_modes_encoder = (
            n_modes if n_modes is not None else min(int(X_flat.shape[1]), 8)
        )
        n_modes_encoder = max(1, min(int(X_flat.shape[1]), int(n_modes_encoder)))
        num_photons_encoder = n_photons if n_photons is not None else 2

        print(
            f"Running EGAS search with n_modes={n_modes_encoder}, num_photons={num_photons_encoder}, "
            f"seq_len={min(8, n_modes_encoder)}"
        )

        pool = build_token_pool(n_modes_encoder)
        seed = 0
        rng = np.random.default_rng(seed)
        search_samples = min(256, len(x_train_flat))
        idx = rng.choice(len(x_train_flat), search_samples, replace=False)
        Xe = x_train_flat[idx]
        ye = y_train_flat[idx]

        gpt, history, buffer = run_photonic_egas(
            pool,
            Xe,
            ye,
            n_modes_encoder,
            seq_len=max(28, n_modes * 3 + 4),
            num_photons=num_photons_encoder,
            computation_space=computation_space,
            n_iters=4000,
            n_candidates=24,
            select_k=6,
            gamma=0.1,
            lr=5e-5,
            temp_max=100.0,
            temp_min=0.04,
            d_model=64,
            n_layers=2,
            n_heads=4,
            seed=seed,
            device=X_flat.device,
            log_every=50,
        )

        G_ids, _ = unique_sorted_candidates(buffer, top=5, bottom=0)
        refined = refine_candidates(
            G_ids,
            pool,
            Xe,
            ye,
            n_modes_encoder,
            num_photons=num_photons_encoder,
            computation_space=computation_space,
            device=X_flat.device,
            epochs=400,
            batch_samples=25,
            lr=5e-4,
            seed=seed,
        )
        if not refined:
            raise RuntimeError("EGAS produced no refined candidates")

        best = min(refined, key=lambda item: item["E_after"])
        print(
            f"Selected EGAS candidate with E_after={best['E_after']:.6f} "
            f"and sequence length={len(best['seq'])}"
        )
        return best["encoder"]

    if output["induced"].get("egas") is None:
        model = _train_egas_model()
        output["induced"]["egas"] = induced_quantum_complexity(
            X,
            Y,
            model,
            hyper_parameters=hyper_parameters_induced,
            epsilon_hilbert_support_dim=epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express=n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express=n_bins_loc_vs_express_induced,
            max_dim_topology=max_dim_topology_induced,
            weights_topology=weights_topology_induced,
            max_samples_topology=max_samples_topology_induced,
            max_samples=max_samples_induced,
        )
        _save()
        print(f"UMAP")
        _save_umap_projection("egas", embedder=model)
        del model
    gc.collect()
    print(f"Complexity of {_total(output['induced']['egas'])}")
    if (
        output["induced"].get("egas") is not None
        and generate_umap_plots
        and not _has_umap_outputs("egas")
    ):
        print("UMAP (cached egas)")
        model = _train_egas_model()
        _save_umap_projection("egas", embedder=model)
        del model
        gc.collect()
    print()
    print(f"Saved results to {output_path}")

    plot_complexity_comparison(
        results=output,
        dataset_name=dataset_name,
        classes=classes,
        feature_reduction=feature_reduction,
        run_dir=complexities_dir,
        filename=f"dataset_complexity_{dataset_name}_plot.pdf",
    )

    # Generate per-encoding and normalized summary plots
    plot_induced_per_encoding(
        output,
        dataset_name=dataset_name,
        run_dir=complexities_dir,
    )
    plot_normalized_summary(
        output,
        dataset_name=dataset_name,
        run_dir=complexities_dir,
    )
    plot_complexity_comparison_normalized(
        output,
        dataset_name=dataset_name,
        classes=classes,
        feature_reduction=feature_reduction,
        run_dir=complexities_dir,
        filename=f"dataset_complexity_{dataset_name}_normalized_comparison.pdf",
    )

    return {"results": output, "config": config_payload}
