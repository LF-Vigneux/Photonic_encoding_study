import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from nn_embedding.utils.utils import (  # noqa: E402
    parse_args,
)


def train_and_evaluate(cfg, run_dir: Path) -> None:
    """
    Dispatch experiment execution based on a config dictionary.

    Parameters
    ----------
    cfg : dict
        Experiment configuration values.
    run_dir : pathlib.Path
        Output directory for plots and artifacts.

    Returns
    -------
    None
    """
    exp_to_run = cfg.get("exp_to_run", "FIG2")

    if exp_to_run == "INDUCED_DATASET_COMPLEXITY":
        print("Running the DATASET_COMPLEXITY experiment")
        from run_dataset_complexity import (
            dataset_complexity_induced_comparison,
        )  # noqa: E402

        classes_raw = cfg.get("classes", None)
        if isinstance(classes_raw, list):
            classes_raw = tuple(classes_raw)

        hyper_cls_raw = cfg.get("hyper_parameters_classical", [1, 1, 1, 1, 1])
        hyper_ind_raw = cfg.get("hyper_parameters_induced", [1, 1, 1, 1, 1, 1, 1])
        weights_cls_raw = cfg.get("weights_topology_classical", None)
        weights_ind_raw = cfg.get("weights_topology_induced", None)

        cs_name = cfg.get("computation_space", "UNBUNCHED")
        import merlin as _ml

        computation_space = _ml.ComputationSpace[cs_name]

        dataset_complexity_induced_comparison(
            dataset_name=cfg.get("dataset_name", "mnist"),
            classes=classes_raw,
            feature_reduction=cfg.get("feature_reduction", None),
            n_modes=cfg.get("n_modes", None),
            n_photons=cfg.get("n_photons", None),
            computation_space=computation_space,
            num_qubits_per_feature_fourier=cfg.get("num_qubits_per_feature_fourier", 1),
            hyper_parameters_classical=hyper_cls_raw,
            max_order_correlation_classical=cfg.get(
                "max_order_correlation_classical", None
            ),
            max_dim_topology_classical=cfg.get("max_dim_topology_classical", 2),
            weights_topology_classical=weights_cls_raw,
            max_samples_topology_classical=cfg.get(
                "max_samples_topology_classical", 1000
            ),
            hyper_parameters_induced=hyper_ind_raw,
            epsilon_hilbert_support_dim_induced=cfg.get(
                "epsilon_hilbert_support_dim_induced", 1e-8
            ),
            n_samples_loc_vs_express_induced=cfg.get(
                "n_samples_loc_vs_express_induced", 1000
            ),
            n_bins_loc_vs_express_induced=cfg.get("n_bins_loc_vs_express_induced", 50),
            max_dim_topology_induced=cfg.get("max_dim_topology_induced", 2),
            weights_topology_induced=weights_ind_raw,
            max_samples_topology_induced=cfg.get("max_samples_topology_induced", 1000),
            max_samples_induced=cfg.get("max_samples_induced", 5000),
            evaluate_evolution=cfg.get("evaluate_evolution", False),
            randomize_entangling=cfg.get("randomize_entangling", False),
            generate_umap_plots=cfg.get("generate_umap_plots", True),
            generate_umap_2d=cfg.get("generate_umap_2d", True),
            generate_umap_3d=cfg.get("generate_umap_3d", True),
            umap_state=cfg.get("umap_state", 42),
            umap_num_points_per_class=cfg.get("umap_num_points_per_class", 50),
            umap_n_neighbors=cfg.get("umap_n_neighbors", 15),
            umap_n_epochs=cfg.get("umap_n_epochs", 200),
            run_dir=run_dir,
        )
    else:
        raise NameError(f"No experiment with the name '{exp_to_run}'")


def main():
    args = parse_args()
    if args.exp_to_run == "INDUCED_DATASET_COMPLEXITY":
        print("Running the DATASET_COMPLEXITY experiment")
        from run_dataset_complexity import (
            dataset_complexity_induced_comparison,
        )  # noqa: E402

        classes_raw = args.classes
        if isinstance(classes_raw, list):
            classes_raw = tuple(classes_raw)

        hyper_cls_raw = args.hyper_parameters_classical
        hyper_ind_raw = args.hyper_parameters_induced
        weights_cls_raw = args.weights_topology_classical
        weights_ind_raw = args.weights_topology_induced

        cs_name = args.computation_space
        import merlin as _ml

        computation_space = _ml.ComputationSpace[cs_name]

        dataset_complexity_induced_comparison(
            dataset_name=args.dataset_name,
            classes=args.feature_reduction,
            n_modes=args.n_modes,
            n_photons=args.n_photons,
            computation_space=computation_space,
            num_qubits_per_feature_fourier=args.num_qubits_per_feature_fourier,
            hyper_parameters_classical=hyper_cls_raw,
            max_order_correlation_classical=args.max_order_correlation_classical,
            max_dim_topology_classical=args.max_dim_topology_classical,
            weights_topology_classical=weights_cls_raw,
            max_samples_topology_classical=args.max_samples_topology_classical,
            hyper_parameters_induced=hyper_ind_raw,
            epsilon_hilbert_support_dim_induced=args.epsilon_hilbert_support_dim_induced,
            n_samples_loc_vs_express_induced=args.n_samples_loc_vs_express_induced,
            n_bins_loc_vs_express_induced=args.n_bins_loc_vs_express_induced,
            max_dim_topology_induced=args.max_dim_topology_induced,
            weights_topology_induced=weights_ind_raw,
            max_samples_topology_induced=args.max_samples_topology_induced,
            max_samples_induced=args.max_samples_induced,
            evaluate_evolution=args.evaluate_evolution,
            randomize_entangling=args.randomize_entangling,
            generate_umap_plots=args.generate_umap_plots,
            generate_umap_2d=args.generate_umap_2d,
            generate_umap_3d=args.generate_umap_3d,
            umap_state=args.umap_state,
            umap_num_points_per_class=args.umap_num_points_per_class,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_n_epochs=args.umap_n_epochs,
        )
    else:
        raise NameError(f"No experiment with the name '{args.exp_to_run}'")


if __name__ == "__main__":
    main()
