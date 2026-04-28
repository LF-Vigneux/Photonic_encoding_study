import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from nn_embedding.lib.figure_reproductions import (  # noqa: E402
    reproduce_figure_2,
    reproduce_figure_3,
    reproduce_figure_4,
    reproduce_figure_5,
    reproduce_figure_6,
)
from nn_embedding.utils.utils import (  # noqa: E402
    parse_args,
    str_to_bool,
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
    use_merlin = str_to_bool(cfg.get("use_merlin", False))
    generate_graph = not str_to_bool(cfg.get("dont_generate_graph", False))

    if exp_to_run == "FIG2":
        print("Running the FIG2 experiment")
        reproduce_figure_2(
            dataset=cfg.get("dataset", "mnist"),
            use_merlin=use_merlin,
            batch_size=cfg.get("batch_size", 100),
            num_epochs_training_embedding=cfg.get("num_epochs_training_embedding", 50),
            num_epochs_training_classifier=cfg.get(
                "num_epochs_training_classifier", 1000
            ),
            lr=cfg.get("lr", 0.01),
            distance=cfg.get("distance", "Trace"),
            samples_per_class=cfg.get("samples_per_class", 150),
            num_classes=cfg.get("num_classes", 2),
            num_repetitions=cfg.get("num_repetitions", 5),
            run_dir=run_dir,
            generate_graph=generate_graph,
        )
    elif exp_to_run == "FIG3":
        print("Running the FIG3 experiment")
        layers = cfg.get("layers_to_test", None)
        if isinstance(layers, str):
            layers = list(map(int, layers.split(",")))
        elif isinstance(layers, list):
            layers = [int(x) for x in layers]
        reproduce_figure_3(
            dataset=cfg.get("dataset", "mnist"),
            use_merlin=use_merlin,
            batch_size=cfg.get("batch_size", 100),
            num_epochs_training_embedding=cfg.get("num_epochs_training_embedding", 50),
            num_epochs_training_classifier=cfg.get(
                "num_epochs_training_classifier", 1000
            ),
            lr=cfg.get("lr", 0.01),
            distance=cfg.get("distance", "Trace"),
            samples_per_class=cfg.get("samples_per_class", 150),
            num_classes=cfg.get("num_classes", 2),
            num_repetitions=cfg.get("num_repetitions", 5),
            layers_to_test=layers,
            run_dir=run_dir,
            generate_graph=generate_graph,
        )
    elif exp_to_run == "FIG4":
        print("Running the FIG4 experiment")
        reproduce_figure_4(
            use_merlin=use_merlin,
            batch_size=cfg.get("batch_size", 25),
            num_epochs_training_embedding=cfg.get("num_epochs_training_embedding", 100),
            lr=cfg.get("lr", 0.01),
            distance=cfg.get("distance", "Trace"),
            samples_per_dataset=cfg.get("samples_per_dataset", 400),
            num_datasets=cfg.get("num_datasets", 10),
            num_repetitions_per_dataset=cfg.get("num_repetitions_per_dataset", 20),
            epsilon=cfg.get("epsilon", 0.01),
            num_samples_int=cfg.get("num_samples_int", 100),
            run_dir=run_dir,
            generate_graph=generate_graph,
        )
    elif exp_to_run == "FIG5":
        print("Running the FIG5 experiment")
        weights = cfg.get("weights", None)
        if isinstance(weights, str):
            weights = list(map(float, weights.split(",")))
        elif isinstance(weights, list):
            weights = [float(w) for w in weights]
        reproduce_figure_5(
            dataset=cfg.get("dataset", "mnist"),
            use_merlin=use_merlin,
            batch_size=cfg.get("batch_size", 100),
            num_epochs_training_embedding=cfg.get(
                "num_epochs_training_embedding", 1000
            ),
            lr=cfg.get("lr", 0.01),
            distance=cfg.get("distance", "Trace"),
            samples_per_class=cfg.get("samples_per_class", 500),
            num_repetitions=cfg.get("num_repetitions", 5),
            weights=weights,
            run_dir=run_dir,
            generate_graph=generate_graph,
        )
    elif exp_to_run == "FIG6":
        print("Running the FIG6 experiment")
        reproduce_figure_6(
            dataset=cfg.get("dataset", "mnist"),
            use_merlin=use_merlin,
            batch_size=cfg.get("batch_size", 100),
            num_epochs_training_embedding=cfg.get("num_epochs_training_embedding", 50),
            lr=cfg.get("lr", 0.01),
            distance=cfg.get("distance", "Trace"),
            samples_per_class=cfg.get("samples_per_class", 150),
            num_repetitions=cfg.get("num_repetitions", 5),
            run_dir=run_dir,
            generate_graph=generate_graph,
        )
    elif exp_to_run == "INDUCED_DATASET_COMPLEXITY":
        print("Running the DATASET_COMPLEXITY experiment")
        from run_dataset_complexity import (
            dataset_complexity_induced_comparison,
        )  # noqa: E402

        classes_raw = cfg.get("classes", None)
        if isinstance(classes_raw, list):
            classes_raw = tuple(classes_raw)

        hyper_cls_raw = cfg.get("hyper_parameters_classical", [1, 1, 1, 1])
        hyper_ind_raw = cfg.get("hyper_parameters_induced", [1, 1, 1, 1, 1, 1])
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
            run_dir=run_dir,
        )
    else:
        raise NameError(f"No experiment with the name '{exp_to_run}'")


def main():
    args = parse_args()
    generate_graph = not args.dont_generate_graph

    if args.exp_to_run == "FIG2":
        print("Running the FIG2 experiment")
        reproduce_figure_2(
            dataset=args.dataset,
            use_merlin=args.use_merlin,
            batch_size=args.batch_size,
            num_epochs_training_embedding=args.num_epochs_training_embedding,
            num_epochs_training_classifier=args.num_epochs_training_classifier,
            lr=args.lr,
            distance=args.distance,
            samples_per_class=args.samples_per_class,
            num_classes=args.num_classes,
            num_repetitions=args.num_repetitions,
            generate_graph=generate_graph,
        )
    elif args.exp_to_run == "FIG3":
        print("Running the FIG3 experiment")
        reproduce_figure_3(
            dataset=args.dataset,
            use_merlin=args.use_merlin,
            batch_size=args.batch_size,
            num_epochs_training_embedding=args.num_epochs_training_embedding,
            num_epochs_training_classifier=args.num_epochs_training_classifier,
            lr=args.lr,
            distance=args.distance,
            samples_per_class=args.samples_per_class,
            num_classes=args.num_classes,
            num_repetitions=args.num_repetitions,
            layers_to_test=args.layers_to_test,
            generate_graph=generate_graph,
        )
    elif args.exp_to_run == "FIG4":
        print("Running the FIG4 experiment")
        reproduce_figure_4(
            use_merlin=args.use_merlin,
            batch_size=args.batch_size,
            num_epochs_training_embedding=args.num_epochs_training_embedding,
            lr=args.lr,
            distance=args.distance,
            samples_per_dataset=args.samples_per_dataset,
            num_datasets=args.num_datasets,
            num_repetitions_per_dataset=args.num_repetitions_per_dataset,
            epsilon=args.epsilon,
            num_samples_int=args.num_samples_int,
            generate_graph=generate_graph,
        )
    elif args.exp_to_run == "FIG5":
        print("Running the FIG5 experiment")
        reproduce_figure_5(
            dataset=args.dataset,
            use_merlin=args.use_merlin,
            batch_size=args.batch_size,
            num_epochs_training_embedding=args.num_epochs_training_embedding,
            lr=args.lr,
            distance=args.distance,
            samples_per_class=args.samples_per_class,
            num_repetitions=args.num_repetitions,
            weights=args.weights,
            generate_graph=generate_graph,
        )
    elif args.exp_to_run == "FIG6":
        print("Running the FIG6 experiment")
        reproduce_figure_6(
            dataset=args.dataset,
            use_merlin=args.use_merlin,
            batch_size=args.batch_size,
            num_epochs_training_embedding=args.num_epochs_training_embedding,
            lr=args.lr,
            distance=args.distance,
            samples_per_class=args.samples_per_class,
            num_repetitions=args.num_repetitions,
            generate_graph=generate_graph,
        )
    else:
        raise NameError(f"No experiment with the name '{args.exp_to_run}'")


if __name__ == "__main__":
    main()
