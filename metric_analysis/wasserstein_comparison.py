from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = SCRIPT_DIR.parent / "results"
EXCLUDED_PCA_CONFIGS = {"moons-clean2-2-classes"}


def collect_wasserstein():
    """Collect classical Wasserstein distances from all datasets."""

    data = []

    for dataset_dir in sorted(RESULTS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        if "-classes" not in dataset_dir.name:
            continue

        complexity_dir = dataset_dir / "complexities"
        if not complexity_dir.exists():
            continue

        json_files = list(complexity_dir.glob("*_results.json"))
        if len(json_files) == 0:
            print(f"No json found in {complexity_dir}")
            continue

        json_file = json_files[0]

        with open(json_file, "r") as f:
            results = json.load(f)

        print(
            f'Doing dataset {results["config"]["dataset_name"]} with {results["config"]["feature_reduction"]} features and the {results["config"]["classes"]} classes.'
        )

        classical = results["results"]["classical"]

        value = classical["wasserstein distance"]

        # last min/max tuple corresponds to Wasserstein
        max_value = classical["min_max"][-1][1]

        normalized = value / max_value

        data.append(
            {
                "dataset": dataset_dir.name,
                "value": value,
                "max": max_value,
                "normalized": normalized,
                "config": results["config"],
            }
        )

    return data


def plot_raw(data):
    labels = [d["dataset"] for d in data]
    values = [d["value"] for d in data]
    percentages = [100 * d["normalized"] for d in data]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)

    plt.ylabel("Wasserstein distance")
    plt.title("Classical Wasserstein Distance")

    plt.xticks(rotation=30, ha="right")

    ymax = max(values) * 1.15

    for bar, value, pct in zip(bars, values, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "classical_wasserstein.pdf")
    plt.close()


def collect_requested_wasserstein():
    """Collect the retained PCA dataset configurations used in the study.

    Returns
    -------
    list[dict]
        Wasserstein records matched to the retained PCA configurations.
    Raises
    ------
    RuntimeError
        If a retained PCA configuration has no matching complexity result.
    """
    from data.pca_explained_variance_report import DATASET_CONFIGS

    available = collect_wasserstein()
    retained_pca_configs = {
        config_name: config
        for config_name, config in DATASET_CONFIGS.items()
        if config_name not in EXCLUDED_PCA_CONFIGS
    }
    requested = []
    used_signatures = set()

    for record in available:
        config = record["config"]
        dataset_name = config["dataset_name"]
        if dataset_name in {"noisy-moons", "noisy_moons"}:
            dataset_name = "moons"
        noise_level = (
            float(config.get("noise_generated") or 0.0)
            if dataset_name == "moons"
            else 0.0
        )
        signature = (
            dataset_name,
            tuple(config["classes"] or []),
            config["feature_reduction"],
            noise_level,
        )
        matching_configs = [
            pca_config
            for pca_config in retained_pca_configs.values()
            if (
                pca_config["dataset"],
                tuple(pca_config["classes"]),
                pca_config["n_features"],
                (
                    float(pca_config.get("noise_generated") or 0.0)
                    if pca_config["dataset"] == "moons"
                    else 0.0
                ),
            ) == signature
        ]
        if len(matching_configs) != 1 or signature in used_signatures:
            continue
        record["pca_config"] = matching_configs[0]
        record["display_name"] = matching_configs[0]["display_name"]
        used_signatures.add(signature)
        requested.append(record)

    if len(requested) != len(retained_pca_configs):
        found = {record["display_name"] for record in requested}
        missing = [
            config["display_name"]
            for config in retained_pca_configs.values()
            if config["display_name"] not in found
        ]
        raise RuntimeError(
            "Missing complexity results for the requested PCA configurations: "
            + ", ".join(missing)
        )
    return requested


def _make_umap_record(record, *, umap_state, umap_n_neighbors, umap_n_epochs, points_per_class):
    """Load one dataset and return a sampled two-dimensional UMAP projection."""
    from data.loader import data_load_and_process
    from dataset_complexity.umap import umap_data

    config = record["pca_config"]
    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=config["dataset"],
        feature_reduction=config["n_features"],
        classes=config["classes"],
        noise_generated=config.get("noise_generated", 0.0),
        shuffle=False,
    )
    features = torch.cat((x_train, x_test), dim=0)
    labels = torch.cat((y_train, y_test), dim=0)

    selected_indices = []
    for class_label in torch.unique(labels):
        class_indices = torch.where(labels == class_label)[0]
        selected_indices.append(class_indices[:points_per_class])
    selected_indices = torch.cat(selected_indices)
    features = features[selected_indices]
    labels = labels[selected_indices]
    reduced_2d, _ = umap_data(
        features,
        labels,
        umap_state=umap_state,
        n_neighbors=umap_n_neighbors,
        n_epochs=umap_n_epochs,
    )
    return {
        "dataset": record["display_name"],
        "normalized_wasserstein_distance": record["normalized"],
        "pca_explained_variance_ratio": record["pca_result"]["explained_variance_ratio"],
        "umap_embedding": reduced_2d,
        "umap_labels": labels.numpy(),
    }


def generate_combined_plot(
    *,
    output_path: Path,
    umap_state: int = 42,
    umap_n_neighbors: int = 15,
    umap_n_epochs: int = 200,
    umap_points_per_class: int = 50,
):
    """Generate the ordered Wasserstein/PCA/UMAP comparison figure."""
    from data.pca_explained_variance_report import compute_pca_explained_variance
    from dataset_complexity.plotter import plot_wasserstein_pca_umap_comparison

    records = collect_requested_wasserstein()
    for record in records:
        record["pca_result"] = compute_pca_explained_variance(
            record["dataset"], record["pca_config"]
        )

    records.sort(key=lambda record: record["normalized"])
    umap_targets = [
        next(
            record
            for record in records
            if record["display_name"] == "MNIST (4 classes)"
        ),
        next(
            record
            for record in records
            if record["display_name"] == "PathMNIST (4 classes)"
        ),
    ]
    umap_records = [
        _make_umap_record(
            record,
            umap_state=umap_state,
            umap_n_neighbors=umap_n_neighbors,
            umap_n_epochs=umap_n_epochs,
            points_per_class=umap_points_per_class,
        )
        for record in umap_targets
    ]
    # Keep only two UMAP projections in the compact figure.
    mnist_umap, pathmnist_umap = umap_records

    # The plotter needs UMAP data only for the two extreme datasets.
    plot_records = []
    for record in records:
        plot_record = {
            "dataset": record["display_name"],
            "normalized_wasserstein_distance": record["normalized"],
            "pca_explained_variance_ratio": record["pca_result"]["explained_variance_ratio"],
        }
        if record["display_name"] == mnist_umap["dataset"]:
            plot_record.update(mnist_umap)
            plot_record["umap_anchor"] = True
        elif record["display_name"] == pathmnist_umap["dataset"]:
            plot_record.update(pathmnist_umap)
            plot_record["umap_anchor"] = True
        else:
            # The plotting API validates that only extreme records need UMAP data.
            plot_record.update(
                {
                    "umap_embedding": np.zeros((1, 2)),
                    "umap_labels": np.array([0]),
                }
            )
        plot_records.append(plot_record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_wasserstein_pca_umap_comparison(
        plot_records,
        run_dir=output_path.parent,
        filename=output_path.name,
    )


def plot_normalized(data):
    labels = [d["dataset"] for d in data]
    values = [d["normalized"] for d in data]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)

    plt.ylabel("Normalized Wasserstein")
    plt.title("Normalized Classical Wasserstein Distance")

    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.05)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "classical_wasserstein_normalized.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Wasserstein distance comparison plots."
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate the combined Wasserstein/PCA/UMAP figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "wasserstein_pca_umap_comparison.pdf",
        help="Output path for the combined figure.",
    )
    parser.add_argument("--umap-state", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-epochs", type=int, default=200)
    parser.add_argument("--umap-points-per-class", type=int, default=50)
    args = parser.parse_args()

    if args.combined:
        generate_combined_plot(
            output_path=args.output,
            umap_state=args.umap_state,
            umap_n_neighbors=args.umap_neighbors,
            umap_n_epochs=args.umap_epochs,
            umap_points_per_class=args.umap_points_per_class,
        )
        print(f"Generated: {args.output}")
        return

    data = collect_wasserstein()

    if not data:
        raise RuntimeError("No complexity JSON files found.")

    # Sort independently for each plot
    raw_data = sorted(data, key=lambda d: d["value"])
    normalized_data = sorted(data, key=lambda d: d["normalized"])

    plot_raw(raw_data)
    plot_normalized(normalized_data)

    print("Generated:")
    print("  results/classical_wasserstein.pdf")
    print("  results/classical_wasserstein_normalized.pdf")


if __name__ == "__main__":
    main()
