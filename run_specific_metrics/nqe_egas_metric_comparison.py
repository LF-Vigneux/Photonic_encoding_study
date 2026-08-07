"""
python3 nqe_egas_metric_comparison.py --output results/nqe_egas_comparison.pdf

python3 nqe_egas_metric_comparison.py --metrics hilbert_space_support_dim entanglement_entropy locality_vs_expressibility encoded_states_classes_overlap --output results/nqe_egas_comparison.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

ENCODINGS = ["nqe", "egas"]

ENCODING_NAMES = {
    "nqe": "NQE",
    "egas": "EGAS",
}

FULL_INDUCED_METRICS = [
    "hilbert_space_support_dim",
    "quantum_fisher_information_spread",
    "entanglement_entropy",
    "kernel_spectrum_flatness",
    "locality_vs_expressibility",
    "topological_invariants_of_embedding",
    "encoded_states_classes_overlap",
]

METRIC_TITLES = {
    "hilbert_space_support_dim": "Hilbert Space Support Dimension",
    "quantum_fisher_information_spread": "Quantum Fisher Information Spread",
    "entanglement_entropy": "Entanglement Entropy",
    "kernel_spectrum_flatness": "Kernel Spectrum Flatness",
    "locality_vs_expressibility": "Locality vs Expressibility",
    "topological_invariants_of_embedding": "Topological Invariants of Embedding",
    "encoded_states_classes_overlap": "Encoded States Class Overlap",
}


def _iter_dataset_results():
    for dataset_dir in sorted(RESULTS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if "-classes" not in dataset_dir.name:
            continue

        complexity_dir = dataset_dir / "complexities"
        if not complexity_dir.exists():
            continue

        json_files = list(complexity_dir.glob("*_results.json"))
        if not json_files:
            continue

        with open(json_files[0]) as f:
            results = json.load(f)

        yield results


def collect_metric(metric_name: str, metric_index: int) -> list[dict]:
    data = []

    for results in _iter_dataset_results():
        cfg = results["config"]
        dataset = cfg["dataset_name"]
        n_features = cfg["feature_reduction"]
        n_classes = len(cfg["classes"])

        induced = results["results"]["induced"]

        for encoding in ENCODINGS:
            if encoding not in induced:
                continue

            metrics = induced[encoding]
            if metrics is None:
                continue

            value = metrics[metric_name]
            max_value = metrics["min_max"][metric_index][1]
            pct = 100 * value / max_value if max_value > 0 else 0

            data.append(
                {
                    "dataset": dataset,
                    "encoding": encoding,
                    "label": (
                        f"{dataset} ({n_features}f,{n_classes}c)"
                        f" • {ENCODING_NAMES[encoding]}"
                    ),
                    "value": value,
                    "percentage": pct,
                }
            )

    return data


def collect_all_datasets() -> list[str]:
    datasets = set()
    for results in _iter_dataset_results():
        datasets.add(results["config"]["dataset_name"])
    return sorted(datasets)


def plot_metric_subplot(
    ax,
    metric_name: str,
    metric_index: int,
    title: str,
    dataset_colors: dict[str, tuple],
) -> int:
    data = collect_metric(metric_name, metric_index)

    if not data:
        ax.axis("off")
        return 0

    data.sort(key=lambda x: x["value"])

    labels = [d["label"] for d in data]
    values = [d["value"] for d in data]
    percentages = [d["percentage"] for d in data]
    bar_colors = [dataset_colors[d["dataset"]] for d in data]

    bars = ax.barh(labels, values, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.invert_yaxis()

    ax.set_xlabel(title, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.tick_params(axis="y", labelsize=7)

    xmax = max(values) if values else 1
    for bar, value, pct in zip(bars, values, percentages):
        ax.text(
            bar.get_width() + 0.01 * xmax,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ({pct:.1f}%)",
            va="center",
            fontsize=6,
        )

    previous_dataset = data[0]["dataset"]
    for i, d in enumerate(data):
        if d["dataset"] != previous_dataset:
            ax.axhline(i - 0.5, color="black", linewidth=0.8, alpha=0.35)
            previous_dataset = d["dataset"]

    return len(data)


def plot_panel(metrics: list[str], output_path: Path) -> Path:
    datasets = collect_all_datasets()
    cmap = plt.get_cmap("tab20")
    dataset_colors = {dataset: cmap(i % 20) for i, dataset in enumerate(datasets)}

    cols = 2
    rows = (len(metrics) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5), squeeze=False)
    axes_flat = axes.flatten()

    max_bars = 0
    for idx, metric_name in enumerate(metrics):
        metric_index = FULL_INDUCED_METRICS.index(metric_name)
        n_bars = plot_metric_subplot(
            axes_flat[idx],
            metric_name,
            metric_index,
            METRIC_TITLES[metric_name],
            dataset_colors,
        )
        max_bars = max(max_bars, n_bars)

    for empty_idx in range(len(metrics), rows * cols):
        axes_flat[empty_idx].axis("off")

    fig.set_figheight(rows * max(5.0, 0.3 * max_bars))

    legend_handles = [
        Patch(facecolor=dataset_colors[dataset], edgecolor="black", label=dataset)
        for dataset in datasets
    ]
    fig.legend(
        handles=legend_handles,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(datasets), 6),
        fontsize=9,
        title_fontsize=11,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NQE and EGAS encodings across datasets, one subplot per metric."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=FULL_INDUCED_METRICS,
        help="List of metric keys to plot (one subplot each).",
    )
    parser.add_argument(
        "--output",
        default="results/nqe_egas_comparison.pdf",
        help="Output filename for the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [m for m in args.metrics if m not in METRIC_TITLES]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")

    fig_path = plot_panel(args.metrics, Path(args.output))
    print(f"Generated figure: {fig_path}")


if __name__ == "__main__":
    main()
