"""
python3 quantum_metrics_panel.py --output-dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Font size hyperparameters
SUBPLOT_TITLE_FONTSIZE = 20
YLABEL_FONTSIZE = 11
XTICK_FONTSIZE = 12
BAR_VALUE_FONTSIZE = 10
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 13

ENCODINGS = [
    "angle",
    "dense_angle",
    "amplitude",
    "dense_amplitude",
    "fourier",
    "nqe",
    "egas",
]

ENCODING_NAMES = {
    "angle": "Angle",
    "dense_angle": "Dense Angle",
    "amplitude": "Amplitude",
    "dense_amplitude": "Dense Amp.",
    "fourier": "Fourier",
    "nqe": "NQE",
    "egas": "EGAS",
}

METRICS = {
    "hilbert_space_support_dim": (
        0,
        "Hilbert Space Support Dimension",
        "hilbert_space_support_dim_per_encoding.pdf",
    ),
    "quantum_fisher_information_spread": (
        1,
        "Quantum Fisher Information Spread",
        "quantum_fisher_information_spread_per_encoding.pdf",
    ),
    "entanglement_entropy": (
        2,
        "Entanglement Entropy",
        "entanglement_entropy_per_encoding.pdf",
    ),
    "kernel_spectrum_flatness": (
        3,
        "Kernel Spectrum Flatness",
        "kernel_spectrum_flatness_per_encoding.pdf",
    ),
    "locality_vs_expressibility": (
        4,
        "Locality vs Expressibility",
        "locality_vs_expressibility_per_encoding.pdf",
    ),
    "topological_invariants_of_embedding": (
        5,
        "Topological Invariants of Embedding",
        "topological_invariants_of_embedding_per_encoding.pdf",
    ),
    "encoded_states_classes_overlap": (
        6,
        "Encoded States Class Overlap",
        "encoded_states_classes_overlap_per_encoding.pdf",
    ),
}


def collect_metric_by_encoding(metric_name: str, metric_index: int) -> list[dict]:
    """One entry per encoding, holding the per-dataset values for a single metric."""

    per_encoding = {encoding: {"labels": [], "values": [], "percentages": []} for encoding in ENCODINGS}

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

        cfg = results["config"]

        dataset = cfg["dataset_name"]
        n_features = cfg["feature_reduction"]
        n_classes = len(cfg["classes"])
        label = f"{dataset}-{n_features}-{n_classes}c"

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

            per_encoding[encoding]["labels"].append(label)
            per_encoding[encoding]["values"].append(value)
            per_encoding[encoding]["percentages"].append(pct)

    encodings_data = []
    for encoding in ENCODINGS:
        entry = per_encoding[encoding]
        if not entry["labels"]:
            continue

        order = sorted(range(len(entry["labels"])), key=lambda i: entry["labels"][i])
        encodings_data.append(
            {
                "encoding": encoding,
                "labels": [entry["labels"][i] for i in order],
                "values": [entry["values"][i] for i in order],
                "percentages": [entry["percentages"][i] for i in order],
            }
        )

    return encodings_data


def collect_all_dataset_labels() -> list[str]:
    labels = set()

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

        cfg = results["config"]
        dataset = cfg["dataset_name"]
        n_features = cfg["feature_reduction"]
        n_classes = len(cfg["classes"])
        labels.add(f"{dataset}-{n_features}-{n_classes}c")

    return sorted(labels)


def plot_encoding_subplot(
    ax,
    encoding_entry: dict,
    dataset_colors: dict[str, tuple],
) -> None:
    labels = encoding_entry["labels"]
    values = encoding_entry["values"]
    percentages = encoding_entry["percentages"]

    x = range(len(labels))
    bar_colors = [dataset_colors[label] for label in labels]

    bars = ax.bar(x, values, width=0.65, color=bar_colors, edgecolor="black", linewidth=0.4)

    ax.set_title(
        ENCODING_NAMES[encoding_entry["encoding"]],
        fontsize=SUBPLOT_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=XTICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ymax = max(values) if values else 1
    for bar, value, pct in zip(bars, values, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * ymax,
            f"{value:.2f} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=BAR_VALUE_FONTSIZE,
            rotation=90,
        )
    ax.set_ylim(0, ymax * 1.3)


def plot_metric_file(metric_name: str, metric_index: int, title: str, output_path: Path) -> Path:
    encodings_data = collect_metric_by_encoding(metric_name, metric_index)

    if not encodings_data:
        raise RuntimeError(metric_name)

    dataset_labels = collect_all_dataset_labels()
    cmap = plt.get_cmap("tab20")
    dataset_colors = {label: cmap(i % 20) for i, label in enumerate(dataset_labels)}

    cols = 2
    num_encodings = len(encodings_data)
    rows = (num_encodings + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9, rows * 8), squeeze=False)
    axes_flat = axes.flatten()

    for idx, encoding_entry in enumerate(encodings_data):
        plot_encoding_subplot(axes_flat[idx], encoding_entry, dataset_colors)
        if idx % cols == 0:
            axes_flat[idx].set_ylabel(title, fontsize=YLABEL_FONTSIZE)

    for empty_idx in range(num_encodings, rows * cols):
        axes_flat[empty_idx].axis("off")

    legend_handles = [
        Patch(facecolor=dataset_colors[label], edgecolor="black", label=label)
        for label in dataset_labels
    ]

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.04)
    fig.legend(
        handles=legend_handles,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(legend_handles), 6),
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For each metric, plot one subplot per dataset comparing encodings."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS.keys()),
        help="List of metric keys to generate a file for.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where the per-metric figures are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [m for m in args.metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")

    output_dir = Path(args.output_dir)

    for metric_name in args.metrics:
        metric_index, title, filename = METRICS[metric_name]
        fig_path = plot_metric_file(metric_name, metric_index, title, output_dir / filename)
        print(f"Generated figure: {fig_path}")


if __name__ == "__main__":
    main()
