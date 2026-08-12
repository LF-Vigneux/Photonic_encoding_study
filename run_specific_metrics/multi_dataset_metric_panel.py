"""
python3 multi_dataset_metric_panel.py   --datasets moons mnist_binary breast_binary wine_3_classes cifar_4_classes cifar_10_classes   --metrics hilbert_space_support_dim entanglement_entropy locality_vs_expressibility encoded_states_classes_overlap   --output results/embedding_results.pdf


Others
python3 multi_dataset_metric_panel.py   --datasets noisy_moons mnist_4_classes mnist_10_classes eurosat kmnist fashion_mnist manifold path_mnist --metrics hilbert_space_support_dim entanglement_entropy locality_vs_expressibility encoded_states_classes_overlap   --output results/embedding_appendix.pdf

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Font size hyperparameters
TITLE_FONTSIZE = 25
BAR_VALUE_FONTSIZE = 16
XTICK_FONTSIZE = 16
YTICK_FONTSIZE = 16
YLABEL_FONTSIZE = 15
ENCODING_LEGEND_FONTSIZE = 24
ENCODING_LEGEND_TITLE_FONTSIZE = 26
METRIC_LEGEND_FONTSIZE = 23
METRIC_LEGEND_TITLE_FONTSIZE = 25

# Bar size hyperparameter (fraction of the per-metric slot filled by all bars)
BAR_GROUP_WIDTH = 0.75

DATASET_FOLDER_MAP = {
    "moons": "moons2-2-classes",
    "noisy_moons": "noisy_moons2-2-classes",
    "mnist_binary": "MNIST8-2-classes",
    "mnist_4_classes": "MNIST8-4-classes",
    "mnist_10_classes": "MNIST8-10-classes",
    "breast_binary": "breast8-2-classes",
    "wine_3_classes": "wine8-3-classes",
    "cifar_4_classes": "CIFAR8-4-classes",
    "cifar_10_classes": "CIFAR8-10-classes",
    "eurosat": "EuroSAT8-2-classes",
    "euro_sat": "EuroSAT8-2-classes",
    "pathmnist": "pathmnist8-4-classes",
    "path_mnist": "pathmnist8-4-classes",
    "path-mnist": "pathmnist8-4-classes",
    "kmnist": "kMNIST8-2-classes",
    "manifold": "manifold8-2-classes",
    "fashion_mnist": "fashionMNSIT8-2-classes",
    "fashion_mnsit": "fashionMNSIT8-2-classes",
}

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

AVAILABLE_METRICS = {
    "hilbert_space_support_dim": "Hilbert Space Support Dim",
    "quantum_fisher_information_spread": "Quantum Fisher Information Spread",
    "entanglement_entropy": "Entanglement Entropy",
    "kernel_spectrum_flatness": "Kernel Spectrum Flatness",
    "locality_vs_expressibility": "Locality vs Expressibility",
    "topological_invariants_of_embedding": "Topological Invariants of Embedding",
    "encoded_states_classes_overlap": "Encoded States Class Overlap",
}

METRIC_ABBREVIATIONS = {
    "hilbert_space_support_dim": "HSSD",
    "quantum_fisher_information_spread": "QFI",
    "entanglement_entropy": "Ent",
    "kernel_spectrum_flatness": "KFlat",
    "locality_vs_expressibility": "Loc/Expr",
    "topological_invariants_of_embedding": "TopoInv",
    "encoded_states_classes_overlap": "ClassOv",
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


def _load_results(dataset_folder: Path) -> dict | None:
    complexity_dir = dataset_folder / "complexities"
    if not complexity_dir.exists():
        return None

    json_files = list(complexity_dir.glob("*_results.json"))
    if not json_files:
        return None

    with open(json_files[0], "r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_bounds(metric_key: str, entry: dict[str, object]) -> tuple[float, float]:
    min_max = entry.get("min_max", [])
    if metric_key not in FULL_INDUCED_METRICS:
        return 0.0, 1.0

    metric_index = FULL_INDUCED_METRICS.index(metric_key)
    if (
        isinstance(min_max, list)
        and metric_index < len(min_max)
        and isinstance(min_max[metric_index], (list, tuple))
        and len(min_max[metric_index]) == 2
    ):
        return float(min_max[metric_index][0]), float(min_max[metric_index][1])
    return 0.0, 1.0


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def collect_dataset_data(
    dataset_key: str, metrics: list[str], encodings: list[str]
) -> tuple[str, str, list[str], list[list[float]]]:
    folder_name = DATASET_FOLDER_MAP.get(dataset_key, dataset_key)
    dataset_folder = RESULTS_DIR / folder_name
    results = _load_results(dataset_folder)
    if results is None:
        raise FileNotFoundError(f"No results found for dataset key '{dataset_key}'")

    cfg = results.get("config", {})
    dataset_name = cfg.get("dataset_name", folder_name)
    classes = cfg.get("classes")
    if isinstance(classes, list):
        class_count = len(classes)
        class_label = f"{class_count} class" if class_count == 1 else f"{class_count} classes"
    else:
        class_label = ""

    induced = results.get("results", {}).get("induced", {})

    rows: list[list[float]] = []
    labels: list[str] = []

    for encoding in encodings:
        if encoding not in induced:
            continue
        encoding_result = induced[encoding]
        if not isinstance(encoding_result, dict):
            continue

        values: list[float] = []
        for metric_key in metrics:
            value = float(encoding_result.get(metric_key, 0.0))
            min_val, max_val = _metric_bounds(metric_key, encoding_result)
            values.append(_normalize(value, min_val, max_val))

        rows.append(values)
        labels.append(ENCODING_NAMES.get(encoding, encoding.replace("_", " ")))

    if not rows:
        raise ValueError(f"No valid encoding data for dataset '{dataset_key}'")

    return dataset_name, class_label, labels, rows


def plot_panel(
    dataset_keys: list[str],
    metrics: list[str],
    encodings: list[str],
    output_path: Path,
) -> Path:
    total_fig_width = 3 * 11.5
    cols = min(2, len(dataset_keys))
    num_datasets = len(dataset_keys)
    rows = (num_datasets + cols - 1) // cols
    subplot_width = total_fig_width / cols
    subplot_height = 9.5
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * subplot_width, rows * subplot_height),
        squeeze=False,
    )

    axes_flat = axes.flatten()
    color_map = plt.get_cmap("tab10")
    colors = [color_map(i) for i in range(len(encodings))]
    subplot_labels = [
        f"({chr(ord('a') + i)})" for i in range(num_datasets)
    ]

    all_encoding_labels: list[str] = []

    for idx, dataset_key in enumerate(dataset_keys):
        ax = axes_flat[idx]
        dataset_name, class_label, labels, rows_data = collect_dataset_data(
            dataset_key, metrics, encodings
        )

        x = np.arange(len(metrics))
        n_encodings = len(rows_data)
        if n_encodings == 0:
            continue
        bar_width = BAR_GROUP_WIDTH / n_encodings

        for encoding_idx, (label, values) in enumerate(zip(labels, rows_data)):
            offset = (encoding_idx - n_encodings / 2 + 0.5) * bar_width
            hatch = "//" if label in {"NQE", "EGAS"} else None
            bars = ax.bar(
                x + offset,
                values,
                bar_width,
                label=label,
                color=colors[encoding_idx % len(colors)],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.88,
                hatch=hatch,
            )
            for bar in bars:
                height = bar.get_height()
                if height >= 0.02:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.015,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=BAR_VALUE_FONTSIZE,
                        rotation=45,
                    )

        for label in labels:
            if label not in all_encoding_labels:
                all_encoding_labels.append(label)

        title = f"{subplot_labels[idx]} {dataset_name}"
        if class_label:
            title += f" ({class_label})"
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=18)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METRIC_ABBREVIATIONS[m] for m in metrics],
            rotation=45,
            ha="right",
            fontsize=XTICK_FONTSIZE,
        )
        ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
        ax.set_xlim(-0.5, len(metrics) - 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        for sep_pos in x[:-1]:
            ax.axvline(
                x=sep_pos + 0.55,
                color="black",
                linestyle=":",
                linewidth=1.5,
                alpha=0.45,
            )

        if idx % cols == 0:
            ax.set_ylabel("Normalized complexity", fontsize=YLABEL_FONTSIZE)

    for empty_idx in range(len(dataset_keys), rows * cols):
        axes_flat[empty_idx].axis("off")

    fig.subplots_adjust(top=0.91, bottom=0.15, left=0.05, right=0.99, hspace=0.30, wspace=0.15)

    last_row_count = num_datasets % cols
    if last_row_count != 0:
        full_row_left = axes_flat[0].get_position().x0
        full_row_right = axes_flat[cols - 1].get_position().x0 + axes_flat[cols - 1].get_position().width
        full_center = 0.5 * (full_row_left + full_row_right)
        last_row_start = (rows - 1) * cols
        last_axes = [axes_flat[last_row_start + i] for i in range(last_row_count)]
        current_left = last_axes[0].get_position().x0
        current_right = last_axes[-1].get_position().x0 + last_axes[-1].get_position().width
        current_center = 0.5 * (current_left + current_right)
        shift = full_center - current_center
        for ax in last_axes:
            pos = ax.get_position()
            ax.set_position([
                pos.x0 + shift,
                pos.y0,
                pos.width,
                pos.height,
            ])

    encoding_handles = []
    for i, label in enumerate(all_encoding_labels):
        hatch = "//" if label in {"NQE", "EGAS"} else None
        encoding_handles.append(
            Patch(
                facecolor=colors[i % len(colors)],
                edgecolor="black",
                hatch=hatch,
                label=label,
            )
        )
    metric_handles = [
        Line2D([0], [0], color="none", label=f"{METRIC_ABBREVIATIONS[k]}: {AVAILABLE_METRICS[k]}")
        for k in metrics
    ]

    legend_y = 1.01
    encoding_legend = fig.legend(
        handles=encoding_handles,
        title="Encodings",
        loc="upper right",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(encoding_handles),
        frameon=False,
        fontsize=ENCODING_LEGEND_FONTSIZE,
        title_fontsize=ENCODING_LEGEND_TITLE_FONTSIZE,
    )
    metric_legend = fig.legend(
        handles=metric_handles,
        title="Metrics",
        loc="upper left",
        bbox_to_anchor=(0.5, legend_y),
        ncol=2,
        frameon=False,
        fontsize=METRIC_LEGEND_FONTSIZE,
        title_fontsize=METRIC_LEGEND_TITLE_FONTSIZE,
    )
    fig.add_artist(encoding_legend)
    fig.add_artist(metric_legend)

    # Measure rendered legend widths and re-center the pair as a single block.
    fig.canvas.draw()
    inv_transform = fig.transFigure.inverted()
    encoding_bbox = encoding_legend.get_window_extent().transformed(inv_transform)
    metric_bbox = metric_legend.get_window_extent().transformed(inv_transform)
    combined_width = (encoding_bbox.x1 - encoding_bbox.x0) + (metric_bbox.x1 - metric_bbox.x0)
    split_x = 0.5 - combined_width / 2.0 + (encoding_bbox.x1 - encoding_bbox.x0)
    encoding_legend.set_bbox_to_anchor((split_x, legend_y), transform=fig.transFigure)
    metric_legend.set_bbox_to_anchor((split_x, legend_y), transform=fig.transFigure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-dataset normalized metric panel figure."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "moons",
            "mnist_binary",
            "breast_binary",
            "wine_3_classes",
            "cifar_4_classes",
            "cifar_10_classes",
        ],
        help="List of dataset keys to include.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "hilbert_space_support_dim",
            "entanglement_entropy",
            "locality_vs_expressibility",
            "encoded_states_classes_overlap",
        ],
        help="List of metric keys to plot.",
    )
    parser.add_argument(
        "--output",
        default="results/multi_dataset_metric_panel.pdf",
        help="Output filename for the figure.",
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=ENCODINGS,
        help="List of encodings to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [m for m in args.metrics if m not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")

    fig_path = plot_panel(args.datasets, args.metrics, args.encodings, Path(args.output))
    print(f"Generated figure: {fig_path}")


if __name__ == "__main__":
    main()
