from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

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

# ENCODINGS = [
#     "angle",
#     "amplitude",
# ]
# ENCODING_NAMES = {
#     "angle": "Angle",
#     "amplitude": "Amplitude",
# }


METRICS = {
    "hilbert_space_support_dim": (
        0,
        "Hilbert Space Support Dimension",
        "hilbert_space_support_dim.pdf",
    ),
    "quantum_fisher_information_spread": (
        1,
        "Quantum Fisher Information Spread",
        "quantum_fisher_information_spread.pdf",
    ),
    "entanglement_entropy": (
        2,
        "Entanglement Entropy",
        "entanglement_entropy.pdf",
    ),
    "kernel_spectrum_flatness": (
        3,
        "Kernel Spectrum Flatness",
        "kernel_spectrum_flatness.pdf",
    ),
    "locality_vs_expressibility": (
        4,
        "Locality vs Expressibility",
        "locality_vs_expressibility.pdf",
    ),
    "topological_invariants_of_embedding": (
        5,
        "Topological Invariants of Embedding",
        "topological_invariants_of_embedding.pdf",
    ),
    "encoded_states_classes_overlap": (
        6,
        "Encoded States Class Overlap",
        "encoded_states_classes_overlap.pdf",
    ),
}


def collect_metric(metric_name: str, metric_index: int):
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
        if not json_files:
            continue

        with open(json_files[0]) as f:
            results = json.load(f)

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


def plot_metric(metric_name, metric_index, ylabel, filename):

    data = collect_metric(metric_name, metric_index)

    if not data:
        raise RuntimeError(metric_name)

    # Sort globally by value (smallest -> largest)
    data.sort(key=lambda x: x["value"])

    datasets = sorted({d["dataset"] for d in data})

    cmap = plt.get_cmap("tab20")
    colors = {dataset: cmap(i % 20) for i, dataset in enumerate(datasets)}
    legend_handles = [
        Patch(facecolor=colors[dataset], edgecolor="black", label=dataset)
        for dataset in datasets
    ]

    labels = [d["label"] for d in data]
    values = [d["value"] for d in data]
    percentages = [d["percentage"] for d in data]
    bar_colors = [colors[d["dataset"]] for d in data]

    fig_height = max(8, 0.34 * len(labels))

    plt.figure(figsize=(16, fig_height))

    bars = plt.barh(
        labels,
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.4,
    )

    # Put smallest at the top
    plt.gca().invert_yaxis()

    plt.xlabel(ylabel, fontsize=12)
    plt.title(ylabel, fontsize=15)

    plt.grid(axis="x", linestyle="--", alpha=0.3)

    xmax = max(values)

    for bar, value, pct in zip(bars, values, percentages):
        plt.text(
            value + 0.01 * xmax,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ({pct:.1f}%)",
            va="center",
            fontsize=8,
        )

    # Draw separators when the dataset changes
    previous_dataset = data[0]["dataset"]

    for i, d in enumerate(data):
        if d["dataset"] != previous_dataset:
            plt.axhline(i - 0.5, color="black", linewidth=0.8, alpha=0.35)
            previous_dataset = d["dataset"]
    plt.legend(
        handles=legend_handles,
        title="Dataset",
        loc="upper right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )

    plt.tight_layout()

    outfile = RESULTS_DIR / filename
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated {outfile}")


def main():

    for metric_name, (
        metric_index,
        ylabel,
        filename,
    ) in METRICS.items():

        plot_metric(
            metric_name,
            metric_index,
            ylabel,
            filename,
        )


if __name__ == "__main__":
    main()
