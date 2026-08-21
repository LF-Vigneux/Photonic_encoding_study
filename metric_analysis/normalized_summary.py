from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

FULL_INDUCED_METRICS = [
    "hilbert_space_support_dim",
    "quantum_fisher_information_spread",
    "entanglement_entropy",
    "kernel_spectrum_flatness",
    "locality_vs_expressibility",
    "topological_invariants_of_embedding",
    "encoded_states_classes_overlap",
]

METRICS = [
    "hilbert_space_support_dim",
    "entanglement_entropy",
    "locality_vs_expressibility",
    "encoded_states_classes_overlap",
]

METRIC_LABELS = {
    "hilbert_space_support_dim": "Hilbert Space Support Dim",
    "entanglement_entropy": "Entanglement Entropy",
    "locality_vs_expressibility": "Locality vs Expressibility",
    "encoded_states_classes_overlap": "Encoded States Class Overlap",
}


def _normalize_metric(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def _bounds_for_metrics(entry: dict[str, object]) -> list[tuple[float, float]]:
    min_max = entry.get("min_max", [])
    bounds: list[tuple[float, float]] = []
    for metric_key in METRICS:
        metric_index = FULL_INDUCED_METRICS.index(metric_key)
        if (
            isinstance(min_max, list)
            and metric_index < len(min_max)
            and isinstance(min_max[metric_index], (list, tuple))
            and len(min_max[metric_index]) == 2
        ):
            bounds.append((float(min_max[metric_index][0]), float(min_max[metric_index][1])))
        else:
            bounds.append((0.0, 1.0))
    return bounds


def plot_dataset_summary(results: dict, dataset_name: str, output_path: Path) -> Path:
    induced = results.get("results", {}).get("induced", {})
    encoding_names = [k for k, v in induced.items() if isinstance(v, dict)]
    if not encoding_names:
        raise RuntimeError(f"No induced encodings available for dataset {dataset_name}")

    normalized_data: list[list[float]] = []
    valid_encoding_names: list[str] = []

    for encoding_name in encoding_names:
        encoding_result = induced[encoding_name]
        if not isinstance(encoding_result, dict):
            continue

        bounds = _bounds_for_metrics(encoding_result)
        normalized_values = []
        for metric_key, (min_val, max_val) in zip(METRICS, bounds):
            value = float(encoding_result.get(metric_key, 0.0))
            normalized_values.append(_normalize_metric(value, min_val, max_val))

        normalized_data.append(normalized_values)
        valid_encoding_names.append(encoding_name)

    if not normalized_data:
        raise RuntimeError(f"No valid encoding data for dataset {dataset_name}")

    n_encodings = len(normalized_data)
    x = np.arange(len(METRICS))
    bar_width = 0.8 / n_encodings

    fig, ax = plt.subplots(figsize=(10.0, max(5.0, 0.75 * len(METRICS))))

    for encoding_idx, (encoding_name, metric_values) in enumerate(
        zip(valid_encoding_names, normalized_data)
    ):
        offset = (encoding_idx - n_encodings / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            metric_values,
            bar_width,
            label=encoding_name.replace("_", " "),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METRIC_LABELS[metric] for metric in METRICS],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Normalized complexity (0-1)", fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_title(
        f"Reduced normalized summary for {dataset_name}",
        fontsize=13,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
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

        with open(json_files[0], "r", encoding="utf-8") as handle:
            results = json.load(handle)

        dataset_name = results.get("config", {}).get("dataset_name", dataset_dir.name)
        output_file = complexity_dir / "complexity_normalized_summary_reduced.pdf"

        try:
            result_path = plot_dataset_summary(results, dataset_name, output_file)
        except Exception as exc:
            print(f"Skipping {dataset_dir.name}: {exc}")
            continue

        print(f"Generated reduced normalized summary: {result_path}")


if __name__ == "__main__":
    main()
