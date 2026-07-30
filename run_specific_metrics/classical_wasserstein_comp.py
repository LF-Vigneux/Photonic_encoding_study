from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"


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
