#!/usr/bin/env python3
"""Temporary script to generate plots from existing results JSON files."""

import json
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_complexity.plotter import (
    plot_induced_per_encoding,
    plot_normalized_summary,
    plot_complexity_comparison_normalized,
)


def generate_plots_from_results(results_path: str, output_dir: str | None = None):
    """Load results JSON and generate all plots."""
    results_path = Path(results_path)

    if not results_path.exists():
        print(f"Error: {results_path} does not exist")
        return

    # Load results
    print(f"Loading results from {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    config = data.get("config", {})
    dataset_name = config.get("dataset_name", "unknown")

    # Set output directory
    if output_dir is None:
        output_dir = results_path.parent
    else:
        output_dir = Path(output_dir)

    print(f"Generating plots for {dataset_name} → {output_dir}")

    # Generate per-encoding plots
    print("  Generating per-encoding plots...")
    plot_induced_per_encoding(
        results,
        dataset_name=dataset_name,
        run_dir=output_dir,
    )

    # Generate normalized summary plot
    print("  Generating normalized summary plot...")
    plot_normalized_summary(
        results,
        dataset_name=dataset_name,
        run_dir=output_dir,
    )

    # Generate normalized stacked bar chart
    print("  Generating normalized complexity comparison plot...")
    plot_complexity_comparison_normalized(
        results,
        dataset_name=dataset_name,
        run_dir=output_dir,
    )

    print(f"✓ Done for {dataset_name}")


if __name__ == "__main__":
    # Generate plots for fashion
    generate_plots_from_results(
        "results/fashionMNSIT8/dataset_complexity_fashion_results.json",
        output_dir="results/fashionMNSIT8",
    )

    # Generate plots for mnist
    generate_plots_from_results(
        "results/First legit results/dataset_complexity_mnist_results.json",
        output_dir="results/First legit results",
    )

    print("\n✓ All plots generated successfully!")
