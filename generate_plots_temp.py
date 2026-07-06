#!/usr/bin/env python3
"""Temporary script to backfill min_max and generate plots from existing results JSON files."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import data_load_and_process
from dataset_complexity.plotter import (
    plot_complexity_comparison,
    plot_induced_per_encoding,
    plot_complexity_comparison_normalized,
    plot_normalized_summary,
)


def _classical_min_max(config: dict, x_all: torch.Tensor) -> list[list[float]]:
    n = int(x_all.size(0))
    max_order = int(config.get("max_order_correlation_classical", 4) or 4)
    max_dim = int(config.get("max_dim_topology_classical", 2) or 2)

    corr_bound = ((2**max_order) - 2) * float(np.log(max_order))
    topo_bound = float((n - 1) + sum(math.comb(n, k) for k in range(2, max_dim + 2)))

    max_wasserstein = float(torch.sum(torch.abs(x_all.max(dim=0).values - x_all.min(dim=0).values)))

    return [
        [0.0, float(np.log2(n))],
        [-corr_bound, corr_bound],
        [0.0, 1.0],
        [0.0, topo_bound],
        [0.0, max_wasserstein],
    ]


def _induced_min_max(
    config: dict, x_all: torch.Tensor, entry: dict
) -> list[list[float]]:
    n_full = int(x_all.size(0))
    n = min(n_full, int(config.get("max_samples_induced", n_full) or n_full))
    d = int(x_all[0].numel())
    eps = float(config.get("epsilon_hilbert_support_dim_induced", 1e-8))
    max_dim = int(config.get("max_dim_topology_induced", 2))

    hsd = float(entry.get("hilbert_space_support_dim", 1.0))
    hilbert_dim = max(1, int(math.ceil(hsd)))

    lve_den = float(np.log2(n)) if n > 1 else 1.0
    lve_bound = max(1.0, (2.0 * float(np.log2(hilbert_dim))) / lve_den)
    topo_bound = float((n - 1) + sum(math.comb(n, k) for k in range(2, max_dim + 2)))

    return [
        [1.0, float(hilbert_dim)],
        [0.0, float((4.0 * d) / (eps**2))],
        [0.0, float(np.log2(hilbert_dim))],
        [1.0, float(n)],
        [0.0, float(lve_bound)],
        [0.0, topo_bound],
    ]


def _load_x_all(config: dict) -> torch.Tensor:
    dataset_name = config["dataset_name"]
    classes = config.get("classes", None)
    feature_reduction = config.get("feature_reduction", None)

    classes_tuple = tuple(classes) if isinstance(classes, list) else classes
    x_train, x_test, _, _ = data_load_and_process(
        dataset=dataset_name,
        classes=classes_tuple,
        feature_reduction=feature_reduction,
    )
    return torch.cat((x_train, x_test), dim=0)


def generate_plots_from_results(results_path: str, output_dir: str | None = None):
    """Load results JSON, backfill min_max when missing, and generate all plots."""
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

    changed = False
    x_all = _load_x_all(config)

    classical = results.get("classical")
    if isinstance(classical, dict) and "min_max" not in classical:
        classical["min_max"] = _classical_min_max(config, x_all)
        changed = True

    induced = results.get("induced", {})
    if isinstance(induced, dict):
        for _, enc_data in induced.items():
            if isinstance(enc_data, dict) and "min_max" not in enc_data:
                enc_data["min_max"] = _induced_min_max(config, x_all, enc_data)
                changed = True

    if changed:
        data["results"] = results
        results_path.write_text(json.dumps(data, indent=2))
        print("  Backfilled missing min_max and saved JSON")

    # Set output directory
    if output_dir is None:
        output_dir = results_path.parent
    else:
        output_dir = Path(output_dir)

    print(f"Generating plots for {dataset_name} → {output_dir}")

    # Generate original stacked complexity plot
    print("  Generating stacked complexity plot...")
    plot_complexity_comparison(
        results=results,
        dataset_name=dataset_name,
        classes=(
            tuple(config["classes"])
            if isinstance(config.get("classes"), list)
            else config.get("classes")
        ),
        feature_reduction=config.get("feature_reduction"),
        run_dir=output_dir,
        filename=f"dataset_complexity_{dataset_name}_plot.pdf",
    )

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
