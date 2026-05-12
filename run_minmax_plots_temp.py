#!/usr/bin/env python3
"""Temporary helper to backfill per-entry min_max and regenerate complexity plots.

This script updates existing results JSON files in place:
- Adds classical min_max when missing
- Adds induced entry min_max when missing
- Regenerates plots using updated min_max
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import data_load_and_process
from dataset_complexity.plotter import (
    plot_complexity_comparison,
    plot_complexity_comparison_normalized,
    plot_induced_per_encoding,
    plot_normalized_summary,
)


def _classical_min_max(config: dict, x_all: torch.Tensor) -> list[list[float]]:
    n = int(x_all.size(0))
    max_order = int(config.get("max_order_correlation_classical", 4))
    max_dim = int(config.get("max_dim_topology_classical", 2))

    corr_bound = ((2**max_order) - 2) * float(np.log(max_order))
    topo_bound = float((n - 1) + sum(math.comb(n, k) for k in range(2, max_dim + 2)))

    return [
        [0.0, float(np.log2(n))],
        [-corr_bound, corr_bound],
        [0.0, 1.0],
        [0.0, topo_bound],
    ]


def _induced_min_max(
    config: dict, x_all: torch.Tensor, entry: dict
) -> list[list[float]]:
    # Same formulas as induced_quantum_complexity(), with hilbert_dim estimated
    # from observed support dim when the original model object is unavailable.
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


def process_results_file(results_path: Path) -> None:
    print(f"\nProcessing: {results_path}")
    if not results_path.exists():
        print("  File not found, skipping")
        return

    payload = json.loads(results_path.read_text())
    results = payload.get("results", {})
    config = payload.get("config", {})

    if not results or not config:
        print("  Missing results/config, skipping")
        return

    x_all = _load_x_all(config)

    changed = False

    classical = results.get("classical")
    if isinstance(classical, dict) and "min_max" not in classical:
        classical["min_max"] = _classical_min_max(config, x_all)
        changed = True
        print("  Added classical min_max")

    induced = results.get("induced", {})
    if isinstance(induced, dict):
        for enc_name, enc_data in induced.items():
            if isinstance(enc_data, dict) and "min_max" not in enc_data:
                enc_data["min_max"] = _induced_min_max(config, x_all, enc_data)
                changed = True
                print(f"  Added induced min_max: {enc_name}")

    if changed:
        payload["results"] = results
        results_path.write_text(json.dumps(payload, indent=2))
        print("  Saved updated results JSON")
    else:
        print("  min_max already present for all existing entries")

    out_dir = results_path.parent
    dataset_name = config.get("dataset_name", "unknown")
    classes = config.get("classes", None)
    classes_tuple = tuple(classes) if isinstance(classes, list) else classes
    feature_reduction = config.get("feature_reduction", None)

    plot_complexity_comparison(
        results=results,
        dataset_name=dataset_name,
        classes=classes_tuple,
        feature_reduction=feature_reduction,
        run_dir=out_dir,
        filename=f"dataset_complexity_{dataset_name}_plot.pdf",
    )
    plot_induced_per_encoding(results, dataset_name=dataset_name, run_dir=out_dir)
    plot_normalized_summary(results, dataset_name=dataset_name, run_dir=out_dir)
    plot_complexity_comparison_normalized(
        results,
        dataset_name=dataset_name,
        classes=classes_tuple,
        feature_reduction=feature_reduction,
        run_dir=out_dir,
        filename=f"dataset_complexity_{dataset_name}_normalized_comparison.pdf",
    )
    print("  Regenerated plots")


if __name__ == "__main__":
    targets = [
        PROJECT_ROOT / "results" / "dataset_complexity_kmnist_results.json",
        PROJECT_ROOT
        / "results"
        / "fashionMNSIT8"
        / "dataset_complexity_fashion_results.json",
        PROJECT_ROOT
        / "results"
        / "First legit results"
        / "dataset_complexity_mnist_results.json",
    ]

    for path in targets:
        process_results_file(path)

    print("\nDone.")
