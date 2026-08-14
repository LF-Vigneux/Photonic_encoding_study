"""
python3 classical_metrics_table.py --output results/classical_metrics_table.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

CLASSICAL_METRICS = [
    "distributional_entropy",
    "correlation_order",
    "kolmogorov_complexity",
    "topological_complexity",
    "wasserstein distance",
]

CLASSICAL_METRIC_LABELS = {
    "distributional_entropy": "Distributional Entropy",
    "correlation_order": "Correlation Order",
    "kolmogorov_complexity": "Kolmogorov Complexity",
    "topological_complexity": "Topological Complexity",
    "wasserstein distance": "Wasserstein Distance",
}

CLASSICAL_METRIC_HEADERS = {
    "distributional_entropy": r"\shortstack{Distributional\\Entropy}",
    "correlation_order": r"\shortstack{Correlation\\Order}",
    "kolmogorov_complexity": r"\shortstack{Kolmogorov\\Complexity}",
    "topological_complexity": r"\shortstack{Topological\\Complexity}",
    "wasserstein distance": r"\shortstack{Wasserstein\\Distance}",
}


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def build_classical_metrics_table() -> pd.DataFrame:
    """Collect every normalized classical metric for every dataset, one row per dataset."""

    rows = []

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

        with open(json_files[0], "r") as f:
            results = json.load(f)

        cfg = results["config"]
        dataset_name = cfg["dataset_name"]
        n_features = cfg["feature_reduction"]
        n_classes = len(cfg["classes"])
        label = f"{dataset_name}-{n_features}-{n_classes}c"

        classical = results["results"]["classical"]
        min_max = classical["min_max"]

        row = {"Dataset": label}
        for metric_index, metric_key in enumerate(CLASSICAL_METRICS):
            value = classical[metric_key]
            min_val, max_val = min_max[metric_index]
            row[CLASSICAL_METRIC_LABELS[metric_key]] = _normalize(value, min_val, max_val)

        rows.append(row)

    columns = ["Dataset"] + [CLASSICAL_METRIC_LABELS[key] for key in CLASSICAL_METRICS]
    return pd.DataFrame(rows, columns=columns).set_index("Dataset").sort_index()


def export_classical_metrics_latex(df: pd.DataFrame, output_path: Path) -> None:
    df = df.rename(columns={CLASSICAL_METRIC_LABELS[key]: CLASSICAL_METRIC_HEADERS[key] for key in CLASSICAL_METRICS})
    latex = df.to_latex(
        float_format="%.3f",
        caption="Normalized classical complexity metrics per dataset.",
        label="tab:classical_metrics",
        escape=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table of normalized classical complexity metrics per dataset."
    )
    parser.add_argument(
        "--output",
        default="results/classical_metrics_table.tex",
        help="Output filename for the LaTeX table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    table_df = build_classical_metrics_table()
    if table_df.empty:
        raise RuntimeError("No complexity JSON files found.")

    output_path = Path(args.output)
    export_classical_metrics_latex(table_df, output_path)

    print(f"Generated table: {output_path}")


if __name__ == "__main__":
    main()
