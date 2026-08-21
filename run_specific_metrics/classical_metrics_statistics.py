"""Compute statistics for the classical complexity metrics of all datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "classical_metrics_statistics.json"
EXCLUDED_DATASETS = {"moons2-2-classes"}
EXPECTED_CONFIGURATION_COUNT = 13

CLASSICAL_METRICS = (
    "distributional_entropy",
    "correlation_order",
    "kolmogorov_complexity",
    "topological_complexity",
    "wasserstein distance",
)

METRIC_SHORT_LABELS = {
    "distributional_entropy": "DE",
    "correlation_order": "CO",
    "kolmogorov_complexity": "KC",
    "topological_complexity": "TC",
    "wasserstein distance": "WD",
}


def _load_classical_metrics(
    json_path: Path,
) -> tuple[str, dict[str, float], dict[str, tuple[float, float]]]:
    """Load the classical metrics from one dataset result JSON file.

    Parameters
    ----------
    json_path : pathlib.Path
        Path to a dataset complexity result JSON file.
    Returns
    -------
    tuple[str, dict[str, float], dict[str, tuple[float, float]]]
        Dataset folder name, the five classical metric values, and their
        theoretical bounds.
    Raises
    ------
    KeyError
        If the expected JSON fields or a classical metric are missing.
    TypeError
        If a metric value is not numeric.
    ValueError
        If a metric value or bound is non-finite, or a bound has no positive
        range.
    """
    with json_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    classical_metrics = result["results"]["classical"]
    metric_values = {
        metric_name: float(classical_metrics[metric_name])
        for metric_name in CLASSICAL_METRICS
    }
    raw_bounds = classical_metrics["min_max"]
    if len(raw_bounds) != len(CLASSICAL_METRICS):
        raise ValueError(
            f"Expected {len(CLASSICAL_METRICS)} metric bounds in {json_path}, "
            f"found {len(raw_bounds)}"
        )
    metric_bounds = {
        metric_name: (float(raw_bounds[index][0]), float(raw_bounds[index][1]))
        for index, metric_name in enumerate(CLASSICAL_METRICS)
    }

    if not all(np.isfinite(value) for value in metric_values.values()):
        raise ValueError(f"Non-finite classical metric in {json_path}")
    for metric_name, (lower_bound, upper_bound) in metric_bounds.items():
        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
            raise ValueError(f"Non-finite {metric_name} bounds in {json_path}")
        if upper_bound <= lower_bound:
            raise ValueError(
                f"Invalid {metric_name} bounds ({lower_bound}, {upper_bound}) "
                f"in {json_path}"
            )

    return json_path.parents[1].name, metric_values, metric_bounds


def collect_classical_metrics(
    results_directory: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Collect classical metric values from all dataset configurations.

    Parameters
    ----------
    results_directory : pathlib.Path
        Directory containing one subdirectory per dataset configuration.
    Returns
    -------
    tuple[list[str], numpy.ndarray, numpy.ndarray]
        Dataset folder names, metric values shaped as
        ``(configuration_count, metric_count)``, and metric bounds shaped as
        ``(configuration_count, metric_count, 2)``.
    Raises
    ------
    FileNotFoundError
        If a dataset configuration has no result JSON file.
    RuntimeError
        If the number of included configurations is not 13 or more than one
        result JSON file is found for a configuration.
    """
    dataset_directories = sorted(
        directory
        for directory in results_directory.iterdir()
        if directory.is_dir() and "-classes" in directory.name
        and directory.name not in EXCLUDED_DATASETS
    )

    if len(dataset_directories) != EXPECTED_CONFIGURATION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CONFIGURATION_COUNT} dataset configurations, "
            f"found {len(dataset_directories)} in {results_directory}"
        )

    dataset_names: list[str] = []
    metric_rows: list[list[float]] = []
    metric_bound_rows: list[list[tuple[float, float]]] = []

    for dataset_directory in dataset_directories:
        complexity_directory = dataset_directory / "complexities"
        json_files = sorted(complexity_directory.glob("*_results.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No result JSON found in {complexity_directory}"
            )
        if len(json_files) != 1:
            raise RuntimeError(
                f"Expected one result JSON in {complexity_directory}, "
                f"found {len(json_files)}"
            )

        dataset_name, metric_values, metric_bounds = _load_classical_metrics(
            json_files[0]
        )
        dataset_names.append(dataset_name)
        metric_rows.append(
            [metric_values[metric_name] for metric_name in CLASSICAL_METRICS]
        )
        metric_bound_rows.append(
            [metric_bounds[metric_name] for metric_name in CLASSICAL_METRICS]
        )

    return (
        dataset_names,
        np.asarray(metric_rows, dtype=float),
        np.asarray(metric_bound_rows, dtype=float),
    )


def compute_statistics(metric_values: np.ndarray) -> dict[str, Any]:
    """Compute descriptive standard deviations and Pearson correlations.

    Parameters
    ----------
    metric_values : numpy.ndarray
        Array shaped as ``(configuration_count, metric_count)``.
    Returns
    -------
    dict[str, Any]
        Standard deviation for every metric and the Pearson correlation matrix.
    """
    standard_deviations = np.std(metric_values, axis=0, ddof=0)
    correlation_matrix = np.corrcoef(metric_values, rowvar=False)

    return {
        "standard_deviation": {
            metric_name: float(standard_deviations[index])
            for index, metric_name in enumerate(CLASSICAL_METRICS)
        },
        "pearson_correlation": {
            metric_name: {
                other_metric_name: float(correlation_matrix[index, other_index])
                for other_index, other_metric_name in enumerate(CLASSICAL_METRICS)
            }
            for index, metric_name in enumerate(CLASSICAL_METRICS)
        },
    }


def build_report(results_directory: Path) -> dict[str, Any]:
    """Build the complete classical metrics statistics report.

    Parameters
    ----------
    results_directory : pathlib.Path
        Directory containing the dataset result directories.
    Returns
    -------
    dict[str, Any]
        Dataset values, configuration count, standard deviations, and
        correlations.
    """
    dataset_names, metric_values, metric_bounds = collect_classical_metrics(
        results_directory
    )
    normalized_metric_values = (
        metric_values - metric_bounds[:, :, 0]
    ) / (metric_bounds[:, :, 1] - metric_bounds[:, :, 0])
    statistics = compute_statistics(metric_values)
    return {
        "metric_names": list(CLASSICAL_METRICS),
        "configuration_count": len(dataset_names),
        "excluded_datasets": sorted(EXCLUDED_DATASETS),
        "datasets": {
            dataset_name: {
                metric_name: float(metric_values[row_index, metric_index])
                for metric_index, metric_name in enumerate(CLASSICAL_METRICS)
            }
            for row_index, dataset_name in enumerate(dataset_names)
        },
        "normalization_definition": (
            "(value - lower_bound) / (upper_bound - lower_bound), using "
            "per-dataset theoretical bounds"
        ),
        "normalized_datasets": {
            dataset_name: {
                metric_name: float(
                    normalized_metric_values[row_index, metric_index]
                )
                for metric_index, metric_name in enumerate(CLASSICAL_METRICS)
            }
            for row_index, dataset_name in enumerate(dataset_names)
        },
        "normalized_standard_deviation": {
            metric_name: float(
                np.std(normalized_metric_values[:, metric_index], ddof=0)
            )
            for metric_index, metric_name in enumerate(CLASSICAL_METRICS)
        },
        "standard_deviation_definition": "population (ddof=0)",
        **statistics,
    }


def _metric_label(metric_name: str) -> str:
    """Return a readable label for a classical metric name.

    Parameters
    ----------
    metric_name : str
        Internal metric name used in the JSON files.
    Returns
    -------
    str
        Human-readable metric label.
    """
    return {
        "distributional_entropy": "Distributional entropy (DE)",
        "correlation_order": "Correlation order (CO)",
        "kolmogorov_complexity": "Kolmogorov complexity (KC)",
        "topological_complexity": "Topological complexity (TC)",
        "wasserstein distance": "Wasserstein distance (WD)",
    }[metric_name]


def plot_standard_deviations(
    report: dict[str, Any], output_path: Path, *, normalized: bool = False
) -> None:
    """Plot metric standard deviations on a logarithmic x-axis.

    Parameters
    ----------
    report : dict[str, Any]
        Statistics report returned by :func:`build_report`.
    output_path : pathlib.Path
        Path of the figure to create.
    normalized : bool
        Whether to plot standard deviations after theoretical-bound
        normalization. Default value is False.
    """
    standard_deviation_key = (
        "normalized_standard_deviation" if normalized else "standard_deviation"
    )
    standard_deviations = report[standard_deviation_key]
    values = [standard_deviations[metric_name] for metric_name in CLASSICAL_METRICS]
    labels = [_metric_label(metric_name) for metric_name in CLASSICAL_METRICS]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color="#4472C4", edgecolor="black")
    axis.set_xscale("log")
    axis.set_yticks(positions, labels)
    x_axis_label = (
        "Standard deviation of normalized metric (log scale)"
        if normalized
        else "Standard deviation (log scale)"
    )
    axis.set_xlabel(x_axis_label)
    axis.grid(axis="x", which="both", linestyle="--", alpha=0.35)
    axis.invert_yaxis()

    for bar, value in zip(bars, values):
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.3g}",
            va="center",
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_correlation_heatmap(report: dict[str, Any], output_path: Path) -> None:
    """Plot the Pearson correlation matrix as a heatmap.

    Parameters
    ----------
    report : dict[str, Any]
        Statistics report returned by :func:`build_report`.
    output_path : pathlib.Path
        Path of the figure to create.
    """
    correlation_matrix = np.asarray(
        [
            [
                report["pearson_correlation"][metric_name][other_metric_name]
                for other_metric_name in CLASSICAL_METRICS
            ]
            for metric_name in CLASSICAL_METRICS
        ],
        dtype=float,
    )
    labels = [METRIC_SHORT_LABELS[metric_name] for metric_name in CLASSICAL_METRICS]

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = correlation_matrix[row_index, column_index]
            text_color = "white" if abs(value) > 0.55 else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_normalized_metric_strip(report: dict[str, Any], output_path: Path) -> None:
    """Plot every normalized dataset value on one row per classical metric.

    Parameters
    ----------
    report : dict[str, Any]
        Statistics report returned by :func:`build_report`.
    output_path : pathlib.Path
        Path of the figure to create.
    Raises
    ------
    ValueError
        If the normalized dataset count differs from the report configuration
        count.
    """
    normalized_datasets = report["normalized_datasets"]
    dataset_names = list(normalized_datasets)
    if len(dataset_names) != report["configuration_count"]:
        raise ValueError(
            "Normalized dataset count does not match configuration_count"
        )

    metric_positions = np.arange(len(CLASSICAL_METRICS), dtype=float)
    dataset_offsets = np.linspace(-0.22, 0.22, len(dataset_names))
    dataset_colors = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, len(dataset_names))
    )

    figure, axis = plt.subplots(figsize=(13, 7))
    for dataset_index, dataset_name in enumerate(dataset_names):
        normalized_values = [
            normalized_datasets[dataset_name][metric_name]
            for metric_name in CLASSICAL_METRICS
        ]
        axis.scatter(
            normalized_values,
            metric_positions + dataset_offsets[dataset_index],
            s=48,
            color=dataset_colors[dataset_index],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=dataset_name,
            zorder=3,
        )

    axis.set_xlim(-0.025, 1.025)
    axis.set_xlabel("Normalized metric value", fontsize=14)
    axis.set_yticks(
        metric_positions,
        [METRIC_SHORT_LABELS[metric_name] for metric_name in CLASSICAL_METRICS],
    )
    axis.tick_params(axis="both", labelsize=16)
    axis.invert_yaxis()
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.grid(axis="y", linestyle="-", alpha=0.15)
    axis.legend(
        loc="upper left",
        fontsize=13,
        title="Dataset",
        title_fontsize=11,
        markerscale=1.15,
        frameon=True,
        facecolor="none",
        edgecolor="#777777",
        framealpha=1.0,
)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_regression(
    axis: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    dataset_names: list[str],
    colors: list[Any],
    adapted_x_values: np.ndarray,
    adapted_y_values: np.ndarray,
) -> None:
    """Plot a colored scatter and its least-squares regression line.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis on which to draw the scatter and regression.
    x_values : numpy.ndarray
        Values for the horizontal axis.
    y_values : numpy.ndarray
        Values for the vertical axis.
    x_label : str
        Horizontal axis label.
    y_label : str
        Vertical axis label.
    dataset_names : list[str]
        Dataset names associated with each point.
    colors : list[Any]
        Color associated with each point.
    adapted_x_values : numpy.ndarray
        Horizontal values used for the adapted fit.
    adapted_y_values : numpy.ndarray
        Vertical values used for the adapted fit.
    """
    axis.scatter(x_values, y_values, c=colors, edgecolors="black", linewidths=0.5)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    x_line = np.linspace(x_values.min(), x_values.max(), 100)
    y_line = slope * x_line + intercept
    correlation = float(np.corrcoef(x_values, y_values)[0, 1])
    axis.plot(
        x_line,
        y_line,
        color="black",
        linestyle="--",
        label=f"r = {correlation:.2f}",
    )

    adapted_slope, adapted_intercept = np.polyfit(
        adapted_x_values, adapted_y_values, 1
    )
    adapted_x_line = np.linspace(
        adapted_x_values.min(), adapted_x_values.max(), 100
    )
    adapted_y_line = adapted_slope * adapted_x_line + adapted_intercept
    adapted_correlation = float(
        np.corrcoef(adapted_x_values, adapted_y_values)[0, 1]
    )
    axis.plot(
        adapted_x_line,
        adapted_y_line,
        color="#D55E00",
        linestyle="-",
        linewidth=2,
        label=f"r_adapt = {adapted_correlation:.2f}",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(linestyle="--", alpha=0.35)
    axis.legend()

    for x_value, y_value, dataset_name in zip(x_values, y_values, dataset_names):
        axis.annotate(
            dataset_name,
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )


def plot_de_regressions(report: dict[str, Any], output_path: Path) -> None:
    """Plot DE versus TC and DE versus KC with original and adapted regressions.

    Parameters
    ----------
    report : dict[str, Any]
        Statistics report returned by :func:`build_report`.
    output_path : pathlib.Path
        Path of the figure to create.
    """
    datasets = report["datasets"]
    dataset_names = list(datasets)
    distributional_entropy = np.asarray(
        [datasets[name]["distributional_entropy"] for name in dataset_names]
    )
    topological_complexity = np.asarray(
        [datasets[name]["topological_complexity"] for name in dataset_names]
    )
    kolmogorov_complexity = np.asarray(
        [datasets[name]["kolmogorov_complexity"] for name in dataset_names]
    )
    colors = [plt.get_cmap("tab20")(index) for index in range(len(dataset_names))]
    excluded_datasets = {"pathmnist8-4-classes"}
    adapted_indices = [
        index
        for index, dataset_name in enumerate(dataset_names)
        if dataset_name not in excluded_datasets
    ]
    adapted_distributional_entropy = distributional_entropy[adapted_indices]
    adapted_topological_complexity = topological_complexity[adapted_indices]
    adapted_kolmogorov_complexity = kolmogorov_complexity[adapted_indices]

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_regression(
        axes[0],
        distributional_entropy,
        topological_complexity,
        "Distributional entropy (DE)",
        "Topological complexity (TC)",
        dataset_names,
        colors,
        adapted_distributional_entropy,
        adapted_topological_complexity,
    )
    _plot_regression(
        axes[1],
        distributional_entropy,
        kolmogorov_complexity,
        "Distributional entropy (DE)",
        "Kolmogorov complexity (KC)",
        dataset_names,
        colors,
        adapted_distributional_entropy,
        adapted_kolmogorov_complexity,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_figures(report: dict[str, Any], figures_directory: Path) -> None:
    """Generate the five classical metric analysis figures.

    Parameters
    ----------
    report : dict[str, Any]
        Statistics report returned by :func:`build_report`.
    figures_directory : pathlib.Path
        Directory where the five PDF figures are written.
    """
    figures_directory.mkdir(parents=True, exist_ok=True)
    plot_standard_deviations(
        report, figures_directory / "classical_metrics_dispersion_log.pdf"
    )
    plot_standard_deviations(
        report,
        figures_directory / "classical_metrics_normalized_dispersion_log.pdf",
        normalized=True,
    )
    plot_correlation_heatmap(
        report, figures_directory / "classical_metrics_correlation_heatmap.pdf"
    )
    plot_normalized_metric_strip(
        report,
        figures_directory / "classical_metrics_normalized_strip_plot.pdf",
    )
    plot_de_regressions(report, figures_directory / "classical_metrics_de_regressions.pdf")


def main() -> None:
    """Parse command-line arguments, compute statistics, and write JSON."""
    parser = argparse.ArgumentParser(
        description="Compute classical metric standard deviations and correlations."
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing the dataset result directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--figures-directory",
        type=Path,
        default=None,
        help="Directory for the five generated PDF figures. Defaults to the JSON output directory.",
    )
    arguments = parser.parse_args()

    report = build_report(arguments.results_directory)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    figures_directory = arguments.figures_directory or arguments.output.parent
    generate_figures(report, figures_directory)

    print(f"Statistics written to {arguments.output}")
    print(f"Figures written to {figures_directory}")


if __name__ == "__main__":
    main()
