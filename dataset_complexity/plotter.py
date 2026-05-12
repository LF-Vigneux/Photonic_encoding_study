from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Metric display names (ordered for the stacked bars)
_CLASSICAL_METRICS = [
    "distributional_entropy",
    "correlation_order",
    "kolmogorov_complexity",
    "topological_complexity",
]
_INDUCED_METRICS = [
    "hilbert_space_support_dim",
    "quantum_fisher_information_spread",
    "entanglement_entropy",
    "kernel_spectrum_flatness",
    "locality_vs_expressibility",
    "topological_invariants_of_embedding",
]
_METRIC_LABELS = {
    "distributional_entropy": "Distr. entropy",
    "correlation_order": "Corr. order",
    "kolmogorov_complexity": "Kolmogorov",
    "topological_complexity": "Topology",
    "hilbert_space_support_dim": "Hilbert dim",
    "quantum_fisher_information_spread": "QFI spread",
    "entanglement_entropy": "Entanglement",
    "kernel_spectrum_flatness": "Kernel flatness",
    "locality_vs_expressibility": "Locality/Expr.",
    "topological_invariants_of_embedding": "Topo. invariants",
}

# Colour palette – one per metric slot (classical uses its own 4, induced its 6)
_CLASSICAL_COLORS = ["#76b7b2", "#59a14f", "#edc948", "#b07aa1"]
_INDUCED_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]


def _save_plot(fig: plt.Figure, filename: str, run_dir: Path | None = None) -> Path:
    if run_dir is None:
        output_path = (
            Path(__file__).parent.parent.resolve()
            / "results"
            / "dataset_complexity"
            / filename
        )
    else:
        output_path = run_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def _get_total(v: Any) -> float:
    """Return the scalar total from either a metrics dict or a plain float."""
    if isinstance(v, dict):
        return float(
            v.get("total", sum(val for key, val in v.items() if key != "total"))
        )
    return float(v)


def plot_complexity_comparison(
    results: dict,
    *,
    dataset_name: str = "",
    classes: tuple | list | None = None,
    feature_reduction: int | None = None,
    figsize: tuple[float, float] = (12.0, 6.0),
    run_dir: Path | None = None,
    filename: str = "complexity_comparison.pdf",
) -> Path:
    """Stacked bar chart comparing classical and induced encoding complexities.

    Each bar is split by metric contribution.  Works with both the new dict
    format (``{"total": …, "metric_a": …, …}``) and legacy plain float values.
    """
    # ── Collect bars ─────────────────────────────────────────────────────────
    classical_raw = results["classical"]
    induced_raw: dict[str, Any] = {
        k: v for k, v in results["induced"].items() if v is not None
    }

    all_labels = ["classical"] + list(induced_raw.keys())
    n_bars = len(all_labels)

    # For each bar build an ordered list of (metric_key, value) segments
    def _segments(label: str, raw: Any):
        if isinstance(raw, dict):
            keys = _CLASSICAL_METRICS if label == "classical" else _INDUCED_METRICS
            segs = [(k, float(raw.get(k, 0.0))) for k in keys if k in raw]
            # Any unknown keys (forward-compat)
            known = set(keys)
            for k, v in raw.items():
                if (
                    k not in known
                    and k != "total"
                    and isinstance(v, (int, float, np.number))
                ):
                    segs.append((k, float(v)))
            return segs
        # Plain float fallback – single segment
        return [(label, float(raw))]

    bar_segments = {
        "classical": _segments("classical", classical_raw),
        **{k: _segments(k, v) for k, v in induced_raw.items()},
    }

    # ── Build colour maps ─────────────────────────────────────────────────────
    metric_color: dict[str, str] = {}
    for i, m in enumerate(_CLASSICAL_METRICS):
        metric_color[m] = _CLASSICAL_COLORS[i % len(_CLASSICAL_COLORS)]
    for i, m in enumerate(_INDUCED_METRICS):
        metric_color[m] = _INDUCED_COLORS[i % len(_INDUCED_COLORS)]

    # ── Draw stacked bars ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_bars)
    bar_width = 0.6

    # Track which metric keys appear (for legend)
    seen_metrics: list[str] = []

    for bar_idx, label in enumerate(all_labels):
        bottom = 0.0
        segments = bar_segments[label]
        for metric_key, seg_val in segments:
            color = metric_color.get(metric_key, "#AAAAAA")
            ax.bar(
                bar_idx,
                seg_val,
                bar_width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += seg_val
            if metric_key not in seen_metrics:
                seen_metrics.append(metric_key)
        # Total label on top
        ax.text(
            bar_idx,
            bottom,
            f"{bottom:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(
        [lbl.replace("_", " ") for lbl in all_labels],
        rotation=30,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Complexity score", fontsize=11)

    title = "Dataset complexity per encoding"
    if dataset_name:
        classes_str = (
            "full"
            if classes is None
            else "(" + ", ".join(str(c) for c in classes) + ")"
        )
        title += f" — {dataset_name}  |  classes: {classes_str}"
        if feature_reduction is not None:
            title += f"  |  PCA→{feature_reduction}"
    ax.set_title(title, fontsize=12)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(
            facecolor=metric_color.get(m, "#AAAAAA"),
            edgecolor="white",
            label=_METRIC_LABELS.get(m, m.replace("_", " ")),
        )
        for m in seen_metrics
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=8,
        loc="upper right",
        framealpha=0.8,
        ncol=2,
    )

    # Add grid for readability
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.7)

    # Keep all spines visible for box around plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    fig.tight_layout()

    output_path = _save_plot(fig, filename, run_dir)
    print(f"Saved complexity plot to {output_path}")
    return output_path


def plot_complexity_comparison_normalized(
    results: dict,
    *,
    dataset_name: str = "",
    classes: tuple | list | None = None,
    feature_reduction: int | None = None,
    figsize: tuple[float, float] = (12.0, 6.0),
    run_dir: Path | None = None,
    filename: str = "complexity_comparison_normalized.pdf",
) -> Path:
    """Stacked bar chart with normalized (0-1) metric contributions.

    Each metric is normalized using the min_max bounds of that specific entry.
    Bars show normalized metric values stacked by encoding type.
    """
    # ── Collect bars ─────────────────────────────────────────────────────────
    classical_raw = results["classical"]
    induced_raw: dict[str, Any] = {
        k: v for k, v in results["induced"].items() if v is not None
    }

    all_labels = ["classical"] + list(induced_raw.keys())
    n_bars = len(all_labels)

    # For each bar build an ordered list of (metric_key, normalized_value) segments
    def _segments_normalized(label: str, raw: Any):
        if isinstance(raw, dict):
            keys = _CLASSICAL_METRICS if label == "classical" else _INDUCED_METRICS
            bounds = _entry_bounds(raw, keys)
            segs = []
            for idx, k in enumerate(keys):
                if k in raw:
                    raw_val = float(raw.get(k, 0.0))
                    min_val, max_val = bounds[idx]
                    norm_val = _normalize_metric(raw_val, min_val, max_val)
                    segs.append((k, norm_val))
            # Any unknown scalar keys (forward-compat)
            known = set(keys)
            for k, v in raw.items():
                if (
                    k not in known
                    and k != "total"
                    and k != "min_max"
                    and isinstance(v, (int, float, np.number))
                ):
                    segs.append((k, float(v)))
            return segs
        # Plain float fallback – single normalized segment
        return [(label, 0.5)]  # Middle of normalized range

    bar_segments = {
        "classical": _segments_normalized("classical", classical_raw),
        **{k: _segments_normalized(k, v) for k, v in induced_raw.items()},
    }

    # ── Build colour maps ─────────────────────────────────────────────────────
    metric_color: dict[str, str] = {}
    for i, m in enumerate(_CLASSICAL_METRICS):
        metric_color[m] = _CLASSICAL_COLORS[i % len(_CLASSICAL_COLORS)]
    for i, m in enumerate(_INDUCED_METRICS):
        metric_color[m] = _INDUCED_COLORS[i % len(_INDUCED_COLORS)]

    # ── Draw stacked bars ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_bars)
    bar_width = 0.6

    # Track which metric keys appear (for legend)
    seen_metrics: list[str] = []

    for bar_idx, label in enumerate(all_labels):
        bottom = 0.0
        segments = bar_segments[label]
        for metric_key, seg_val in segments:
            color = metric_color.get(metric_key, "#AAAAAA")
            ax.bar(
                bar_idx,
                seg_val,
                bar_width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += seg_val
            if metric_key not in seen_metrics:
                seen_metrics.append(metric_key)
        # Total label on top (sum of normalized metrics)
        ax.text(
            bar_idx,
            bottom,
            f"{bottom:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(
        [lbl.replace("_", " ") for lbl in all_labels],
        rotation=30,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Normalized complexity (sum of normalized metrics)", fontsize=11)

    title = "Dataset complexity per encoding (normalized metrics)"
    if dataset_name:
        classes_str = (
            "full"
            if classes is None
            else "(" + ", ".join(str(c) for c in classes) + ")"
        )
        title += f" — {dataset_name}  |  classes: {classes_str}"
        if feature_reduction is not None:
            title += f"  |  PCA→{feature_reduction}"
    ax.set_title(title, fontsize=12)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(
            facecolor=metric_color.get(m, "#AAAAAA"),
            edgecolor="white",
            label=_METRIC_LABELS.get(m, m.replace("_", " ")),
        )
        for m in seen_metrics
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=8,
        loc="upper right",
        framealpha=0.8,
        ncol=2,
    )

    # Add grid for readability
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.7)

    # Keep all spines visible for box around plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    fig.tight_layout()

    output_path = _save_plot(fig, filename, run_dir)
    print(f"Saved normalized complexity plot to {output_path}")
    return output_path


def _normalize_metric(value: float, min_val: float, max_val: float) -> float:
    """Normalize a metric value to [0, 1] using min_max bounds.

    If min == max (degenerate case), return 0.5.
    """
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def _entry_bounds(raw: dict[str, Any], metrics: list[str]) -> list[tuple[float, float]]:
    """Return metric bounds for a single entry using its own min_max list.

    The min_max list is expected to follow the metric order.
    Falls back to (0.0, 1.0) when missing or malformed.
    """
    min_max = raw.get("min_max", []) if isinstance(raw, dict) else []
    bounds: list[tuple[float, float]] = []
    for i, _ in enumerate(metrics):
        if (
            isinstance(min_max, list)
            and i < len(min_max)
            and isinstance(min_max[i], (list, tuple))
            and len(min_max[i]) == 2
        ):
            bounds.append((float(min_max[i][0]), float(min_max[i][1])))
        else:
            bounds.append((0.0, 1.0))
    return bounds


def plot_induced_per_encoding(
    results: dict,
    *,
    dataset_name: str = "",
    figsize: tuple[float, float] = (14.0, 5.0),
    run_dir: Path | None = None,
) -> list[Path]:
    """Create one plot per induced encoding showing individual metrics normalized to 0-1.

    Bars show normalized values (relative to max), actual values labeled on top,
    and min/max bounds displayed below metric titles.
    Returns list of output paths created.
    """
    induced_raw: dict[str, Any] = {
        k: v for k, v in results["induced"].items() if v is not None
    }

    metrics = _INDUCED_METRICS

    output_paths = []

    for encoding_name, encoding_result in induced_raw.items():
        if not isinstance(encoding_result, dict):
            continue

        # Extract values and bounds
        values = []
        normalized_values = []
        bounds = _entry_bounds(encoding_result, metrics)
        metric_names_display = []

        for idx, metric_key in enumerate(metrics):
            val = encoding_result.get(metric_key, 0.0)
            values.append(val)

            # Normalize with this encoding's own min/max for this metric
            min_val, max_val = bounds[idx]
            norm_val = _normalize_metric(float(val), min_val, max_val)
            normalized_values.append(norm_val)

            metric_names_display.append(
                _METRIC_LABELS.get(metric_key, metric_key.replace("_", " "))
            )

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(metrics))
        bar_width = 0.6

        # Draw bars (normalized to 0-1)
        colors = [
            _INDUCED_COLORS[i % len(_INDUCED_COLORS)] for i in range(len(metrics))
        ]
        bars = ax.bar(
            x,
            normalized_values,
            bar_width,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add actual value labels on top of bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Create x-axis labels with min/max values below
        tick_labels = []
        for i, (metric_name, (min_val, max_val)) in enumerate(
            zip(metric_names_display, bounds)
        ):
            label = f"{metric_name}\nmin={min_val:.2f}\nmax={max_val:.2f}"
            tick_labels.append(label)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Normalized complexity (0-1, relative to max)", fontsize=11)
        ax.set_ylim([0, 1])
        ax.set_title(
            f"Complexity metrics for {encoding_name.replace('_', ' ')} encoding\n{dataset_name}",
            fontsize=12,
        )

        # Add grid for readability
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.7)

        # Keep all spines visible for box around plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        fig.tight_layout()

        filename = f"complexity_{encoding_name}_per_metric.pdf"
        output_path = _save_plot(fig, filename, run_dir)
        output_paths.append(output_path)
        print(f"Saved {encoding_name} per-metric plot to {output_path}")

    return output_paths


def plot_normalized_summary(
    results: dict,
    *,
    dataset_name: str = "",
    figsize: tuple[float, float] = (14.0, 6.0),
    run_dir: Path | None = None,
) -> Path:
    """Create summary plot showing normalized (0-1) metrics across all induced encodings.

    Each encoding appears as a set of bars, one per metric, all normalized to [0, 1].
    """
    induced_raw: dict[str, Any] = {
        k: v for k, v in results["induced"].items() if v is not None
    }

    # Collect all encoding names and normalize metrics
    encoding_names = list(induced_raw.keys())
    n_encodings = len(encoding_names)

    if n_encodings == 0:
        fig, ax = plt.subplots(figsize=figsize)
        title = "Normalized complexity comparison across encodings"
        if dataset_name:
            title += f" — {dataset_name}"
        ax.set_title(title, fontsize=12)
        ax.text(
            0.5,
            0.5,
            "No induced encoding results available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        fig.tight_layout()
        filename = "complexity_normalized_summary.pdf"
        output_path = _save_plot(fig, filename, run_dir)
        print(f"Saved normalized summary plot to {output_path}")
        return output_path

    metrics = _INDUCED_METRICS
    n_metrics = len(metrics)

    # normalized_data[encoding_idx][metric_idx] = normalized_value
    normalized_data = []

    for encoding_name in encoding_names:
        encoding_result = induced_raw[encoding_name]
        if not isinstance(encoding_result, dict):
            continue

        norm_metrics = []
        bounds = _entry_bounds(encoding_result, metrics)

        for idx, metric_key in enumerate(metrics):
            val = encoding_result.get(metric_key, 0.0)
            min_val, max_val = bounds[idx]
            norm_val = _normalize_metric(float(val), min_val, max_val)
            norm_metrics.append(norm_val)

        normalized_data.append(norm_metrics)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_metrics)
    bar_width = 0.8 / n_encodings

    for encoding_idx, (encoding_name, norm_metrics) in enumerate(
        zip(encoding_names, normalized_data)
    ):
        offset = (encoding_idx - n_encodings / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            norm_metrics,
            bar_width,
            label=encoding_name.replace("_", " "),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels
        for bar, val in zip(bars, norm_metrics):
            height = bar.get_height()
            if height > 0.05:  # Only show label if bar is tall enough
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Min/max reference lines
    ax.axhline(
        y=0.0, color="red", linestyle=":", linewidth=1.5, alpha=0.5, label="Min (0)"
    )
    ax.axhline(
        y=1.0, color="green", linestyle=":", linewidth=1.5, alpha=0.5, label="Max (1)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_METRIC_LABELS.get(m, m.replace("_", " ")) for m in metrics],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Normalized complexity (0-1)", fontsize=11)
    ax.set_ylim([0, 1])

    title = "Normalized complexity comparison across encodings"
    if dataset_name:
        title += f" — {dataset_name}"
    ax.set_title(title, fontsize=12)

    ax.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.9)

    # Add grid for readability
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.7)

    # Keep all spines visible for box around plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    fig.tight_layout()

    filename = "complexity_normalized_summary.pdf"
    output_path = _save_plot(fig, filename, run_dir)
    print(f"Saved normalized summary plot to {output_path}")
    return output_path
