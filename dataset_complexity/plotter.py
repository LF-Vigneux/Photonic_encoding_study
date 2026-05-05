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
        return float(v.get("total", sum(val for key, val in v.items() if key != "total")))
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
                if k not in known and k != "total":
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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = _save_plot(fig, filename, run_dir)
    print(f"Saved complexity plot to {output_path}")
    return output_path
