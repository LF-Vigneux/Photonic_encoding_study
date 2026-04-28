from pathlib import Path

import matplotlib.pyplot as plt


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


def plot_complexity_comparison(
    results: dict,
    *,
    dataset_name: str = "",
    figsize: tuple[float, float] = (10.0, 5.0),
    run_dir: Path | None = None,
    filename: str = "complexity_comparison.pdf",
) -> Path:
    """Bar chart comparing classical and induced encoding complexities.

    One bar is drawn for the classical complexity (distinct colour) and one
    bar per encoding key found in ``results["induced"]``.

    Parameters
    ----------
    results
        Dict with keys ``"classical"`` (a float) and ``"induced"`` (a dict
        mapping encoding name → float), as returned by
        ``dataset_complexity_induced_comparison``.
    dataset_name
        Optional dataset label appended to the figure title.
    figsize, run_dir, filename
        Standard plotting/output configuration.

    Returns
    -------
    Path
        Absolute path of the saved PDF.
    """
    classical_value = float(results["classical"])
    induced: dict[str, float] = {k: float(v) for k, v in results["induced"].items()}

    labels = ["classical"] + list(induced.keys())
    values = [classical_value] + list(induced.values())

    classical_color = "#E15759"
    induced_color = "#4E79A7"
    colors = [classical_color] + [induced_color] * len(induced)

    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [lbl.replace("_", " ") for lbl in labels],
        rotation=30,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Complexity score", fontsize=11)
    title = "Dataset complexity per encoding"
    if dataset_name:
        title += f" — {dataset_name}"
    ax.set_title(title, fontsize=12)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Legend patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=classical_color, edgecolor="white", label="Classical"),
        Patch(facecolor=induced_color, edgecolor="white", label="Induced (quantum)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = _save_plot(fig, filename, run_dir)
    print(f"Saved complexity plot to {output_path}")
    return output_path
