"""
Visualization utilities for CLCS evaluation results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_clcs_heatmap(
    matrix_df: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
) -> None:
    """Plot a 27×27 heatmap of pairwise CLCS scores.

    Parameters
    ----------
    matrix_df:
        Symmetric DataFrame of shape ``(N, N)`` produced by
        :func:`src.metrics.clcs.clcs_matrix`.  Index and columns are language
        codes.
    model_name:
        Display name for the model (used in the plot title).
    output_path:
        File path where the figure will be saved (e.g.
        ``results/figures/heatmap_xlmr-base.png``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(matrix_df)
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        matrix_df,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cmap="YlOrRd",
        annot=n <= 15,  # only annotate if small enough to be readable
        fmt=".2f",
        linewidths=0.4,
        square=True,
        cbar_kws={"label": "CLCS", "shrink": 0.8},
    )
    ax.set_title(f"Cross-Lingual Consistency Score — {model_name}", fontsize=14, pad=14)
    ax.set_xlabel("Language B", fontsize=11)
    ax.set_ylabel("Language A", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_clcs_heatmap] saved → {output_path}")


def plot_family_bar(
    family_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot a horizontal bar chart of family-level mean CLCS scores.

    Parameters
    ----------
    family_df:
        Output of :func:`src.metrics.clcs.family_clcs` with columns
        ``["family", "mean_clcs", "num_pairs"]``.
    output_path:
        File path where the figure will be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = family_df.sort_values("mean_clcs", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.55)))
    bars = ax.barh(df["family"], df["mean_clcs"], color="steelblue", edgecolor="white")

    # Annotate each bar with its value.
    for bar, val in zip(bars, df["mean_clcs"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean CLCS", fontsize=11)
    ax.set_title("CLCS by Language Family", fontsize=13, pad=12)
    ax.axvline(df["mean_clcs"].mean(), color="tomato", linestyle="--", linewidth=1.2,
               label="overall mean")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_family_bar] saved → {output_path}")


def plot_model_comparison(
    results_dict: dict[str, float],
    output_path: str | Path,
) -> None:
    """Plot a grouped bar chart comparing global CLCS across models.

    Parameters
    ----------
    results_dict:
        Mapping ``{model_key: global_clcs_score}``.
    output_path:
        File path where the figure will be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(results_dict.keys())
    scores = [results_dict[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(models, scores, color=colors[: len(models)], edgecolor="white",
                  width=0.5)

    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Global CLCS", fontsize=11)
    ax.set_title("Global Cross-Lingual Consistency Score by Model", fontsize=12, pad=12)
    ax.tick_params(axis="x", labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_model_comparison] saved → {output_path}")
