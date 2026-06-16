#!/usr/bin/env python3
"""
Phase 2 full analysis: CLCS matrices, family analysis, figures, statistical tests.

Outputs
-------
results/scores/
    {model}_pairwise_clcs.csv       — 210 language pairs
    {model}_family_clcs.csv         — within-family mean CLCS
    {model}_clcs_matrix.csv         — 21×21 symmetric matrix
    {model}_worst_pairs.csv         — bottom-10 pairs by CLCS
    {model}_family_comparison.csv   — within vs cross-family mean CLCS
    {model}_worst_languages.csv     — per-language mean CLCS ranked asc
    global_clcs.json                — global CLCS for all 3 models
    statistical_tests.json          — Wilcoxon / MWU test results

results/figures/
    heatmap_{model}.png             — 21×21 heatmap ordered by family (300 dpi)
    family_bar_comparison.png       — family CLCS bar chart, 3 subplots (300 dpi)
    model_comparison.png            — global CLCS + std error bars (300 dpi)
    worst_pairs.png                 — bottom-10 pairs, 3 subplots (300 dpi)
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

from src.data.loader import LANGUAGE_FAMILIES
from src.metrics.clcs import (
    clcs_matrix,
    compute_all_pairwise,
    family_clcs,
    global_clcs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = ["mbert", "xlmr-base", "mdeberta", "llama3.1", "aya-expanse"]
PREDICTIONS_DIR = _REPO_ROOT / "results" / "predictions"
SCORES_DIR = _REPO_ROOT / "results" / "scores"
FIGURES_DIR = _REPO_ROOT / "results" / "figures"

# Extend LANGUAGE_FAMILIES with languages present in the corpus but missing
# from the base dict (el = Greek/Hellenic, id = Indonesian/Austronesian).
EXTENDED_FAMILIES: dict[str, str] = {
    **LANGUAGE_FAMILIES,
    "el": "Hellenic",
    "id": "Austronesian",
}

# Consistent color palette for models across all figures
MODEL_COLORS = {
    "mbert": "#4C72B0",
    "xlmr-base": "#55A868",
    "mdeberta": "#DD8452",
    "llama3.1": "#C44E52",
    "aya-expanse": "#8172B2",
}

# ---------------------------------------------------------------------------
# Section 1 — Data loading
# ---------------------------------------------------------------------------


def load_predictions(model_key: str) -> dict[str, dict[int, int]]:
    """Load all prediction checkpoints for *model_key*.

    Returns ``{lang: {source_id: label}}``.
    """
    result: dict[str, dict[int, int]] = {}
    for path in sorted(PREDICTIONS_DIR.glob(f"{model_key}_*.json")):
        lang = path.stem[len(model_key) + 1:]
        data = json.loads(path.read_text(encoding="utf-8"))
        result[lang] = dict(zip(data["source_ids"], data["labels"]))
    return result


def align_predictions(
    preds_by_lang: dict[str, dict[int, int]],
) -> dict[str, list[int]]:
    """Align all languages to the intersection of source_ids."""
    common_ids = sorted(
        set.intersection(*[set(ids) for ids in preds_by_lang.values()])
    )
    return {
        lang: [id_map[sid] for sid in common_ids]
        for lang, id_map in preds_by_lang.items()
    }


# ---------------------------------------------------------------------------
# Section 2 — T2.1 score matrix
# ---------------------------------------------------------------------------


def compute_and_save_t21(
    model_key: str,
    aligned: dict[str, list[int]],
    languages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    """Compute and persist all T2.1 outputs for one model."""
    pairwise_df = compute_all_pairwise(aligned, languages)
    pairwise_df.to_csv(SCORES_DIR / f"{model_key}_pairwise_clcs.csv", index=False)

    fam_df = family_clcs(pairwise_df, EXTENDED_FAMILIES)
    fam_df.to_csv(SCORES_DIR / f"{model_key}_family_clcs.csv", index=False)

    g = global_clcs(pairwise_df)

    mat_df = clcs_matrix(pairwise_df, languages)
    mat_df.to_csv(SCORES_DIR / f"{model_key}_clcs_matrix.csv")

    return pairwise_df, fam_df, g, mat_df


def save_global_clcs_json(global_scores: dict[str, float]) -> None:
    path = SCORES_DIR / "global_clcs.json"
    path.write_text(json.dumps(global_scores, indent=2), encoding="utf-8")
    print(f"  Saved → {path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Section 3 — T2.2 cross-family analysis
# ---------------------------------------------------------------------------


def compute_worst_pairs(pairwise_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        pairwise_df.nsmallest(n, "clcs")[["lang_a", "lang_b", "clcs"]]
        .reset_index(drop=True)
    )


def compute_family_comparison(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Within-family vs cross-family mean CLCS."""
    df = pairwise_df.copy()
    df["family_a"] = df["lang_a"].map(EXTENDED_FAMILIES)
    df["family_b"] = df["lang_b"].map(EXTENDED_FAMILIES)
    within = df[df["family_a"] == df["family_b"]]["clcs"]
    cross = df[df["family_a"] != df["family_b"]]["clcs"]
    return pd.DataFrame(
        [
            {"family_type": "within_family", "mean_clcs": within.mean(), "n_pairs": len(within)},
            {"family_type": "cross_family", "mean_clcs": cross.mean(), "n_pairs": len(cross)},
        ]
    )


def compute_worst_languages(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Mean CLCS per language across all pairs it appears in, sorted ascending."""
    a = pairwise_df[["lang_a", "clcs"]].rename(columns={"lang_a": "lang"})
    b = pairwise_df[["lang_b", "clcs"]].rename(columns={"lang_b": "lang"})
    combined = pd.concat([a, b], ignore_index=True)
    agg = (
        combined.groupby("lang")["clcs"]
        .agg(mean_clcs="mean", n_pairs="count")
        .reset_index()
        .sort_values("mean_clcs", ascending=True)
        .reset_index(drop=True)
    )
    return agg


def save_t22_outputs(
    model_key: str,
    pairwise_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    worst_pairs = compute_worst_pairs(pairwise_df)
    worst_pairs.to_csv(SCORES_DIR / f"{model_key}_worst_pairs.csv", index=False)

    fam_cmp = compute_family_comparison(pairwise_df)
    fam_cmp.to_csv(SCORES_DIR / f"{model_key}_family_comparison.csv", index=False)

    worst_langs = compute_worst_languages(pairwise_df)
    worst_langs.to_csv(SCORES_DIR / f"{model_key}_worst_languages.csv", index=False)

    print(f"  Most inconsistent language: {worst_langs.iloc[0]['lang']} "
          f"(mean CLCS={worst_langs.iloc[0]['mean_clcs']:.4f})")

    return {
        "worst_pairs": worst_pairs,
        "family_comparison": fam_cmp,
        "worst_languages": worst_langs,
    }


# ---------------------------------------------------------------------------
# Section 4 — T2.4 statistical tests
# ---------------------------------------------------------------------------


def _rank_biserial_signed_rank(W: float, n_nonzero: int) -> float:
    if n_nonzero == 0:
        return 0.0
    return 1.0 - (2.0 * W) / (n_nonzero * (n_nonzero + 1))


def _rank_biserial_mwu(U: float, n1: int, n2: int) -> float:
    denom = n1 * n2
    if denom == 0:
        return 0.0
    return 1.0 - (2.0 * U) / denom


def run_model_comparison_tests(
    all_pairwise: dict[str, pd.DataFrame],
) -> list[dict]:
    """Wilcoxon signed-rank tests between each pair of models (aligned on common pairs)."""
    results = []
    for m1, m2 in combinations(MODELS, 2):
        merged = all_pairwise[m1].merge(
            all_pairwise[m2], on=["lang_a", "lang_b"], suffixes=("_m1", "_m2")
        )
        x = merged["clcs_m1"].values
        y = merged["clcs_m2"].values
        diffs = x - y
        n_nonzero = int((diffs != 0).sum())
        if n_nonzero < 10:
            stat, p = float("nan"), float("nan")
            r = float("nan")
        else:
            res = wilcoxon(x, y)
            stat = float(res.statistic)
            p = float(res.pvalue)
            r = _rank_biserial_signed_rank(stat, n_nonzero)
        results.append(
            {
                "test": "wilcoxon_signed_rank",
                "model_a": m1,
                "model_b": m2,
                "W_statistic": stat,
                "pvalue": p,
                "n_nonzero_diffs": n_nonzero,
                "effect_r": r,
                "significant_0.05": bool(p < 0.05) if not np.isnan(p) else None,
            }
        )
    return results


def run_within_cross_family_tests(
    all_pairwise: dict[str, pd.DataFrame],
) -> list[dict]:
    """Mann-Whitney U test: within-family vs cross-family CLCS for each model."""
    results = []
    for model_key in MODELS:
        df = all_pairwise[model_key].copy()
        df["family_a"] = df["lang_a"].map(EXTENDED_FAMILIES)
        df["family_b"] = df["lang_b"].map(EXTENDED_FAMILIES)
        within = df[df["family_a"] == df["family_b"]]["clcs"].values
        cross = df[df["family_a"] != df["family_b"]]["clcs"].values
        n1, n2 = len(within), len(cross)
        res = mannwhitneyu(within, cross, alternative="two-sided")
        U = float(res.statistic)
        p = float(res.pvalue)
        r = _rank_biserial_mwu(U, n1, n2)
        direction = "within > cross" if within.mean() > cross.mean() else "cross > within"
        results.append(
            {
                "test": "mann_whitney_u",
                "model": model_key,
                "U_statistic": U,
                "pvalue": p,
                "n_within": n1,
                "n_cross": n2,
                "mean_within": float(within.mean()),
                "mean_cross": float(cross.mean()),
                "effect_r": r,
                "direction": direction,
                "significant_0.05": bool(p < 0.05),
            }
        )
    return results


def save_statistical_tests(
    model_tests: list[dict],
    family_tests: list[dict],
) -> None:
    payload = {
        "model_comparisons": model_tests,
        "within_vs_cross_family": family_tests,
    }
    path = SCORES_DIR / "statistical_tests.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved → {path.relative_to(_REPO_ROOT)}")


def print_stats_summary(
    model_tests: list[dict],
    family_tests: list[dict],
) -> None:
    print("\n" + "=" * 68)
    print("MODEL COMPARISON TESTS (Wilcoxon signed-rank, paired 210 pairs)")
    print("=" * 68)
    hdr = f"{'Comparison':<28} {'W':>10} {'p-value':>12} {'r':>8}  sig?"
    print(hdr)
    print("-" * 68)
    for t in model_tests:
        cmp = f"{t['model_a']} vs {t['model_b']}"
        W = f"{t['W_statistic']:.1f}" if not np.isnan(t["W_statistic"]) else "  n/a"
        p = f"{t['pvalue']:.4e}" if not np.isnan(t["pvalue"]) else "     n/a"
        r = f"{t['effect_r']:.4f}" if not np.isnan(t["effect_r"]) else "   n/a"
        sig = "yes" if t.get("significant_0.05") else "no"
        print(f"{cmp:<28} {W:>10} {p:>12} {r:>8}  {sig}")

    print("\n" + "=" * 68)
    print("WITHIN vs CROSS-FAMILY TESTS (Mann-Whitney U)")
    print("=" * 68)
    hdr = f"{'Model':<14} {'U':>10} {'p-value':>12} {'r':>8}  {'direction':<20}  sig?"
    print(hdr)
    print("-" * 68)
    for t in family_tests:
        U = f"{t['U_statistic']:.1f}"
        p = f"{t['pvalue']:.4e}"
        r = f"{t['effect_r']:.4f}"
        sig = "yes" if t["significant_0.05"] else "no"
        print(
            f"{t['model']:<14} {U:>10} {p:>12} {r:>8}  {t['direction']:<20}  {sig}"
        )
    print()


# ---------------------------------------------------------------------------
# Section 5 — T2.3 figures
# ---------------------------------------------------------------------------


def _family_ordered_languages(
    languages: list[str],
) -> tuple[list[str], list[int]]:
    """Return languages sorted by family then alphabetically, plus boundary indices."""
    family_order = sorted(set(EXTENDED_FAMILIES.get(l, "ZZZ") for l in languages))
    grouped: dict[str, list[str]] = {f: [] for f in family_order}
    for lang in sorted(languages):
        fam = EXTENDED_FAMILIES.get(lang, "ZZZ")
        grouped[fam].append(lang)

    ordered: list[str] = []
    for fam in family_order:
        ordered.extend(sorted(grouped[fam]))

    boundaries: list[int] = []
    prev_fam = EXTENDED_FAMILIES.get(ordered[0], "ZZZ")
    for i, lang in enumerate(ordered[1:], 1):
        fam = EXTENDED_FAMILIES.get(lang, "ZZZ")
        if fam != prev_fam:
            boundaries.append(i)
        prev_fam = fam

    return ordered, boundaries


def plot_heatmap(
    model_key: str,
    pairwise_df: pd.DataFrame,
    global_score: float,
    languages: list[str],
) -> None:
    ordered, boundaries = _family_ordered_languages(languages)
    mat = clcs_matrix(pairwise_df, ordered)

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        mat,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        annot=False,
        square=True,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "CLCS", "shrink": 0.75},
    )

    # Family boundary lines
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=1.5)
        ax.axvline(b, color="black", linewidth=1.5)

    ax.set_title(
        f"{model_key}  —  Global CLCS: {global_score:.4f}",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Language", fontsize=11)
    ax.set_ylabel("Language", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    out = FIGURES_DIR / f"heatmap_{model_key}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(_REPO_ROOT)}")


def plot_family_bar_comparison(
    all_family_dfs: dict[str, pd.DataFrame],
) -> None:
    """One figure, one subplot per model, horizontal bar charts."""
    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 6), sharey=False)

    for ax, model_key in zip(axes, MODELS):
        df = all_family_dfs[model_key].sort_values("mean_clcs", ascending=True)
        color = MODEL_COLORS[model_key]
        bars = ax.barh(df["family"], df["mean_clcs"], color=color, alpha=0.85, edgecolor="white")
        ax.axvline(df["mean_clcs"].mean(), color="black", linewidth=1.2,
                   linestyle="--", label=f"mean={df['mean_clcs'].mean():.3f}")
        ax.set_xlim(0.0, 1.05)
        ax.set_title(model_key, fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean CLCS", fontsize=10)
        ax.legend(fontsize=8)
        # Value labels
        for bar, (_, row) in zip(bars, df.iterrows()):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row['mean_clcs']:.3f}",
                va="center",
                fontsize=8,
            )

    fig.suptitle("Family-level CLCS by Model", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "family_bar_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(_REPO_ROOT)}")


def plot_model_comparison(
    global_scores: dict[str, float],
    all_pairwise: dict[str, pd.DataFrame],
) -> None:
    means = [global_scores[m] for m in MODELS]
    stds = [all_pairwise[m]["clcs"].std() for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]

    x = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(max(7, len(MODELS) * 1.5), 5))
    bars = ax.bar(
        x, means, yerr=stds, capsize=7, width=0.5,
        color=colors, edgecolor="white", alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel("Global CLCS", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Global CLCS by Model (error bars = std of 210 pairs)", fontsize=12)
    ax.axhline(np.mean(means), color="black", linewidth=1, linestyle=":", alpha=0.6)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.02,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(_REPO_ROOT)}")


def plot_worst_pairs(all_worst_pairs: dict[str, pd.DataFrame]) -> None:
    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 6))

    for ax, model_key in zip(axes, MODELS):
        df = all_worst_pairs[model_key]
        labels = [f"{r.lang_a}–{r.lang_b}" for r in df.itertuples()]
        color = MODEL_COLORS[model_key]
        ax.barh(labels, df["clcs"].values, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(
            df["clcs"].mean(), color="black", linewidth=1.2, linestyle="--",
            label=f"mean={df['clcs'].mean():.3f}",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_title(model_key, fontsize=12, fontweight="bold")
        ax.set_xlabel("CLCS", fontsize=10)
        ax.legend(fontsize=8)
        # Value labels
        for i, v in enumerate(df["clcs"].values):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle("Bottom-10 Language Pairs by CLCS", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "worst_pairs.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Section 6 — main
# ---------------------------------------------------------------------------


def main() -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_pairwise: dict[str, pd.DataFrame] = {}
    all_family: dict[str, pd.DataFrame] = {}
    global_scores: dict[str, float] = {}
    all_worst_pairs: dict[str, pd.DataFrame] = {}

    # ---- T2.1 + T2.2 ----
    for model_key in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        preds_by_lang = load_predictions(model_key)
        aligned = align_predictions(preds_by_lang)
        languages = sorted(aligned.keys())
        n_samples = len(next(iter(aligned.values())))
        print(f"  Languages : {languages}")
        print(f"  Samples   : {n_samples} (aligned)")

        pairwise_df, fam_df, g, _ = compute_and_save_t21(model_key, aligned, languages)
        all_pairwise[model_key] = pairwise_df
        all_family[model_key] = fam_df
        global_scores[model_key] = g
        print(f"  Global CLCS: {g:.4f}")

        top5 = pairwise_df.nlargest(5, "clcs")[["lang_a", "lang_b", "clcs"]]
        bot5 = pairwise_df.nsmallest(5, "clcs")[["lang_a", "lang_b", "clcs"]]
        print("\n  Top-5 pairs:")
        print(top5.to_string(index=False))
        print("\n  Bottom-5 pairs:")
        print(bot5.to_string(index=False))

        t22 = save_t22_outputs(model_key, pairwise_df)
        all_worst_pairs[model_key] = t22["worst_pairs"]

        fam_cmp = t22["family_comparison"]
        within_row = fam_cmp[fam_cmp["family_type"] == "within_family"].iloc[0]
        cross_row = fam_cmp[fam_cmp["family_type"] == "cross_family"].iloc[0]
        print(f"\n  Within-family mean CLCS: {within_row['mean_clcs']:.4f}  (n={within_row['n_pairs']})")
        print(f"  Cross-family mean CLCS:  {cross_row['mean_clcs']:.4f}  (n={cross_row['n_pairs']})")

    # ---- T2.1 global JSON ----
    save_global_clcs_json(global_scores)

    print(f"\n{'='*60}")
    print("GLOBAL CLCS SUMMARY")
    print(f"{'='*60}")
    for m in sorted(global_scores, key=lambda x: global_scores[x], reverse=True):
        print(f"  {m:<14}: {global_scores[m]:.4f}")

    # ---- T2.3 Figures ----
    print(f"\n{'='*60}")
    print("Generating figures …")
    print(f"{'='*60}")

    for model_key in MODELS:
        languages = sorted(all_pairwise[model_key]["lang_a"].unique().tolist() +
                           all_pairwise[model_key]["lang_b"].unique().tolist())
        languages = sorted(set(languages))
        plot_heatmap(model_key, all_pairwise[model_key], global_scores[model_key], languages)

    plot_family_bar_comparison(all_family)
    plot_model_comparison(global_scores, all_pairwise)
    plot_worst_pairs(all_worst_pairs)

    # ---- T2.4 Statistical Tests ----
    print(f"\n{'='*60}")
    print("Statistical tests …")
    print(f"{'='*60}")

    model_tests = run_model_comparison_tests(all_pairwise)
    family_tests = run_within_cross_family_tests(all_pairwise)
    save_statistical_tests(model_tests, family_tests)
    print_stats_summary(model_tests, family_tests)

    print("Phase 2 analysis complete.")


if __name__ == "__main__":
    main()
