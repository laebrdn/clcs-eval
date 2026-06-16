#!/usr/bin/env python3
"""
Phase 3 analysis: CLCS evaluation of the consistency fine-tuned XLM-R model.

Steps
-----
1. Run inference on the full parallel corpus using models/xlmr-base_consistency/
2. Save predictions to results/predictions/xlmr-base-consistency_{lang}.json
3. Compute pairwise, family-level, and global CLCS
4. Generate 4 figure types (heatmap, family bar, model comparison, worst pairs)
   with 'consistency' in filename
5. Statistical tests: Wilcoxon comparing xlmr-base-consistency vs every other model
6. Print before/after comparison table by language family
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
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, mannwhitneyu

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

CONSISTENCY_KEY = "xlmr-base-consistency"
CONSISTENCY_CKPT = _REPO_ROOT / "models" / "xlmr-base_consistency"
PARALLEL_CHECKPOINT = _REPO_ROOT / "data" / "processed" / "parallel_checkpoint.json"

PREDICTIONS_DIR = _REPO_ROOT / "results" / "predictions"
SCORES_DIR = _REPO_ROOT / "results" / "scores"
FIGURES_DIR = _REPO_ROOT / "results" / "figures"

ALL_MODELS = ["mbert", "xlmr-base", "mdeberta", CONSISTENCY_KEY]

EXTENDED_FAMILIES: dict[str, str] = {
    **LANGUAGE_FAMILIES,
    "el": "Hellenic",
    "id": "Austronesian",
}

MODEL_COLORS = {
    "mbert": "#4C72B0",
    "xlmr-base": "#55A868",
    "mdeberta": "#DD8452",
    CONSISTENCY_KEY: "#C44E52",
}

_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Section 1 — Inference
# ---------------------------------------------------------------------------


def run_consistency_inference(
    parallel_checkpoint: Path,
    checkpoint_dir: Path,
    max_length: int = 128,
    batch_size: int = 32,
) -> dict[str, dict[int, int]]:
    """Run inference on all languages in the parallel corpus.

    Returns ``{lang: {source_id: label}}``.
    Saves each language's predictions to results/predictions/.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[inference] Loading checkpoint: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(_DEVICE)
    model.eval()
    print(f"[inference] Device: {_DEVICE}")

    print(f"[inference] Loading parallel corpus: {parallel_checkpoint}")
    corpus = json.loads(parallel_checkpoint.read_text(encoding="utf-8"))

    result: dict[str, dict[int, int]] = {}

    for lang, rows in sorted(corpus.items()):
        pred_path = PREDICTIONS_DIR / f"{CONSISTENCY_KEY}_{lang}.json"
        if pred_path.exists():
            print(f"  [skip] {lang} — predictions already exist")
            data = json.loads(pred_path.read_text(encoding="utf-8"))
            result[lang] = dict(zip(data["source_ids"], data["labels"]))
            continue

        texts = [r["translated_text"] for r in rows]
        source_ids = [r["source_id"] for r in rows]

        all_labels: list[int] = []
        all_probs: list[list[float]] = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(_DEVICE) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                labels = probs.argmax(axis=-1).tolist()
                all_labels.extend(labels)
                all_probs.extend(probs.tolist())

        payload = {
            "model": CONSISTENCY_KEY,
            "language": lang,
            "source_ids": source_ids,
            "labels": all_labels,
            "probs": all_probs,
        }
        pred_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        label_counts = {0: 0, 1: 0, 2: 0}
        for l in all_labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        print(f"  [{lang}] {len(texts)} texts  labels: {label_counts}")
        result[lang] = dict(zip(source_ids, all_labels))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Section 2 — Load existing model predictions
# ---------------------------------------------------------------------------


def load_predictions(model_key: str) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for path in sorted(PREDICTIONS_DIR.glob(f"{model_key}_*.json")):
        lang = path.stem[len(model_key) + 1:]
        data = json.loads(path.read_text(encoding="utf-8"))
        result[lang] = dict(zip(data["source_ids"], data["labels"]))
    return result


def align_predictions(preds_by_lang: dict[str, dict[int, int]]) -> dict[str, list[int]]:
    common_ids = sorted(
        set.intersection(*[set(ids) for ids in preds_by_lang.values()])
    )
    return {
        lang: [id_map[sid] for sid in common_ids]
        for lang, id_map in preds_by_lang.items()
    }


# ---------------------------------------------------------------------------
# Section 3 — CLCS computation
# ---------------------------------------------------------------------------


def compute_and_save(
    model_key: str,
    aligned: dict[str, list[int]],
    languages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    pairwise_df = compute_all_pairwise(aligned, languages)
    pairwise_df.to_csv(SCORES_DIR / f"{model_key}_pairwise_clcs.csv", index=False)

    fam_df = family_clcs(pairwise_df, EXTENDED_FAMILIES)
    fam_df.to_csv(SCORES_DIR / f"{model_key}_family_clcs.csv", index=False)

    g = global_clcs(pairwise_df)

    mat_df = clcs_matrix(pairwise_df, languages)
    mat_df.to_csv(SCORES_DIR / f"{model_key}_clcs_matrix.csv")

    # worst pairs / family comparison / worst languages
    worst_pairs = pairwise_df.nsmallest(10, "clcs")[["lang_a", "lang_b", "clcs"]].reset_index(drop=True)
    worst_pairs.to_csv(SCORES_DIR / f"{model_key}_worst_pairs.csv", index=False)

    df2 = pairwise_df.copy()
    df2["family_a"] = df2["lang_a"].map(EXTENDED_FAMILIES)
    df2["family_b"] = df2["lang_b"].map(EXTENDED_FAMILIES)
    within = df2[df2["family_a"] == df2["family_b"]]["clcs"]
    cross  = df2[df2["family_a"] != df2["family_b"]]["clcs"]
    fam_cmp = pd.DataFrame([
        {"family_type": "within_family", "mean_clcs": within.mean(), "n_pairs": len(within)},
        {"family_type": "cross_family",  "mean_clcs": cross.mean(),  "n_pairs": len(cross)},
    ])
    fam_cmp.to_csv(SCORES_DIR / f"{model_key}_family_comparison.csv", index=False)

    a = pairwise_df[["lang_a", "clcs"]].rename(columns={"lang_a": "lang"})
    b = pairwise_df[["lang_b", "clcs"]].rename(columns={"lang_b": "lang"})
    worst_langs = (
        pd.concat([a, b], ignore_index=True)
        .groupby("lang")["clcs"]
        .agg(mean_clcs="mean", n_pairs="count")
        .reset_index()
        .sort_values("mean_clcs")
        .reset_index(drop=True)
    )
    worst_langs.to_csv(SCORES_DIR / f"{model_key}_worst_languages.csv", index=False)

    print(f"  Global CLCS: {g:.4f}")
    print(f"  Most inconsistent: {worst_langs.iloc[0]['lang']} ({worst_langs.iloc[0]['mean_clcs']:.4f})")
    print(f"  Within-family: {within.mean():.4f}  Cross-family: {cross.mean():.4f}")

    return pairwise_df, fam_df, g, mat_df


# ---------------------------------------------------------------------------
# Section 4 — Figures (consistency model only)
# ---------------------------------------------------------------------------


def _family_ordered_languages(languages: list[str]) -> tuple[list[str], list[int]]:
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


def plot_heatmap_consistency(
    pairwise_df: pd.DataFrame,
    global_score: float,
    languages: list[str],
) -> None:
    ordered, boundaries = _family_ordered_languages(languages)
    mat = clcs_matrix(pairwise_df, ordered)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        mat, ax=ax, vmin=0.0, vmax=1.0, cmap="RdYlGn",
        annot=False, square=True, linewidths=0.2, linecolor="white",
        cbar_kws={"label": "CLCS", "shrink": 0.75},
    )
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=1.5)
        ax.axvline(b, color="black", linewidth=1.5)
    ax.set_title(f"{CONSISTENCY_KEY}  —  Global CLCS: {global_score:.4f}", fontsize=14, pad=14)
    ax.set_xlabel("Language", fontsize=11)
    ax.set_ylabel("Language", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    out = FIGURES_DIR / f"heatmap_{CONSISTENCY_KEY}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(_REPO_ROOT)}")


def plot_family_bar_consistency(fam_df: pd.DataFrame) -> None:
    df = fam_df.sort_values("mean_clcs", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    color = MODEL_COLORS[CONSISTENCY_KEY]
    bars = ax.barh(df["family"], df["mean_clcs"], color=color, alpha=0.85, edgecolor="white")
    ax.axvline(df["mean_clcs"].mean(), color="black", linewidth=1.2,
               linestyle="--", label=f"mean={df['mean_clcs'].mean():.3f}")
    ax.set_xlim(0.0, 1.05)
    ax.set_title(f"{CONSISTENCY_KEY} — Family CLCS", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean CLCS", fontsize=10)
    ax.legend(fontsize=9)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row['mean_clcs']:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / f"family_bar_{CONSISTENCY_KEY}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(_REPO_ROOT)}")


def plot_model_comparison_with_consistency(
    all_pairwise: dict[str, pd.DataFrame],
    global_scores: dict[str, float],
) -> None:
    """4-bar comparison: mbert / xlmr-base / mdeberta / xlmr-base-consistency."""
    models = ["mbert", "xlmr-base", "mdeberta", CONSISTENCY_KEY]
    means  = [global_scores[m] for m in models]
    stds   = [all_pairwise[m]["clcs"].std() for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=7, width=0.5,
                  color=colors, edgecolor="white", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Global CLCS", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Global CLCS — all models incl. consistency (±std of 210 pairs)", fontsize=11)
    ax.axhline(np.mean(means), color="black", linewidth=1, linestyle=":", alpha=0.6)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02,
                f"{mean:.4f}\n±{std:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "model_comparison_consistency.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(_REPO_ROOT)}")


def plot_worst_pairs_consistency(worst_pairs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [f"{r.lang_a}–{r.lang_b}" for r in worst_pairs.itertuples()]
    color = MODEL_COLORS[CONSISTENCY_KEY]
    ax.barh(labels, worst_pairs["clcs"].values, color=color, alpha=0.85, edgecolor="white")
    ax.axvline(worst_pairs["clcs"].mean(), color="black", linewidth=1.2, linestyle="--",
               label=f"mean={worst_pairs['clcs'].mean():.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"{CONSISTENCY_KEY} — Bottom-10 Language Pairs", fontsize=12, fontweight="bold")
    ax.set_xlabel("CLCS", fontsize=10)
    ax.legend(fontsize=9)
    for i, v in enumerate(worst_pairs["clcs"].values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / f"worst_pairs_{CONSISTENCY_KEY}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Section 5 — Statistical tests
# ---------------------------------------------------------------------------


def _rank_biserial_sr(W: float, n_nonzero: int) -> float:
    return 0.0 if n_nonzero == 0 else 1.0 - (2.0 * W) / (n_nonzero * (n_nonzero + 1))


def run_consistency_tests(all_pairwise: dict[str, pd.DataFrame]) -> list[dict]:
    """Wilcoxon: xlmr-base-consistency vs every other model."""
    results = []
    con_clcs = all_pairwise[CONSISTENCY_KEY]["clcs"].values
    for m2 in ["mbert", "xlmr-base", "mdeberta"]:
        other_clcs = all_pairwise[m2]["clcs"].values
        diffs = con_clcs - other_clcs
        n_nonzero = int((diffs != 0).sum())
        if n_nonzero < 10:
            stat, p, r = float("nan"), float("nan"), float("nan")
        else:
            res = wilcoxon(con_clcs, other_clcs)
            stat = float(res.statistic)
            p = float(res.pvalue)
            r = _rank_biserial_sr(stat, n_nonzero)
        results.append({
            "test": "wilcoxon_signed_rank",
            "model_a": CONSISTENCY_KEY,
            "model_b": m2,
            "W_statistic": stat,
            "pvalue": p,
            "n_nonzero_diffs": n_nonzero,
            "effect_r": r,
            "mean_a": float(con_clcs.mean()),
            "mean_b": float(other_clcs.mean()),
            "significant_0.05": bool(p < 0.05) if not np.isnan(p) else None,
        })
    return results


def update_statistical_tests_json(new_tests: list[dict]) -> None:
    path = SCORES_DIR / "statistical_tests.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"model_comparisons": [], "within_vs_cross_family": []}

    # Remove stale consistency entries then add fresh ones
    payload["model_comparisons"] = [
        t for t in payload["model_comparisons"]
        if CONSISTENCY_KEY not in (t.get("model_a", ""), t.get("model_b", ""))
    ]
    payload["model_comparisons"].extend(new_tests)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved → {path.relative_to(_REPO_ROOT)}")


def print_test_results(tests: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("STATISTICAL TESTS — xlmr-base-consistency vs other models")
    print("(Wilcoxon signed-rank, paired 210 pairs)")
    print("=" * 72)
    hdr = f"{'Comparison':<40} {'W':>10} {'p-value':>12} {'r':>8}  sig?"
    print(hdr)
    print("-" * 72)
    for t in tests:
        cmp = f"{t['model_a']} vs {t['model_b']}"
        W   = f"{t['W_statistic']:.1f}" if not np.isnan(t["W_statistic"]) else "  n/a"
        p   = f"{t['pvalue']:.4e}"      if not np.isnan(t["pvalue"])      else "     n/a"
        r   = f"{t['effect_r']:.4f}"    if not np.isnan(t["effect_r"])    else "   n/a"
        sig = "yes" if t.get("significant_0.05") else "no"
        direction = "↑" if t["mean_a"] > t["mean_b"] else "↓"
        print(f"{cmp:<40} {W:>10} {p:>12} {r:>8}  {sig}  {direction}{abs(t['mean_a']-t['mean_b']):.4f}")


# ---------------------------------------------------------------------------
# Section 6 — Before/after family comparison table
# ---------------------------------------------------------------------------


def print_family_comparison_table(
    pairwise_sentiment: pd.DataFrame,
    pairwise_consistency: pd.DataFrame,
) -> None:
    def family_means(pairwise_df: pd.DataFrame) -> pd.Series:
        df = pairwise_df.copy()
        df["family_a"] = df["lang_a"].map(EXTENDED_FAMILIES)
        df["family_b"] = df["lang_b"].map(EXTENDED_FAMILIES)
        # Use pairs where both languages are in the same family
        within = df[df["family_a"] == df["family_b"]].copy()
        # mean CLCS per family
        return (
            within.groupby("family_a")["clcs"]
            .mean()
            .rename("mean_clcs")
        )

    sent_means = family_means(pairwise_sentiment)
    cons_means = family_means(pairwise_consistency)

    families = sorted(set(sent_means.index) | set(cons_means.index))

    print("\n" + "=" * 68)
    print("BEFORE / AFTER — Family-level CLCS (within-family pairs)")
    print(f"{'Family':<18} {'xlmr-base':>12} {'consistency':>13} {'Δ':>8}")
    print("-" * 68)
    for fam in families:
        s = sent_means.get(fam, float("nan"))
        c = cons_means.get(fam, float("nan"))
        delta = c - s if not (np.isnan(s) or np.isnan(c)) else float("nan")
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "   n/a"
        s_str = f"{s:.4f}" if not np.isnan(s) else "   n/a"
        c_str = f"{c:.4f}" if not np.isnan(c) else "   n/a"
        print(f"{fam:<18} {s_str:>12} {c_str:>13} {delta_str:>8}")

    # Overall summary row
    s_global = pairwise_sentiment["clcs"].mean()
    c_global = pairwise_consistency["clcs"].mean()
    print("-" * 68)
    print(f"{'GLOBAL (all pairs)':<18} {s_global:>12.4f} {c_global:>13.4f} {c_global-s_global:>+8.4f}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not CONSISTENCY_CKPT.exists():
        print(f"[ERROR] Consistency checkpoint not found: {CONSISTENCY_CKPT}")
        sys.exit(1)

    # ---- 1. Inference ----
    print(f"\n{'='*60}")
    print("Step 1: Inference (xlmr-base-consistency)")
    print(f"{'='*60}")
    run_consistency_inference(PARALLEL_CHECKPOINT, CONSISTENCY_CKPT)

    # ---- 2. CLCS metrics ----
    print(f"\n{'='*60}")
    print("Step 2: CLCS computation")
    print(f"{'='*60}")

    preds = load_predictions(CONSISTENCY_KEY)
    aligned = align_predictions(preds)
    languages = sorted(aligned.keys())
    n_samples = len(next(iter(aligned.values())))
    print(f"  Languages : {languages}")
    print(f"  Samples   : {n_samples} (aligned)")

    pairwise_df, fam_df, g, _ = compute_and_save(CONSISTENCY_KEY, aligned, languages)

    # Update global_clcs.json
    global_path = SCORES_DIR / "global_clcs.json"
    global_scores_all: dict[str, float] = {}
    if global_path.exists():
        global_scores_all = json.loads(global_path.read_text(encoding="utf-8"))
    global_scores_all[CONSISTENCY_KEY] = g
    global_path.write_text(json.dumps(global_scores_all, indent=2), encoding="utf-8")
    print(f"  Updated → {global_path.relative_to(_REPO_ROOT)}")

    print(f"\nGLOBAL CLCS SUMMARY")
    for m in sorted(global_scores_all, key=lambda x: global_scores_all[x], reverse=True):
        marker = " ← consistency" if m == CONSISTENCY_KEY else ""
        print(f"  {m:<28}: {global_scores_all[m]:.4f}{marker}")

    # ---- 3. Figures ----
    print(f"\n{'='*60}")
    print("Step 3: Figures")
    print(f"{'='*60}")

    plot_heatmap_consistency(pairwise_df, g, languages)
    plot_family_bar_consistency(fam_df)

    # Load other models' pairwise for joint comparison plot
    all_pairwise: dict[str, pd.DataFrame] = {}
    for m in ["mbert", "xlmr-base", "mdeberta"]:
        p = SCORES_DIR / f"{m}_pairwise_clcs.csv"
        if p.exists():
            all_pairwise[m] = pd.read_csv(p)
        else:
            print(f"  [warn] Missing {p.name} — skipping in comparison plot")
    all_pairwise[CONSISTENCY_KEY] = pairwise_df

    worst_pairs = pd.read_csv(SCORES_DIR / f"{CONSISTENCY_KEY}_worst_pairs.csv")
    plot_worst_pairs_consistency(worst_pairs)

    if len(all_pairwise) == 4:
        plot_model_comparison_with_consistency(all_pairwise, global_scores_all)
    else:
        print("  [skip] model_comparison_consistency.png — missing some model pairwise CSVs")

    # ---- 4. Statistical tests ----
    print(f"\n{'='*60}")
    print("Step 4: Statistical tests")
    print(f"{'='*60}")

    if len(all_pairwise) >= 2:
        consistency_tests = run_consistency_tests(all_pairwise)
        update_statistical_tests_json(consistency_tests)
        print_test_results(consistency_tests)

    # ---- 5. Before/after family table ----
    print(f"\n{'='*60}")
    print("Step 5: Before/after family comparison")
    print(f"{'='*60}")

    sent_path = SCORES_DIR / "xlmr-base_pairwise_clcs.csv"
    if sent_path.exists():
        pairwise_sentiment = pd.read_csv(sent_path)
        print_family_comparison_table(pairwise_sentiment, pairwise_df)
    else:
        print("  [skip] xlmr-base_pairwise_clcs.csv not found")

    print("\nPhase 3 analysis complete.")


if __name__ == "__main__":
    main()
