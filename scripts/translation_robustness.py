#!/usr/bin/env python3
"""
Translation robustness check for CLCS.

Validates that CLCS findings are not artifacts of OPUS-MT translation quality by
re-translating a stratified sample with NLLB-200 and measuring rank stability
(Spearman ρ) of pairwise CLCS scores across both translation systems.

Steps
-----
1. Stratified sample of 50 source sentences, weighted toward he/hi/bg hard cases
   → data/processed/validation_subset.csv   (OPUS-MT translations)
2. Re-translate with facebook/nllb-200-distilled-600M (CPU)
   → data/processed/validation_subset_nllb.csv
3. Run mBERT, XLM-R-base, mDeBERTa on both subsets
4. Compute pairwise CLCS per translation system (mean across 3 models)
5. Spearman ρ between OPUS-MT CLCS and NLLB CLCS across all language pairs
6. Print summary and save results/scores/translation_robustness.csv

Usage
-----
    python scripts/translation_robustness.py
    python scripts/translation_robustness.py --skip-translation   # reuse cached NLLB file
    python scripts/translation_robustness.py --nllb-batch-size 4  # slower CPUs

Runtime
-------
NLLB translation (~50 sentences × 27 languages, CPU): 20–60 min depending on hardware.
Inference on 50 sentences × 3 models: a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import LANGUAGE_FAMILIES
from src.metrics.clcs import compute_all_pairwise
from src.models.inference import MODEL_REGISTRY, SentimentInference

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_LANGS: list[str] = sorted([
    "ar", "bg", "bs", "cs", "da", "de", "el", "es", "fi", "fr",
    "he", "hi", "hr", "hu", "id", "it", "lv", "nl", "pl", "pt",
    "ro", "ru", "sl", "sr", "sv", "uk", "zh",
])

# NLLB-200 Flores-200 language codes for each ISO 639-1 code in SHARED_LANGS
NLLB_CODES: dict[str, str] = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "bs": "bos_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "es": "spa_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "lv": "lvs_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sl": "slv_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
}

ENCODER_MODELS: list[str] = ["mbert", "xlmr-base", "mdeberta"]
N_SAMPLE = 50
# Languages involved in the worst-performing pairs from prior results
WORST_LANGS: list[str] = ["he", "hi", "bg"]

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_PREDICTIONS_DIR = _REPO_ROOT / "results" / "predictions"
_SCORES_DIR = _REPO_ROOT / "results" / "scores"
_SUBSET_OPUS_PATH = _PROCESSED_DIR / "validation_subset.csv"
_SUBSET_NLLB_PATH = _PROCESSED_DIR / "validation_subset_nllb.csv"
_ROBUSTNESS_CSV = _SCORES_DIR / "translation_robustness.csv"


# ---------------------------------------------------------------------------
# Step 1 — stratified sampling
# ---------------------------------------------------------------------------


def _load_disagreement_scores(source_ids: list[int]) -> dict[int, float]:
    """Score each source_id by prediction disagreement rate across he/hi/bg.

    Loads existing transformer-model prediction checkpoints and computes, for
    each source_id, the fraction of (model, lang_a, lang_b) triples where
    predictions disagree across the three worst-performing languages.  Higher
    score → more inconsistent → higher sampling priority.

    Falls back to a uniform score of 0.0 for any missing checkpoint.
    """
    sid_set = set(source_ids)
    # {lang: {source_id: {model: label}}}
    preds: dict[str, dict[int, dict[str, int]]] = {lg: {} for lg in WORST_LANGS}

    for model_key in ENCODER_MODELS:
        for lang in WORST_LANGS:
            ckpt = _PREDICTIONS_DIR / f"{model_key}_{lang}.json"
            if not ckpt.exists():
                continue
            data = json.loads(ckpt.read_text(encoding="utf-8"))
            for sid, lbl in zip(data["source_ids"], data["labels"]):
                if sid in sid_set:
                    preds[lang].setdefault(sid, {})[model_key] = lbl

    scores: dict[int, float] = {}
    for sid in source_ids:
        disagreements = 0
        comparisons = 0
        for mk in ENCODER_MODELS:
            label_vec = [preds[lg].get(sid, {}).get(mk) for lg in WORST_LANGS]
            label_vec = [v for v in label_vec if v is not None]
            if len(label_vec) < 2:
                continue
            for i in range(len(label_vec)):
                for j in range(i + 1, len(label_vec)):
                    comparisons += 1
                    if label_vec[i] != label_vec[j]:
                        disagreements += 1
        scores[sid] = disagreements / comparisons if comparisons else 0.0

    return scores


def sample_validation_subset(corpus_path: Path, n: int = N_SAMPLE) -> pd.DataFrame:
    """Stratified sample of *n* source sentences from the parallel corpus.

    Strategy
    --------
    - Restrict to source_ids that have translations in all 27 SHARED_LANGS.
    - Within each gold_label stratum (neg/neu/pos), rank source_ids by
      prediction disagreement across he/hi/bg (descending) so that the
      hardest cases for those language pairs are over-represented.
    - Take ceil(n/3) from each stratum, then trim to exactly *n*.

    Parameters
    ----------
    corpus_path:
        Path to ``parallel_corpus.csv``.
    n:
        Total number of source sentences to sample.

    Returns
    -------
    pd.DataFrame
        Subset of *corpus_path* rows (all languages) for the sampled source_ids.
    """
    df = pd.read_csv(corpus_path)

    # English rows have NaN lang; target rows have a lang code.
    source_df = df[df["lang"].isna()][["source_id", "source_text", "gold_label"]].drop_duplicates("source_id")
    target_df = df[df["lang"].notna()]

    # Restrict to source_ids with all 27 target translations.
    lang_counts = target_df.groupby("source_id")["lang"].count()
    complete_ids = set(lang_counts[lang_counts >= len(SHARED_LANGS)].index)
    source_df = source_df[source_df["source_id"].isin(complete_ids)].copy()

    print(f"[sample] {len(source_df)} source_ids with all {len(SHARED_LANGS)} translations")

    # Compute disagreement scores for prioritisation.
    print("[sample] Computing he/hi/bg disagreement scores from existing predictions …")
    scores = _load_disagreement_scores(source_df["source_id"].tolist())
    source_df["_disagree"] = source_df["source_id"].map(scores).fillna(0.0)

    per_stratum = int(np.ceil(n / 3))
    sampled_ids: list[int] = []

    for label in [0, 1, 2]:
        stratum = source_df[source_df["gold_label"] == label].sort_values(
            "_disagree", ascending=False
        )
        chosen = stratum["source_id"].tolist()[:per_stratum]
        sampled_ids.extend(chosen)
        print(f"  label={label}: {len(chosen)} sentences sampled")

    # Trim to exactly n, preserving label balance as much as possible.
    sampled_ids = sampled_ids[:n]
    sampled_set = set(sampled_ids)

    # Build the subset CSV: target-language rows for sampled source_ids.
    subset = target_df[
        target_df["source_id"].isin(sampled_set) & target_df["lang"].isin(SHARED_LANGS)
    ].copy()

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    subset.to_csv(_SUBSET_OPUS_PATH, index=False)
    print(f"[sample] Saved {len(subset)} rows ({len(sampled_ids)} sources × {len(SHARED_LANGS)} langs)"
          f" → {_SUBSET_OPUS_PATH.relative_to(_REPO_ROOT)}")
    return subset


# ---------------------------------------------------------------------------
# Step 2 — NLLB-200 re-translation
# ---------------------------------------------------------------------------


def translate_with_nllb(
    source_sentences: list[str],
    source_ids: list[int],
    gold_labels: list[int],
    batch_size: int = 8,
) -> pd.DataFrame:
    """Translate *source_sentences* into all SHARED_LANGS with NLLB-200.

    Uses AutoModelForSeq2SeqLM + AutoTokenizer directly (transformers ≥ 5.x
    removed the "translation" pipeline task).

    Parameters
    ----------
    source_sentences:
        English source texts (one per unique source_id).
    source_ids:
        Corresponding source_id values.
    gold_labels:
        Corresponding gold_label values.
    batch_size:
        Number of sentences per NLLB forward pass (lower = less RAM).

    Returns
    -------
    pd.DataFrame
        Same schema as ``validation_subset.csv`` with ``translated_text``
        containing NLLB translations.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cpu")

    import warnings
    # Suppress the harmless "max_new_tokens overrides max_length" info message.
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")

    print(f"\n[nllb] Loading {NLLB_MODEL_ID} on CPU …")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
    model.to(device)
    model.eval()

    rows: list[dict] = []
    for lang in SHARED_LANGS:
        nllb_tgt = NLLB_CODES[lang]
        forced_bos_id = tokenizer.convert_tokens_to_ids(nllb_tgt)
        print(f"  → {lang} ({nllb_tgt}, bos={forced_bos_id}) …", end=" ", flush=True)

        tokenizer.src_lang = "eng_Latn"
        translated: list[str] = []

        with torch.no_grad():
            for start in range(0, len(source_sentences), batch_size):
                batch = source_sentences[start : start + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(device)
                output_ids = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_id,
                    max_new_tokens=256,
                )
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                translated.extend(decoded)

        print(f"done ({len(translated)} sentences)")

        for sid, src, lbl, trans in zip(source_ids, source_sentences, gold_labels, translated):
            rows.append({
                "source_id": sid,
                "source_text": src,
                "gold_label": lbl,
                "lang": lang,
                "translated_text": trans,
            })

    nllb_df = pd.DataFrame(rows)
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nllb_df.to_csv(_SUBSET_NLLB_PATH, index=False)
    print(f"[nllb] Saved {len(nllb_df)} rows → {_SUBSET_NLLB_PATH.relative_to(_REPO_ROOT)}")
    return nllb_df


# ---------------------------------------------------------------------------
# Step 3 — inference on a subset DataFrame
# ---------------------------------------------------------------------------


def run_inference_on_subset(
    subset_df: pd.DataFrame,
    label: str = "subset",
    batch_size: int = 32,
) -> dict[str, dict[str, list[int]]]:
    """Run all three encoder models on a translated subset.

    Parameters
    ----------
    subset_df:
        DataFrame with columns ``source_id``, ``lang``, ``translated_text``
        covering all SHARED_LANGS.
    label:
        Human-readable label for progress output (e.g. ``"OPUS-MT"``).
    batch_size:
        Inference batch size.

    Returns
    -------
    dict
        ``{model_key: {lang: [label_0, label_1, ...]}}`` — predictions
        aligned to the sorted source_id order within each language.
    """
    # Build {lang: (sorted_source_ids, texts)} for alignment
    texts_by_lang: dict[str, tuple[list[int], list[str]]] = {}
    for lang in SHARED_LANGS:
        lang_df = subset_df[subset_df["lang"] == lang].sort_values("source_id")
        texts_by_lang[lang] = (
            lang_df["source_id"].tolist(),
            lang_df["translated_text"].tolist(),
        )

    results: dict[str, dict[str, list[int]]] = {}

    for model_key in ENCODER_MODELS:
        print(f"\n[inference/{label}] {model_key}")
        engine = SentimentInference(model_key, batch_size=batch_size)
        results[model_key] = {}
        for lang in SHARED_LANGS:
            _, texts = texts_by_lang[lang]
            labels, _ = engine.predict(texts, show_progress=False)
            results[model_key][lang] = labels
            print(f"  {lang}: {len(labels)} predictions")

    return results


# ---------------------------------------------------------------------------
# Step 4 — CLCS per language pair, averaged across models
# ---------------------------------------------------------------------------


def compute_mean_pairwise_clcs(
    inference_results: dict[str, dict[str, list[int]]],
) -> pd.DataFrame:
    """Compute pairwise CLCS for each model and return the mean across models.

    Parameters
    ----------
    inference_results:
        Output of :func:`run_inference_on_subset`.

    Returns
    -------
    pd.DataFrame
        Columns ``["lang_a", "lang_b", "clcs"]`` with CLCS averaged across
        all models in *inference_results*.
    """
    dfs: list[pd.DataFrame] = []
    for model_key, preds_by_lang in inference_results.items():
        # Only include langs present in this model's predictions
        langs = [lg for lg in SHARED_LANGS if lg in preds_by_lang]
        pw = compute_all_pairwise(preds_by_lang, langs)
        pw = pw.rename(columns={"clcs": model_key})
        dfs.append(pw)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["lang_a", "lang_b"], how="inner")

    model_cols = ENCODER_MODELS
    merged["clcs"] = merged[model_cols].mean(axis=1)
    return merged[["lang_a", "lang_b", "clcs"]]


# ---------------------------------------------------------------------------
# Step 5 — Spearman correlation + results CSV
# ---------------------------------------------------------------------------


def compute_robustness(
    opus_clcs: pd.DataFrame,
    nllb_clcs: pd.DataFrame,
) -> pd.DataFrame:
    """Merge OPUS-MT and NLLB CLCS scores and compute Spearman ρ.

    Parameters
    ----------
    opus_clcs, nllb_clcs:
        Output of :func:`compute_mean_pairwise_clcs` for each translation system.

    Returns
    -------
    pd.DataFrame
        Columns ``["lang_pair", "clcs_opus", "clcs_nllb", "delta"]`` sorted
        by absolute delta descending.
    """
    opus = opus_clcs.rename(columns={"clcs": "clcs_opus"})
    nllb = nllb_clcs.rename(columns={"clcs": "clcs_nllb"})
    merged = opus.merge(nllb, on=["lang_a", "lang_b"])
    merged["lang_pair"] = merged["lang_a"] + "-" + merged["lang_b"]
    merged["delta"] = (merged["clcs_opus"] - merged["clcs_nllb"]).abs()

    rho, p_value = spearmanr(merged["clcs_opus"], merged["clcs_nllb"])

    print(f"\n{'='*60}")
    print("SPEARMAN CORRELATION (OPUS-MT vs NLLB CLCS across {n} language pairs)".format(
        n=len(merged)
    ))
    print(f"{'='*60}")
    print(f"  ρ = {rho:.4f}   p = {p_value:.4e}")
    if rho >= 0.9:
        verdict = "Very high rank stability — findings are robust to translation choice."
    elif rho >= 0.7:
        verdict = "High rank stability — minor sensitivity to translation system."
    elif rho >= 0.5:
        verdict = "Moderate rank stability — some sensitivity to translation system."
    else:
        verdict = "Low rank stability — CLCS rankings are sensitive to translation quality."
    print(f"  → {verdict}\n")

    result = merged[["lang_pair", "clcs_opus", "clcs_nllb", "delta"]].sort_values(
        "delta", ascending=False
    )
    return result, rho, p_value


# ---------------------------------------------------------------------------
# Step 6 — summary
# ---------------------------------------------------------------------------


def print_summary(
    result_df: pd.DataFrame,
    opus_clcs: pd.DataFrame,
    nllb_clcs: pd.DataFrame,
    rho: float,
    p_value: float,
) -> None:
    """Print divergent pairs and family-level ranking comparison."""
    print(f"{'='*60}")
    print("TOP-10 MOST DIVERGENT LANGUAGE PAIRS")
    print(f"{'='*60}")
    top10 = result_df.head(10)
    for _, row in top10.iterrows():
        print(f"  {row['lang_pair']:8s}  opus={row['clcs_opus']:.4f}  nllb={row['clcs_nllb']:.4f}"
              f"  Δ={row['delta']:.4f}")

    # Family-level ranking
    def _family_ranking(pairwise_df: pd.DataFrame, label: str) -> pd.Series:
        df = pairwise_df.copy()
        df["family_a"] = df["lang_a"].map(LANGUAGE_FAMILIES)
        df["family_b"] = df["lang_b"].map(LANGUAGE_FAMILIES)
        same = df[df["family_a"] == df["family_b"]].copy()
        fam_mean = same.groupby("family_a")["clcs"].mean().sort_values(ascending=False)
        print(f"\n  [{label}] Family ranking:")
        for fam, score in fam_mean.items():
            print(f"    {fam:15s} {score:.4f}")
        return fam_mean

    print(f"\n{'='*60}")
    print("FAMILY-LEVEL CLCS RANKING COMPARISON")
    print(f"{'='*60}")
    opus_fam = _family_ranking(opus_clcs, "OPUS-MT")
    nllb_fam  = _family_ranking(nllb_clcs,  "NLLB-200")

    # Spearman on family rankings
    common_fams = sorted(set(opus_fam.index) & set(nllb_fam.index))
    if len(common_fams) >= 3:
        fam_rho, fam_p = spearmanr(
            [opus_fam[f] for f in common_fams],
            [nllb_fam[f] for f in common_fams],
        )
        print(f"\n  Family-level Spearman ρ = {fam_rho:.4f}  (p = {fam_p:.4e})")
        preserved = "YES" if fam_rho >= 0.8 else "PARTIAL" if fam_rho >= 0.5 else "NO"
        print(f"  Family-level ranking preserved: {preserved}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Overall Spearman ρ (pair-level): {rho:.4f}  p={p_value:.4e}")
    worst_pair = result_df.iloc[0]
    print(f"  Most divergent pair : {worst_pair['lang_pair']}  "
          f"(Δ={worst_pair['delta']:.4f}, "
          f"opus={worst_pair['clcs_opus']:.4f}, nllb={worst_pair['clcs_nllb']:.4f})")
    mean_delta = result_df["delta"].mean()
    print(f"  Mean |Δ| across all pairs: {mean_delta:.4f}")

    # Flag whether he/hi/bg pairs are in the top divergers
    worst_involve = result_df[
        result_df["lang_pair"].str.contains("|".join(WORST_LANGS))
    ].head(10)
    if len(worst_involve):
        print(f"\n  Worst-performing langs (he/hi/bg) in top-10 divergers:")
        for _, row in worst_involve.iterrows():
            print(f"    {row['lang_pair']:8s}  Δ={row['delta']:.4f}")
    else:
        print("\n  he/hi/bg pairs are NOT in the top-10 divergers — "
              "translation system does not differentially affect these languages.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translation robustness check: OPUS-MT vs NLLB-200 CLCS rank stability."
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help=f"Skip NLLB translation and reuse existing {_SUBSET_NLLB_PATH.name}.",
    )
    parser.add_argument(
        "--skip-opus-subset",
        action="store_true",
        help=f"Skip sampling and reuse existing {_SUBSET_OPUS_PATH.name}.",
    )
    parser.add_argument(
        "--nllb-batch-size",
        type=int,
        default=8,
        dest="nllb_batch_size",
        help="Batch size for NLLB translation pipeline (default: 8; lower if OOM).",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=32,
        dest="inference_batch_size",
        help="Batch size for encoder model inference (default: 32).",
    )
    parser.add_argument(
        "--corpus",
        default=str(_REPO_ROOT / "data" / "processed" / "parallel_corpus.csv"),
        help="Path to parallel_corpus.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: OPUS-MT validation subset
    # ------------------------------------------------------------------
    if args.skip_opus_subset:
        if not _SUBSET_OPUS_PATH.exists():
            raise FileNotFoundError(
                f"--skip-opus-subset set but {_SUBSET_OPUS_PATH} does not exist."
            )
        print(f"[step 1] Reusing existing {_SUBSET_OPUS_PATH.name}")
        opus_subset = pd.read_csv(_SUBSET_OPUS_PATH)
    else:
        print("\n[step 1] Sampling validation subset from parallel corpus …")
        opus_subset = sample_validation_subset(Path(args.corpus))

    sampled_source_ids = sorted(opus_subset["source_id"].unique().tolist())
    print(f"  {len(sampled_source_ids)} unique source_ids in subset")

    # ------------------------------------------------------------------
    # Step 2: NLLB re-translation
    # ------------------------------------------------------------------
    if args.skip_translation:
        if not _SUBSET_NLLB_PATH.exists():
            raise FileNotFoundError(
                f"--skip-translation set but {_SUBSET_NLLB_PATH} does not exist."
            )
        print(f"\n[step 2] Reusing existing {_SUBSET_NLLB_PATH.name}")
        nllb_subset = pd.read_csv(_SUBSET_NLLB_PATH)
    else:
        print(f"\n[step 2] Translating {len(sampled_source_ids)} sentences with NLLB-200 …")
        print(f"  NOTE: this runs on CPU and may take 20–60 minutes.\n")

        # Load English source texts for the sampled source_ids.
        corpus_df = pd.read_csv(Path(args.corpus))
        src_rows = (
            corpus_df[corpus_df["lang"].isna()]
            [["source_id", "source_text", "gold_label"]]
            .drop_duplicates("source_id")
            .set_index("source_id")
            .loc[sampled_source_ids]
            .reset_index()
        )
        nllb_subset = translate_with_nllb(
            source_sentences=src_rows["source_text"].tolist(),
            source_ids=src_rows["source_id"].tolist(),
            gold_labels=src_rows["gold_label"].tolist(),
            batch_size=args.nllb_batch_size,
        )

    # ------------------------------------------------------------------
    # Step 3: Inference on both subsets
    # ------------------------------------------------------------------
    print(f"\n[step 3] Running inference on OPUS-MT subset …")
    opus_preds = run_inference_on_subset(
        opus_subset, label="OPUS-MT", batch_size=args.inference_batch_size
    )

    print(f"\n[step 3] Running inference on NLLB subset …")
    nllb_preds = run_inference_on_subset(
        nllb_subset, label="NLLB", batch_size=args.inference_batch_size
    )

    # ------------------------------------------------------------------
    # Step 4: Pairwise CLCS averaged across models
    # ------------------------------------------------------------------
    print("\n[step 4] Computing pairwise CLCS (mean across 3 models) …")
    opus_clcs = compute_mean_pairwise_clcs(opus_preds)
    nllb_clcs  = compute_mean_pairwise_clcs(nllb_preds)
    print(f"  {len(opus_clcs)} language pairs computed for each system")

    # ------------------------------------------------------------------
    # Step 5: Spearman correlation + save CSV
    # ------------------------------------------------------------------
    print("\n[step 5] Computing Spearman correlation …")
    result_df, rho, p_value = compute_robustness(opus_clcs, nllb_clcs)

    save_df = result_df[["lang_pair", "clcs_opus", "clcs_nllb"]].copy()
    save_df.to_csv(_ROBUSTNESS_CSV, index=False)
    print(f"  Saved → {_ROBUSTNESS_CSV.relative_to(_REPO_ROOT)}")

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    print_summary(result_df, opus_clcs, nllb_clcs, rho, p_value)


if __name__ == "__main__":
    main()
