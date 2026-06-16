#!/usr/bin/env python3
"""
Compute CLCS scores from prediction checkpoints.

Loads results/predictions/{model}_{lang}.json for all (model, lang) pairs,
aligns predictions by source_id, then computes:
  - pairwise CLCS for all language pairs
  - global CLCS per model
  - family CLCS per model

Default mode — saves original results to:
  results/scores/clcs_pairwise_{model}.csv
  results/scores/clcs_family_{model}.csv
  results/scores/clcs_global.csv
  results/scores/clcs_matrix_{model}.csv

Corrected mode (--corrected) — fair cross-model evaluation:
  Restricts to SHARED_LANGS (27 languages, English excluded because two LLMs
  were never run on English, making 28- vs 27-language comparisons invalid)
  and to the intersection of source_ids available for ALL models × languages
  (LLM API inference had partial failures; using different sample sets per
  model inflates/deflates scores inconsistently).
  Saves to results/scores/clcs_*_corrected_{model}.csv and
  results/scores/clcs_corrected.json without touching the originals.

Usage
-----
    python scripts/compute_clcs.py              # original behaviour
    python scripts/compute_clcs.py --corrected  # fair evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import LANGUAGE_FAMILIES
from src.metrics.clcs import (
    clcs_matrix,
    compute_all_pairwise,
    family_clcs,
    global_clcs,
    pairwise_kappa,
)

_PREDICTIONS_DIR = _REPO_ROOT / "results" / "predictions"
_SCORES_DIR = _REPO_ROOT / "results" / "scores"

# The 27 languages present in every model's prediction set (English excluded:
# encoder models ran on English but LLMs did not, making the sets incomparable).
SHARED_LANGS = sorted([
    "ar", "bg", "bs", "cs", "da", "de", "el", "es", "fi", "fr",
    "he", "hi", "hr", "hu", "id", "it", "lv", "nl", "pl", "pt",
    "ro", "ru", "sl", "sr", "sv", "uk", "zh",
])


def load_predictions(
    model_key: str,
    restrict_langs: list[str] | None = None,
) -> dict[str, dict[int, int]]:
    """Load prediction files for a model. Returns {lang: {source_id: label}}.

    Parameters
    ----------
    model_key:
        Model identifier (prefix of prediction filenames).
    restrict_langs:
        If given, only load these language codes.
    """
    result = {}
    for path in sorted(_PREDICTIONS_DIR.glob(f"{model_key}_*.json")):
        lang = path.stem[len(model_key) + 1:]
        if restrict_langs is not None and lang not in restrict_langs:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        result[lang] = dict(zip(data["source_ids"], data["labels"]))
    return result


def align_predictions(
    preds_by_lang: dict[str, dict[int, int]],
    restrict_ids: list[int] | None = None,
) -> dict[str, list[int]]:
    """Align all languages to a common source_id list.

    Parameters
    ----------
    preds_by_lang:
        Output of :func:`load_predictions`.
    restrict_ids:
        If given, use exactly these source_ids (must be present in every
        language).  If ``None``, use the per-model intersection.
    """
    if restrict_ids is not None:
        common_ids = restrict_ids
    else:
        common_ids = sorted(
            set.intersection(*[set(ids) for ids in preds_by_lang.values()])
        )
    return {
        lang: [id_map[sid] for sid in common_ids]
        for lang, id_map in preds_by_lang.items()
    }


def find_common_ids(model_keys: list[str], langs: list[str]) -> list[int]:
    """Return source_ids that have complete predictions for all models × langs."""
    per_model: list[set[int]] = []
    for m in model_keys:
        sets = [
            set(json.loads((_PREDICTIONS_DIR / f"{m}_{lang}.json").read_text())["source_ids"])
            for lang in langs
            if (_PREDICTIONS_DIR / f"{m}_{lang}.json").exists()
        ]
        if sets:
            per_model.append(set.intersection(*sets))
    return sorted(set.intersection(*per_model))


def run(corrected: bool = False) -> None:
    """Compute and save CLCS scores for all models found in the predictions directory.

    Iterates over every (model, language) prediction file, aligns predictions
    to a common source-ID list, then writes pairwise, matrix, family, and
    global CLCS CSVs.  In corrected mode the evaluation is restricted to
    ``SHARED_LANGS`` and the cross-model source-ID intersection so that all
    models are compared on identical inputs.

    Parameters
    ----------
    corrected:
        When ``True``, applies the fair cross-model evaluation protocol
        (27 shared languages, common source IDs, saves ``*_corrected*`` files
        and ``clcs_corrected.json`` / ``kappa_corrected.json``).
    """
    _SCORES_DIR.mkdir(parents=True, exist_ok=True)

    model_keys = sorted({p.stem.rsplit("_", 1)[0] for p in _PREDICTIONS_DIR.glob("*.json")})
    print(f"Models found: {model_keys}")

    suffix = "_corrected" if corrected else ""

    # In corrected mode, pre-compute the shared source_id set once across all models.
    common_ids: list[int] | None = None
    if corrected:
        print(f"\n[corrected] Restricting to {len(SHARED_LANGS)} shared languages: {SHARED_LANGS}")
        common_ids = find_common_ids(model_keys, SHARED_LANGS)
        print(f"[corrected] Common source_ids across all models: {len(common_ids)}")
        print(f"[corrected] Evaluation: {len(common_ids)} samples × {len(SHARED_LANGS)} languages "
              f"= {len(common_ids) * len(SHARED_LANGS):,} predictions per model")
        print(f"[corrected] Language pairs per model: {len(SHARED_LANGS)*(len(SHARED_LANGS)-1)//2}")

    global_rows = []

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        preds_by_lang = load_predictions(
            model_key,
            restrict_langs=SHARED_LANGS if corrected else None,
        )
        aligned = align_predictions(preds_by_lang, restrict_ids=common_ids)
        languages = sorted(aligned.keys())
        n_samples = len(next(iter(aligned.values())))
        print(f"  Languages : {len(languages)}  {languages}")
        print(f"  Samples   : {n_samples} (aligned)")

        # Pairwise CLCS.
        pairwise_df = compute_all_pairwise(aligned, languages)
        pairwise_path = _SCORES_DIR / f"clcs_pairwise{suffix}_{model_key}.csv"
        pairwise_df.to_csv(pairwise_path, index=False)
        print(f"  Saved pairwise → {pairwise_path.relative_to(_REPO_ROOT)}")

        # Matrix.
        mat = clcs_matrix(pairwise_df, languages)
        mat.to_csv(_SCORES_DIR / f"clcs_matrix{suffix}_{model_key}.csv")

        # Family CLCS.
        fam_df = family_clcs(pairwise_df, LANGUAGE_FAMILIES)
        fam_path = _SCORES_DIR / f"clcs_family{suffix}_{model_key}.csv"
        fam_df.to_csv(fam_path, index=False)
        print(f"  Saved family  → {fam_path.relative_to(_REPO_ROOT)}")

        # Mean Cohen's kappa across all language pairs.
        kappa_vals = []
        for lang_a in languages:
            for lang_b in languages:
                if lang_a >= lang_b:
                    continue
                k = pairwise_kappa(aligned, lang_a, lang_b)
                kappa_vals.append(k)
        mean_kappa = sum(kappa_vals) / len(kappa_vals) if kappa_vals else float("nan")
        print(f"  Mean κ (kappa): {mean_kappa:.4f}")

        # Global CLCS.
        g = global_clcs(pairwise_df)
        global_rows.append({"model": model_key, "global_clcs": g, "_kappa_vals": kappa_vals})
        print(f"  Global CLCS   : {g:.4f}")

        # Top/bottom 5 pairs.
        top5 = pairwise_df.nlargest(5, "clcs")[["lang_a", "lang_b", "clcs"]]
        bot5 = pairwise_df.nsmallest(5, "clcs")[["lang_a", "lang_b", "clcs"]]
        print("\n  Top-5 pairs:")
        print(top5.to_string(index=False))
        print("\n  Bottom-5 pairs:")
        print(bot5.to_string(index=False))

    # Global summary — compute mean kappa per model then drop the raw lists.
    for row in global_rows:
        kv = row.pop("_kappa_vals", [])
        row["mean_kappa"] = sum(kv) / len(kv) if kv else float("nan")

    global_df = pd.DataFrame(global_rows).sort_values("global_clcs", ascending=False)
    global_csv_path = _SCORES_DIR / f"clcs_global{suffix}.csv"
    global_df.to_csv(global_csv_path, index=False)

    if corrected:
        global_json_path = _SCORES_DIR / "clcs_corrected.json"
        kappa_json_path  = _SCORES_DIR / "kappa_corrected.json"
        global_json_path.write_text(
            json.dumps(
                {row["model"]: row["global_clcs"] for _, row in global_df.iterrows()},
                indent=2,
            ),
            encoding="utf-8",
        )
        kappa_json_path.write_text(
            json.dumps(
                {row["model"]: round(row["mean_kappa"], 6) for _, row in global_df.iterrows()},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved corrected JSON → {global_json_path.relative_to(_REPO_ROOT)}")
        print(f"Saved kappa JSON     → {kappa_json_path.relative_to(_REPO_ROOT)}")

    print(f"\n{'='*60}")
    print("GLOBAL CLCS SUMMARY")
    print(f"{'='*60}")
    print(global_df.to_string(index=False))
    print(f"\nSaved → {global_csv_path.relative_to(_REPO_ROOT)}")


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute CLCS scores from prediction files.")
    parser.add_argument(
        "--corrected",
        action="store_true",
        help=(
            "Fair cross-model evaluation: restrict to 27 shared languages "
            "(no English) and the intersection of source_ids across all models."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(corrected=args.corrected)
