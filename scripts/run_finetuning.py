#!/usr/bin/env python3
"""
CLI script to fine-tune sentiment classifiers on Brand24/mms,
with an optional consistency-augmented training mode.

Standard mode (default):
    Loads the mms train split, optionally subsamples to --max-train-rows rows
    (stratified by language × label), then calls fine_tune_sentiment().
    Fine-tuned checkpoints are saved to models/{model_key}_sentiment/.

Consistency mode (--consistency):
    Loads the pre-built parallel corpus checkpoint, then calls
    fine_tune_consistency() with a CE + symmetric-KL objective.
    Checkpoints are saved to models/{model_key}_consistency/.

Usage
-----
    # Standard: fine-tune xlmr-base on 50k mms rows
    python scripts/run_finetuning.py --model xlmr-base --max-train-rows 50000

    # Consistency smoke test: 2 000 pairs
    python scripts/run_finetuning.py --model xlmr-base --consistency --max-train-rows 2000

    # Consistency full run
    python scripts/run_finetuning.py --model xlmr-base --consistency --max-train-rows 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import load_mms
from src.models.inference import MODEL_REGISTRY
from src.training.finetune import fine_tune_consistency, fine_tune_sentiment

_MODELS_DIR = _REPO_ROOT / "models"
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "raw"
_DEFAULT_CHECKPOINT = _REPO_ROOT / "data" / "processed" / "parallel_checkpoint.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_parallel_df(checkpoint_path: str | Path) -> pd.DataFrame:
    """Load the parallel corpus checkpoint and flatten it to a tidy DataFrame."""
    from src.data.parallel import load_or_resume_corpus

    checkpoint = load_or_resume_corpus(checkpoint_path)
    if not checkpoint:
        print(
            f"[ERROR] Parallel corpus checkpoint not found at {checkpoint_path}.\n"
            "Run translation first (e.g. python scripts/run_translation.py)."
        )
        sys.exit(1)

    rows = [row for lang_rows in checkpoint.values() for row in lang_rows]
    df = pd.DataFrame(rows, columns=["source_id", "source_text", "gold_label", "lang", "translated_text"])
    print(f"[parallel] loaded {len(df):,} rows  ({df['source_id'].nunique()} source IDs × {df['lang'].nunique()} languages)")
    return df


# ---------------------------------------------------------------------------
# Main run logic
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Dispatch to standard or consistency-augmented fine-tuning.

    In standard mode, loads Brand24/mms and calls :func:`fine_tune_sentiment`
    for each requested model.  In consistency mode, loads the pre-built
    parallel corpus checkpoint and calls :func:`fine_tune_consistency`.
    Existing checkpoints are skipped unless ``--overwrite`` is set.

    Parameters
    ----------
    args:
        Parsed CLI arguments (see :func:`_parse_args`).
    """
    model_keys = [args.model] if args.model else list(MODEL_REGISTRY.keys())
    max_rows: int | None = args.max_train_rows if args.max_train_rows > 0 else None

    if args.consistency:
        # ----------------------------------------------------------------
        # Consistency-augmented fine-tuning path
        # ----------------------------------------------------------------
        parallel_df = _load_parallel_df(args.checkpoint_path)

        for model_key in model_keys:
            if model_key not in MODEL_REGISTRY:
                print(f"[WARN] Unknown model key '{model_key}', skipping.")
                continue

            output_dir = _MODELS_DIR / f"{model_key}_consistency"
            if output_dir.exists() and not args.overwrite:
                print(
                    f"\n[skip] {model_key} — consistency checkpoint already exists at "
                    f"{output_dir}. Use --overwrite to re-train."
                )
                continue

            print(f"\n{'='*60}")
            model_cfg = MODEL_REGISTRY[model_key]
            model_display = model_cfg.get("model_id", model_key) if isinstance(model_cfg, dict) else model_cfg
            print(f"Consistency fine-tuning: {model_key}  ({model_display})")
            print(f"{'='*60}")

            # Auto-detect sentiment checkpoint for warm start.
            sentiment_ckpt = _MODELS_DIR / f"{model_key}_sentiment"
            warm_start = sentiment_ckpt if (sentiment_ckpt / "config.json").exists() else None
            if warm_start:
                print(f"[warm start] using sentiment checkpoint: {warm_start}")
            else:
                print("[warm start] no sentiment checkpoint found — starting from base weights")

            fine_tune_consistency(
                model_key=model_key,
                parallel_df=parallel_df,
                output_dir=output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                val_frac=args.val_frac,
                max_train_rows=max_rows,
                consistency_weight=args.consistency_weight,
                min_ce_weight=args.min_ce_weight,
                warm_start_from=warm_start,
                seed=args.seed,
                cache_dir=args.cache_dir,
            )

    else:
        # ----------------------------------------------------------------
        # Standard sentiment fine-tuning path
        # ----------------------------------------------------------------
        print("Loading Brand24/mms …")
        dataset = load_mms(cache_dir=args.cache_dir)

        for model_key in model_keys:
            if model_key not in MODEL_REGISTRY:
                print(f"[WARN] Unknown model key '{model_key}', skipping.")
                continue

            output_dir = _MODELS_DIR / f"{model_key}_sentiment"
            if output_dir.exists() and not args.overwrite:
                print(
                    f"\n[skip] {model_key} — checkpoint already exists at {output_dir}. "
                    "Use --overwrite to re-train."
                )
                continue

            print(f"\n{'='*60}")
            model_cfg = MODEL_REGISTRY[model_key]
            model_display = model_cfg.get("model_id", model_key) if isinstance(model_cfg, dict) else model_cfg
            print(f"Fine-tuning: {model_key}  ({model_display})")
            print(f"{'='*60}")

            fine_tune_sentiment(
                model_key=model_key,
                train_dataset=dataset,
                output_dir=output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                val_frac=args.val_frac,
                max_train_rows=max_rows,
                seed=args.seed,
                cache_dir=args.cache_dir,
            )

    print("\nAll requested models fine-tuned.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune multilingual sentiment models."
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model key to fine-tune (default: all models).",
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Use consistency-augmented fine-tuning (CE + symmetric KL on parallel pairs).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(_DEFAULT_CHECKPOINT),
        dest="checkpoint_path",
        help="Path to parallel corpus JSON checkpoint (used with --consistency).",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=1.0,
        dest="consistency_weight",
        help="Weight α for the KL consistency loss term (default: 1.0).",
    )
    parser.add_argument(
        "--min-ce-weight",
        type=float,
        default=1.0,
        dest="min_ce_weight",
        help="Coefficient for CE loss — always 1.0 (never scaled down). Total loss = min_ce_weight * L_ce + consistency_weight * L_kl (default: 1.0).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=50_000,
        dest="max_train_rows",
        help=(
            "Subsample training data to at most this many rows / pairs before "
            "the train/val split.  Set to 0 to use the full dataset (default: 50000)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        dest="batch_size",
        help="Per-device training batch size (default: 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="AdamW peak learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        dest="max_length",
        help="Tokeniser max length (default: 128).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        dest="val_frac",
        help="Fraction of rows / source IDs reserved for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_DEFAULT_CACHE_DIR),
        dest="cache_dir",
        help="HuggingFace cache directory (default: data/raw).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-train even if a checkpoint already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
