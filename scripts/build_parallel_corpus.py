#!/usr/bin/env python3
"""
CLI script to build the parallel translation corpus from Brand24/mms.

Steps
-----
1. Load Brand24/mms (cached locally after first download).
2. Sample 500 English texts stratified by label via sample_english_source().
3. Translate into all 26 OPUS-MT target languages via build_parallel_corpus().
4. Save final tidy corpus to data/processed/parallel_corpus.csv.

Usage
-----
    python scripts/build_parallel_corpus.py
    python scripts/build_parallel_corpus.py --n 500 --batch-size 16
    python scripts/build_parallel_corpus.py --langs de fr es --cache-dir data/raw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import load_mms, sample_english_source
from src.data.parallel import OPUS_MT_LANGS, build_parallel_corpus

_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_CHECKPOINT_PATH = _PROCESSED_DIR / "parallel_checkpoint.json"
_OUTPUT_PATH = _PROCESSED_DIR / "parallel_corpus.csv"


def run(args: argparse.Namespace) -> None:
    """Execute the full corpus-building pipeline and save the result.

    1. Loads Brand24/mms from the HuggingFace cache.
    2. Samples ``args.n`` English source texts (stratified by label).
    3. Translates them into all target languages via OPUS-MT (or resumes from
       ``data/processed/parallel_checkpoint.json``).
    4. Saves the final tidy corpus to ``data/processed/parallel_corpus.csv``.

    Parameters
    ----------
    args:
        Parsed CLI arguments (see :func:`_parse_args`).
    """
    total_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading Brand24/mms …")
    dataset = load_mms(cache_dir=args.cache_dir)

    # ------------------------------------------------------------------
    # 2. Sample English source texts
    # ------------------------------------------------------------------
    print(f"\nSampling {args.n} English source texts (stratified by label) …")
    source_df = sample_english_source(
        dataset,
        n=args.n,
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
    )
    label_counts = source_df["gold_label"].value_counts().sort_index()
    print(f"  Total sampled : {len(source_df)}")
    print("  Label distribution:")
    for label, count in label_counts.items():
        print(f"    label {label}: {count}")

    # Save source texts for reference.
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source_path = _PROCESSED_DIR / "parallel_source.csv"
    source_df.to_csv(source_path, index=False)
    print(f"  Saved source texts → {source_path.relative_to(_REPO_ROOT)}")

    # ------------------------------------------------------------------
    # 3. Build (or resume) parallel corpus
    # ------------------------------------------------------------------
    target_langs = args.langs if args.langs else OPUS_MT_LANGS
    n_langs = len(target_langs) + 1  # +1 for English
    n_texts = len(source_df)
    print(
        f"\nTranslating {n_texts} texts into {len(target_langs)} languages"
        f" (+ English source = {n_langs} total) …"
    )
    print(f"  Checkpoint     : {_CHECKPOINT_PATH.relative_to(_REPO_ROOT)}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Device         : {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")
    print()

    corpus_df = build_parallel_corpus(
        source_df=source_df,
        target_langs=target_langs,
        cache_dir=args.cache_dir,
        checkpoint_path=_CHECKPOINT_PATH,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # 4. Save final corpus
    # ------------------------------------------------------------------
    corpus_df.to_csv(_OUTPUT_PATH, index=False)
    elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Corpus shape   : {corpus_df.shape}")
    print(f"  Languages      : {sorted(corpus_df['lang'].dropna().unique())}")
    print(f"  Total time     : {elapsed/60:.1f} min")
    print(f"  Saved corpus   → {_OUTPUT_PATH.relative_to(_REPO_ROOT)}")


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the parallel translation corpus for CLCS evaluation."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of English source texts to sample (default: 100).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=None,
        help="Target language codes (default: all 26 OPUS-MT languages).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        dest="batch_size",
        help="Translation batch size (default: 16).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_REPO_ROOT / "data" / "raw"),
        dest="cache_dir",
        help="HuggingFace cache directory (default: data/raw).",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        dest="confidence_threshold",
        help="Minimum cleanlab_self_confidence to keep a row (default: 0.5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
