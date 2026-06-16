








#!/usr/bin/env python3
"""
CLI script to run sentiment inference on the parallel corpus.

Loads data/processed/parallel_corpus.csv, groups texts by language, then runs
each sentiment model over each language's texts.  Results are checkpointed per
(model, language) to results/predictions/{model}_{lang}.json so a crash does
not lose completed work.

Usage
-----
    python scripts/run_inference.py [--models mbert xlmr-base mdeberta]
                                    [--corpus data/processed/parallel_corpus.csv]
                                    [--batch-size 32]
                                    [--max-length 128]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.models.inference import MODEL_REGISTRY, SentimentInference
from src.models.llm_inference import LLM_REGISTRY, LLMInference

_PREDICTIONS_DIR = _REPO_ROOT / "results" / "predictions"
_DEFAULT_CORPUS = _REPO_ROOT / "data" / "processed" / "parallel_corpus.csv"


def _checkpoint_path(model_key: str, lang: str) -> Path:
    """Return the expected checkpoint path for a (model, language) combination."""
    return _PREDICTIONS_DIR / f"{model_key}_{lang}.json"


def _save_checkpoint(
    model_key: str,
    lang: str,
    labels: list[int],
    probs: list[list[float]],
    source_ids: list[int],
) -> None:
    """Persist transformer-model predictions for one (model, language) pair to JSON."""
    _PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_key,
        "language": lang,
        "source_ids": source_ids,
        "labels": labels,
        "probs": probs,
    }
    path = _checkpoint_path(model_key, lang)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [saved] {path.relative_to(_REPO_ROOT)}")


def _save_checkpoint_llm(
    model_key: str,
    lang: str,
    labels: list[int],
    source_ids: list[int],
) -> None:
    """Checkpoint for LLM predictions — no probability scores, label -1 excluded."""
    _PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter out unparseable responses (label == -1) before saving.
    filtered_ids = [sid for sid, lbl in zip(source_ids, labels) if lbl != -1]
    filtered_labels = [lbl for lbl in labels if lbl != -1]

    n_dropped = len(labels) - len(filtered_labels)
    if n_dropped:
        print(f"  [filter] dropped {n_dropped} unparseable sample(s) from {model_key}_{lang}")

    payload = {
        "model": model_key,
        "language": lang,
        "source_ids": filtered_ids,
        "labels": filtered_labels,
        "probs": None,  # LLMs do not produce calibrated probabilities
    }
    path = _checkpoint_path(model_key, lang)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [saved] {path.relative_to(_REPO_ROOT)}")


def _load_corpus(corpus_path: Path) -> dict[str, tuple[list[int], list[str]]]:
    """Load parallel corpus and return {lang: (source_ids, texts)}."""
    import pandas as pd

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Parallel corpus not found at {corpus_path}. "
            "Run scripts/build_parallel_corpus.py first."
        )

    df = pd.read_csv(corpus_path)
    df["translated_text"] = df["translated_text"].fillna(df["source_text"])
    result: dict[str, tuple[list[int], list[str]]] = {}
    for lang, group in df.groupby("lang"):
        result[str(lang)] = (
            group["source_id"].tolist(),
            group["translated_text"].tolist(),
        )
    return result


def run(args: argparse.Namespace) -> None:
    """Run inference for every requested (model, language) pair.

    Loads the parallel corpus, then iterates over all requested model keys and
    languages.  Already-completed (model, language) pairs are skipped if their
    checkpoint file exists.  Results are written to
    ``results/predictions/{model}_{lang}.json`` via :func:`_save_checkpoint` or
    :func:`_save_checkpoint_llm`.

    Parameters
    ----------
    args:
        Parsed CLI arguments (see :func:`_parse_args`).
    """
    corpus_path = Path(args.corpus)
    print(f"Loading parallel corpus from {corpus_path} …")
    texts_by_language = _load_corpus(corpus_path)
    languages = sorted(texts_by_language.keys())
    print(f"Languages : {languages}")
    print(f"Models    : {args.models}\n")

    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    _all_known = set(MODEL_REGISTRY.keys()) | set(LLM_REGISTRY.keys())

    for model_key in args.models:
        if model_key not in _all_known:
            print(f"[WARN] Unknown model key '{model_key}', skipping.")
            continue

        is_llm = model_key in LLM_REGISTRY

        print(f"\n{'='*60}")
        if is_llm:
            print(f"Model: {model_key}  (Ollama / zero-shot LLM)")
        else:
            model_cfg = MODEL_REGISTRY[model_key]
            model_display = model_cfg.get("model_id", model_key) if isinstance(model_cfg, dict) else model_cfg
            print(f"Model: {model_key}  ({model_display})")
        print(f"{'='*60}")

        llm_engine: LLMInference | None = None
        transformer_engine: SentimentInference | None = None

        for lang in languages:
            tag = f"{model_key}_{lang}"
            ckpt = _checkpoint_path(model_key, lang)

            if ckpt.exists():
                print(f"  [skip] {tag} — checkpoint already exists")
                skipped.append(tag)
                continue

            source_ids, texts = texts_by_language[lang]
            print(f"  → {lang}: {len(texts)} samples …")

            try:
                if is_llm:
                    if llm_engine is None:
                        llm_engine = LLMInference(model_key)
                    labels, _raw = llm_engine.predict(texts, batch_size=args.batch_size)
                    _save_checkpoint_llm(model_key, lang, labels, source_ids)
                else:
                    if transformer_engine is None:
                        transformer_engine = SentimentInference(
                            model_key,
                            batch_size=args.batch_size,
                            max_length=args.max_length,
                        )
                    labels, probs = transformer_engine.predict(texts, show_progress=True)
                    _save_checkpoint(model_key, lang, labels, probs.tolist(), source_ids)

                completed.append(tag)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {tag}: {exc}")
                failed.append(tag)

    total = len(args.models) * len(languages)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total combinations : {total}")
    print(f"  Completed this run : {len(completed)}")
    print(f"  Skipped (cached)   : {len(skipped)}")
    print(f"  Failed             : {len(failed)}")
    if failed:
        print("\n  Failed combinations:")
        for tag in failed:
            print(f"    - {tag}")
    print()


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run sentiment inference on the parallel corpus."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()) + list(LLM_REGISTRY.keys()),
        help=(
            "Model keys to evaluate. Transformer models: "
            + ", ".join(MODEL_REGISTRY.keys())
            + ". LLM baselines (Ollama): "
            + ", ".join(LLM_REGISTRY.keys())
            + ". Default: all."
        ),
    )
    parser.add_argument(
        "--corpus",
        default=str(_DEFAULT_CORPUS),
        help="Path to parallel_corpus.csv (default: data/processed/parallel_corpus.csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        dest="max_length",
        help="Tokeniser max length (default: 128).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
