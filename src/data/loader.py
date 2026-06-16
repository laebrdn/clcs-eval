"""
Data loading utilities for the Brand24/mms multilingual sentiment dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import DatasetDict, load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

LANGUAGE_FAMILIES: dict[str, str] = {
    # Germanic
    "en": "Germanic",
    "de": "Germanic",
    "nl": "Germanic",
    "sv": "Germanic",
    "da": "Germanic",
    "no": "Germanic",
    # Romance
    "fr": "Romance",
    "es": "Romance",
    "it": "Romance",
    "pt": "Romance",
    "ro": "Romance",
    # Slavic
    "pl": "Slavic",
    "cs": "Slavic",
    "sk": "Slavic",
    "ru": "Slavic",
    "uk": "Slavic",
    "bg": "Slavic",
    # Semitic
    "ar": "Semitic",
    "he": "Semitic",
    # Turkic
    "tr": "Turkic",
    # Uralic
    "fi": "Uralic",
    "hu": "Uralic",
    # Sino-Tibetan
    "zh": "Sino-Tibetan",
    # Japonic
    "ja": "Japonic",
    # Indo-Iranian
    "hi": "Indo-Iranian",
    "fa": "Indo-Iranian",
    # Koreanic
    "ko": "Koreanic",
}


def get_language_family(lang_code: str) -> str:
    """Return the language family name for a given ISO 639-1 code.

    Parameters
    ----------
    lang_code:
        Two-letter ISO 639-1 language code (e.g. ``"en"``, ``"fr"``).

    Returns
    -------
    str
        Language family name, or ``"Unknown"`` if the code is not in
        :data:`LANGUAGE_FAMILIES`.
    """
    return LANGUAGE_FAMILIES.get(lang_code, "Unknown")


def load_mms(
    split: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
) -> DatasetDict:
    """Load the Brand24/mms multilingual sentiment dataset from HuggingFace.

    Parameters
    ----------
    split:
        Dataset split to load (e.g. ``"train"``, ``"test"``).  If ``None``
        (default) the full :class:`~datasets.DatasetDict` is returned.
    cache_dir:
        Optional path to a local cache directory for the HuggingFace datasets
        library.

    Returns
    -------
    DatasetDict
        The loaded dataset (or a single split if *split* is given).
    """
    cache_dir = str(cache_dir) if cache_dir is not None else None
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dataset = load_dataset("Brand24/mms", cache_dir=cache_dir, token=token, trust_remote_code=True)

    # --- print schema and row counts ---
    print("=== Brand24/mms schema ===")
    first_split = next(iter(dataset.values()))
    print(first_split.features)
    print()
    print("=== Row counts per split ===")
    for name, ds in dataset.items():
        print(f"  {name}: {len(ds):,} rows")
    print()

    if split is not None:
        return dataset[split]
    return dataset


def sample_english_source(
    dataset,
    n: int = 500,
    stratify_col: str = "label",
    seed: int = 42,
    confidence_threshold: float = 0.5,
) -> "pd.DataFrame":
    """Sample English source texts for parallel corpus construction.

    Filters the dataset to English rows, applies an optional quality filter on
    ``cleanlab_self_confidence``, then draws *n* stratified samples.

    Parameters
    ----------
    dataset:
        A HuggingFace ``Dataset`` or ``DatasetDict`` (the ``"train"`` split is
        used if a ``DatasetDict`` is passed).  May also be a plain
        ``pd.DataFrame``.
    n:
        Number of rows to sample.  If fewer qualifying rows exist, all are
        returned.
    stratify_col:
        Column to stratify on (default ``"label"``).
    seed:
        Random seed for reproducibility.
    confidence_threshold:
        Rows with ``cleanlab_self_confidence <= threshold`` are dropped before
        sampling.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["source_id", "source_text", "gold_label"]``
        (and all original columns retained).  ``source_id`` is a zero-based
        integer index assigned after filtering.
    """
    import pandas as pd
    from datasets import DatasetDict, Dataset

    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    if isinstance(dataset, Dataset):
        df = dataset.to_pandas()
    else:
        df = dataset.copy()

    # Filter to English.
    lang_col = next(
        (c for c in ["language", "lang", "Language"] if c in df.columns), None
    )
    if lang_col is None:
        raise RuntimeError(
            f"Could not identify a language column. Available: {list(df.columns)}"
        )
    df = df[df[lang_col] == "en"].copy()

    # Quality filter.
    conf_col = "cleanlab_self_confidence"
    if conf_col in df.columns:
        df = df[df[conf_col] > confidence_threshold].copy()

    # Detect text column.
    text_col = next(
        (c for c in ["text", "sentence", "content", "review"] if c in df.columns),
        None,
    )
    if text_col is None:
        raise RuntimeError(
            f"Could not identify a text column. Available: {list(df.columns)}"
        )

    df = df.reset_index(drop=True)

    if len(df) <= n:
        sampled = df
    elif stratify_col in df.columns:
        classes = sorted(df[stratify_col].unique())
        k_per_class = n // len(classes)
        parts = []
        for cls in classes:
            grp = df[df[stratify_col] == cls]
            k = min(k_per_class, len(grp))
            parts.append(grp.sample(n=k, random_state=seed))
        sampled = pd.concat(parts)
        # Top-up to n if rounding left us short, drawing from whatever is left.
        if len(sampled) < n:
            remaining = df.drop(sampled.index)
            extra = min(n - len(sampled), len(remaining))
            sampled = pd.concat([sampled, remaining.sample(n=extra, random_state=seed)])
    else:
        sampled = df.sample(n=n, random_state=seed)

    sampled = sampled.reset_index(drop=True)
    sampled.insert(0, "source_id", sampled.index)
    sampled = sampled.rename(columns={text_col: "source_text", stratify_col: "gold_label"})
    return sampled


def build_parallel_samples(
    dataset: DatasetDict,
) -> dict:
    """Build a mapping of parallel (cross-lingual) samples from the dataset.

    TODO: Complete this function after EDA confirms the column schema.
          Expected approach:
          1. Identify the column that groups semantically equivalent sentences
             across languages (e.g. a ``parallel_id`` or ``sample_id`` field).
          2. Group rows by that ID.
          3. Retain only IDs that have an entry for every language so that
             pairwise comparison is fully balanced.
          4. Return a dict keyed by parallel_id whose values are
             ``{lang_code: {"text": ..., "label": ...}}`` mappings.

    Parameters
    ----------
    dataset:
        The full :class:`~datasets.DatasetDict` returned by :func:`load_mms`.

    Returns
    -------
    dict
        ``{parallel_id: {lang_code: {"text": str, "label": int}}}``
    """
    raise NotImplementedError(
        "build_parallel_samples() is not yet implemented. "
        "Run notebooks/01_eda.ipynb first to confirm the parallel-sample "
        "column schema, then complete this function."
    )
