"""
Parallel corpus construction via Helsinki-NLP OPUS-MT translation models.

Workflow
--------
1. Start from a DataFrame of English source texts + gold labels.
2. Translate each text into each of the 26 target languages using
   Helsinki-NLP/opus-mt-en-{lang} models.
3. Checkpoint after every language so crashes do not lose progress.
4. Return / reload a tidy DataFrame ready for inference.

Translation model reference
---------------------------
Helsinki-NLP OPUS-MT:
    Tiedemann, J., & Thottingal, S. (2020).
    OPUS-MT — Building open translation services for the World.
    EAMT 2020. https://aclanthology.org/2020.eamt-1.61/

Dataset source (English originals):
    Brand24/mms multilingual social-media sentiment dataset.
    https://huggingface.co/datasets/Brand24/mms
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Target language registry
# ---------------------------------------------------------------------------

#: Target language codes.  Languages without a direct ``opus-mt-en-{lang}``
#: model are handled via :data:`LANG_MODEL_OVERRIDES` below.  A language is
#: silently skipped at runtime if its model cannot be loaded.
OPUS_MT_LANGS: list[str] = [
    "ar",  # Arabic
    "bg",  # Bulgarian
    "bs",  # Bosnian         — via opus-mt-en-sla
    "cs",  # Czech
    "da",  # Danish
    "de",  # German
    "el",  # Greek
    "es",  # Spanish
    "fi",  # Finnish
    "fr",  # French
    "he",  # Hebrew
    "hi",  # Hindi
    "hr",  # Croatian        — via opus-mt-en-sla
    "hu",  # Hungarian
    "id",  # Indonesian
    "it",  # Italian
    "ja",  # Japanese        — no model available; silently skipped
    "ko",  # Korean          — no model available; silently skipped
    "lv",  # Latvian         — via opus-mt-tc-big-en-lv
    "nl",  # Dutch
    "no",  # Norwegian       — no model available; silently skipped
    "pl",  # Polish          — via opus-mt-en-sla
    "pt",  # Portuguese      — via opus-mt-en-ROMANCE
    "ro",  # Romanian
    "ru",  # Russian
    "sl",  # Slovenian       — via opus-mt-en-sla
    "sr",  # Serbian         — via opus-mt-en-sla (Cyrillic)
    "sv",  # Swedish
    "tr",  # Turkish         — no model available; silently skipped
    "uk",  # Ukrainian
    "zh",  # Chinese
]

#: Overrides for languages that require a group/big model instead of the
#: standard ``Helsinki-NLP/opus-mt-en-{lang}`` checkpoint.
#:
#: Format: ``lang_code -> (model_name, lang_prefix | None)``
#:
#: ``lang_prefix`` is the ``>>id<<`` token that group models require as a
#: sentence-initial target-language specifier.  Set to ``None`` for models
#: that are already language-specific (e.g. ``opus-mt-tc-big-en-lv``).
LANG_MODEL_OVERRIDES: dict[str, tuple[str, str | None]] = {
    "bs": ("Helsinki-NLP/opus-mt-en-sla", ">>bos_Latn<<"),
    "hr": ("Helsinki-NLP/opus-mt-en-sla", ">>hrv<<"),
    "lv": ("Helsinki-NLP/opus-mt-tc-big-en-lv", None),
    "pl": ("Helsinki-NLP/opus-mt-en-sla", ">>pol<<"),
    "pt": ("Helsinki-NLP/opus-mt-en-ROMANCE", ">>pt<<"),
    "sl": ("Helsinki-NLP/opus-mt-en-sla", ">>slv<<"),
    "sr": ("Helsinki-NLP/opus-mt-en-sla", ">>srp_Cyrl<<"),
}

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Single-text translation
# ---------------------------------------------------------------------------


def translate_text(
    text: str,
    target_lang: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> str:
    """Translate a single English text into *target_lang*.

    Parameters
    ----------
    text:
        Source English string.
    target_lang:
        ISO 639-1 target language code (used only for caller context; the
        model is already language-specific).
    model:
        Loaded ``MarianMTModel`` for the target language.
    tokenizer:
        Corresponding ``MarianTokenizer``.
    max_length:
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Translated text.
    """
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Batch translation helper
# ---------------------------------------------------------------------------


def _translate_batch(
    texts: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    lang_prefix: str | None = None,
) -> list[str]:
    """Translate a list of texts in batches.

    Parameters
    ----------
    lang_prefix:
        Optional ``>>id<<`` token prepended to every source text.  Required
        by group models (e.g. ``opus-mt-en-sla``, ``opus-mt-en-ROMANCE``)
        to select the target language.  Ignored when ``None``.
    """
    if lang_prefix is not None:
        texts = [f"{lang_prefix} {t}" for t in texts]

    results: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=max_length)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results.extend(decoded)
    return results


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_or_resume_corpus(checkpoint_path: str | Path) -> dict[str, list]:
    """Load an existing translation checkpoint if present.

    Parameters
    ----------
    checkpoint_path:
        Path to the JSON checkpoint file written by :func:`build_parallel_corpus`.

    Returns
    -------
    dict
        The checkpoint dict ``{lang: [{source_id, source_text, gold_label,
        lang, translated_text}, ...]}`` or an empty dict if the file does not
        exist.
    """
    path = Path(checkpoint_path)
    if path.exists():
        print(f"[resume] Loading checkpoint from {path}")
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(
    checkpoint: dict[str, list],
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main corpus builder
# ---------------------------------------------------------------------------


def build_parallel_corpus(
    source_df: pd.DataFrame,
    target_langs: Optional[list[str]] = None,
    cache_dir: Optional[str | Path] = None,
    checkpoint_path: str | Path = "data/processed/parallel_checkpoint.json",
    batch_size: int = 16,
) -> pd.DataFrame:
    """Translate English source texts into all target languages.

    For each target language the function:

    1. Checks whether a checkpoint already contains that language (skips if so).
    2. Loads ``Helsinki-NLP/opus-mt-en-{lang}`` from HuggingFace (skips the
       language silently if the model does not exist).
    3. Translates all source texts in batches of *batch_size*.
    4. Appends the results to the checkpoint JSON.

    The English source texts themselves are included in the output as language
    ``"en"`` (no translation needed).

    Parameters
    ----------
    source_df:
        DataFrame with at least columns ``["source_id", "source_text",
        "gold_label"]`` — as produced by
        :func:`src.data.loader.sample_english_source`.
    target_langs:
        List of ISO 639-1 codes to translate into.  Defaults to
        :data:`OPUS_MT_LANGS`.
    cache_dir:
        Optional HuggingFace model cache directory.
    checkpoint_path:
        Path where the JSON checkpoint is written after each language.
    batch_size:
        Number of texts per translation batch.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns
        ``["source_id", "source_text", "gold_label", "lang", "translated_text"]``.
        The ``"en"`` row has ``translated_text == source_text``.
    """
    if target_langs is None:
        target_langs = OPUS_MT_LANGS

    checkpoint_path = Path(checkpoint_path)
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    checkpoint = load_or_resume_corpus(checkpoint_path)

    source_ids = source_df["source_id"].tolist()
    source_texts = source_df["source_text"].tolist()
    gold_labels = source_df["gold_label"].tolist()

    # Include English source as-is.
    if "en" not in checkpoint:
        checkpoint["en"] = [
            {
                "source_id": int(sid),
                "source_text": st,
                "gold_label": int(gl),
                "lang": "en",
                "translated_text": st,
            }
            for sid, st, gl in zip(source_ids, source_texts, gold_labels)
        ]
        _save_checkpoint(checkpoint, checkpoint_path)
        print("  [en] included source texts (no translation needed)")

    for lang in target_langs:
        if lang in checkpoint:
            print(f"  [skip] {lang} — already in checkpoint")
            continue

        override = LANG_MODEL_OVERRIDES.get(lang)
        if override is not None:
            model_name, lang_prefix = override
        else:
            model_name, lang_prefix = f"Helsinki-NLP/opus-mt-en-{lang}", None

        try:
            print(f"  [{lang}] Loading {model_name} …", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir_str
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, cache_dir=cache_dir_str
            )
            model.to(_DEVICE)
            model.eval()
        except Exception as exc:
            print(f"  [skip] {lang} — model not available: {exc}")
            continue

        t0 = time.time()
        translations = _translate_batch(
            source_texts, model, tokenizer, batch_size=batch_size,
            lang_prefix=lang_prefix,
        )
        elapsed = time.time() - t0
        print(f"  [{lang}] translated {len(translations)} texts in {elapsed:.1f}s")

        checkpoint[lang] = [
            {
                "source_id": int(sid),
                "source_text": st,
                "gold_label": int(gl),
                "lang": lang,
                "translated_text": tr,
            }
            for sid, st, gl, tr in zip(source_ids, source_texts, gold_labels, translations)
        ]
        _save_checkpoint(checkpoint, checkpoint_path)

        # Free GPU memory between languages.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Flatten checkpoint into a tidy DataFrame.
    rows = [row for lang_rows in checkpoint.values() for row in lang_rows]
    df = pd.DataFrame(
        rows,
        columns=["source_id", "source_text", "gold_label", "lang", "translated_text"],
    )
    return df.reset_index(drop=True)
