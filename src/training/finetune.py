"""
Fine-tuning utilities for multilingual sentiment classification on Brand24/mms.

Workflow
--------
1. Load the mms train split.
2. Optionally subsample to *max_train_rows* (stratified by language + label).
3. 90 / 10 train / val split (stratified by language + label).
4. Fine-tune a HuggingFace sequence-classification model for *num_epochs* epochs.
5. Save the best checkpoint (lowest val loss) to *output_dir*.

Method references
-----------------
AdamW optimizer:
    Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization.
    ICLR 2019. https://arxiv.org/abs/1711.05101

Linear warmup + linear decay (fine-tuning schedule):
    Devlin et al. (2019). BERT. NAACL-HLT. https://arxiv.org/abs/1810.04805

Consistency regularization via symmetric KL divergence (fine_tune_consistency):
    Loss = L_CE + α · (KL(p_a||p_b) + KL(p_b||p_a)) / 2
    Inspired by Mean Teacher (Tarvainen & Valpola, 2017):
    https://arxiv.org/abs/1703.01780
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.models.inference import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

_LABEL_COL = "label"
_TEXT_COL = "text"
_LANG_COL = "language"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _SentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenised (text, label) pairs for CE training."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int,
    ) -> None:
        """Store tokenizer, texts, labels, and max_length."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize and return one sample as a dict of tensors."""
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Stratified subsampling helpers
# ---------------------------------------------------------------------------


def _stratified_subsample(
    df: pd.DataFrame,
    n: int,
    stratify_col: str,
    seed: int,
) -> pd.DataFrame:
    """Draw at most *n* rows from *df* stratified by *stratify_col*."""
    if len(df) <= n:
        return df.copy()
    classes = sorted(df[stratify_col].unique())
    k = n // len(classes)
    parts = []
    for cls in classes:
        grp = df[df[stratify_col] == cls]
        parts.append(grp.sample(n=min(k, len(grp)), random_state=seed))
    sampled = pd.concat(parts)
    # top-up
    if len(sampled) < n:
        remaining = df.drop(sampled.index)
        extra = min(n - len(sampled), len(remaining))
        sampled = pd.concat([sampled, remaining.sample(n=extra, random_state=seed)])
    return sampled.reset_index(drop=True)


def _stratified_split(
    df: pd.DataFrame,
    val_frac: float,
    stratify_col: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val stratified by *stratify_col*."""
    train_parts, val_parts = [], []
    for cls in sorted(df[stratify_col].unique()):
        grp = df[df[stratify_col] == cls].sample(frac=1, random_state=seed)
        n_val = max(1, int(len(grp) * val_frac))
        val_parts.append(grp.iloc[:n_val])
        train_parts.append(grp.iloc[n_val:])
    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Core fine-tuning function
# ---------------------------------------------------------------------------


def fine_tune_sentiment(
    model_key: str,
    train_dataset,
    output_dir: str | Path,
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    val_frac: float = 0.1,
    max_train_rows: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """Fine-tune a multilingual sentiment classifier on *train_dataset*.

    Parameters
    ----------
    model_key:
        Key from :data:`~src.models.inference.MODEL_REGISTRY` or a full
        HuggingFace model identifier.
    train_dataset:
        HuggingFace ``Dataset`` or ``DatasetDict`` (the ``"train"`` split is
        used if a ``DatasetDict`` is passed).  May also be a ``pd.DataFrame``
        with columns ``text``, ``label``, ``language``.
    output_dir:
        Directory where the best model checkpoint is saved.
    num_epochs:
        Number of training epochs.
    batch_size:
        Per-device training batch size.
    lr:
        AdamW peak learning rate.
    max_length:
        Tokeniser maximum sequence length.
    val_frac:
        Fraction of rows reserved for validation (default 0.10).
    max_train_rows:
        If set, subsample the dataset to at most this many rows before splitting
        (stratified by ``language × label``).
    seed:
        Random seed for reproducibility.
    cache_dir:
        Optional HuggingFace model cache directory.

    Returns
    -------
    Path
        Directory containing the saved best model and tokenizer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    _cfg = MODEL_REGISTRY.get(model_key, model_key)
    model_name = _cfg.get("model_id", model_key) if isinstance(_cfg, dict) else _cfg
    print(f"[finetune] model    : {model_name}")
    print(f"[finetune] device   : {_DEVICE}")
    print(f"[finetune] output   : {output_dir}")

    # ------------------------------------------------------------------
    # 1. Convert to DataFrame
    # ------------------------------------------------------------------
    from datasets import DatasetDict, Dataset

    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]
    if isinstance(train_dataset, Dataset):
        df = train_dataset.to_pandas()
    else:
        df = train_dataset.copy()

    # Normalise column names.
    df = df.rename(columns={"sentence": "text", "content": "text", "review": "text"})

    required = {_TEXT_COL, _LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"DataFrame is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df = df[[c for c in [_TEXT_COL, _LABEL_COL, _LANG_COL] if c in df.columns]].dropna(
        subset=[_TEXT_COL, _LABEL_COL]
    )
    df[_LABEL_COL] = df[_LABEL_COL].astype(int)

    print(f"[finetune] rows after cleaning : {len(df):,}")

    # ------------------------------------------------------------------
    # 2. Optional subsample
    # ------------------------------------------------------------------
    if max_train_rows is not None and len(df) > max_train_rows:
        strat_col = "lang_label"
        if _LANG_COL in df.columns:
            df[strat_col] = df[_LANG_COL].astype(str) + "_" + df[_LABEL_COL].astype(str)
        else:
            strat_col = _LABEL_COL
        df = _stratified_subsample(df, max_train_rows, strat_col, seed)
        if "lang_label" in df.columns:
            df = df.drop(columns=["lang_label"])
        print(f"[finetune] subsampled to     : {len(df):,} rows")

    # ------------------------------------------------------------------
    # 3. Train / val split (stratified by label)
    # ------------------------------------------------------------------
    train_df, val_df = _stratified_split(df, val_frac, _LABEL_COL, seed)
    print(f"[finetune] train               : {len(train_df):,}")
    print(f"[finetune] val                 : {len(val_df):,}")

    label_dist = train_df[_LABEL_COL].value_counts().sort_index()
    print("[finetune] train label dist    :")
    for lbl, cnt in label_dist.items():
        print(f"  label {lbl}: {cnt:,}")

    # ------------------------------------------------------------------
    # 4. Tokeniser + datasets
    # ------------------------------------------------------------------
    print(f"\n[finetune] Loading tokeniser for {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_str)

    train_texts = train_df[_TEXT_COL].tolist()
    train_labels = train_df[_LABEL_COL].tolist()
    val_texts = val_df[_TEXT_COL].tolist()
    val_labels = val_df[_LABEL_COL].tolist()

    num_labels = len(set(train_labels))
    print(f"[finetune] num_labels          : {num_labels}")

    train_ds = _SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = _SentimentDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    print(f"[finetune] Loading model {model_name} …")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir_str,
        ignore_mismatched_sizes=True,
    )
    model.to(_DEVICE)

    # ------------------------------------------------------------------
    # 6. Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        # --- train ---
        model.train()
        train_loss_sum = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(_DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss_sum += loss.item()
            if step % 100 == 0:
                print(
                    f"  epoch {epoch}/{num_epochs}  step {step}/{len(train_loader)}"
                    f"  train_loss={train_loss_sum/step:.4f}",
                    flush=True,
                )

        avg_train_loss = train_loss_sum / len(train_loader)

        # --- val ---
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(_DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss_sum += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()

        avg_val_loss = val_loss_sum / len(val_loader)
        val_acc = correct / len(val_ds)
        print(
            f"\nepoch {epoch}/{num_epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  [saved] best checkpoint at epoch {epoch} → {output_dir}", flush=True)

    print(
        f"\n[finetune] done. Best epoch: {best_epoch}  val_loss: {best_val_loss:.4f}"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Consistency pair dataset
# ---------------------------------------------------------------------------


class _ConsistencyPairDataset(Dataset):
    """Yields (text_a, text_b, gold_label) triples from a parallel corpus.

    All C(n_langs, 2) translation pairs are generated for each source ID.
    Tokenisation is performed lazily in ``__getitem__``.
    """

    def __init__(
        self,
        parallel_df: pd.DataFrame,
        tokenizer,
        max_length: int,
    ) -> None:
        """Build all C(n_langs, 2) translation pairs from the parallel corpus.

        Parameters
        ----------
        parallel_df:
            DataFrame with columns ``[source_id, translated_text, gold_label]``.
        tokenizer:
            HuggingFace tokenizer for lazy tokenisation in :meth:`__getitem__`.
        max_length:
            Tokenisation max length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs: list[tuple[str, str, int]] = []

        for _sid, grp in parallel_df.groupby("source_id"):
            texts = grp["translated_text"].tolist()
            label = int(grp["gold_label"].iloc[0])
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    self.pairs.append((texts[i], texts[j], label))

    def __len__(self) -> int:
        """Return the total number of translation pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize and return one translation pair as a dict of tensors."""
        text_a, text_b, label = self.pairs[idx]
        enc_a = self.tokenizer(
            text_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Val CLCS helper
# ---------------------------------------------------------------------------


def _compute_val_clcs(
    model,
    tokenizer,
    val_parallel_df: pd.DataFrame,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute CLCS over all (source_id, lang_pair) combinations in val set."""
    model.eval()
    texts = val_parallel_df["translated_text"].tolist()
    sids = val_parallel_df["source_id"].tolist()
    langs = val_parallel_df["lang"].tolist()

    all_preds: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            preds = model(**enc).logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

    pred_map: dict[int, dict[str, int]] = {}
    for sid, lang, pred in zip(sids, langs, all_preds):
        pred_map.setdefault(int(sid), {})[lang] = pred

    agree = 0
    total = 0
    for lang_preds in pred_map.values():
        vals = list(lang_preds.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                total += 1
                agree += int(vals[i] == vals[j])

    return agree / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Consistency-augmented fine-tuning
# ---------------------------------------------------------------------------


def fine_tune_consistency(
    model_key: str,
    parallel_df: pd.DataFrame,
    output_dir: str | Path,
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    val_frac: float = 0.1,
    max_train_rows: Optional[int] = None,
    consistency_weight: float = 1.0,
    min_ce_weight: float = 1.0,
    warm_start_from: Optional[str | Path] = None,
    seed: int = 42,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """Fine-tune a multilingual classifier with a consistency regularisation term.

    Each training sample is a pair of translations of the same source text.
    The loss is:

        L = L_CE + consistency_weight × L_KL_sym

    where ``L_CE`` is the mean cross-entropy loss for both texts against their
    shared gold label, and ``L_KL_sym`` is the symmetric KL divergence between
    the softmax distributions predicted for ``text_a`` and ``text_b``.

    The val split is performed at the *source ID* level to avoid leakage.
    Best checkpoint (lowest val_ce + α·val_kl) is saved to *output_dir*.

    Parameters
    ----------
    model_key:
        Key from :data:`~src.models.inference.MODEL_REGISTRY`.
    parallel_df:
        Tidy DataFrame with columns ``[source_id, source_text, gold_label,
        lang, translated_text]`` — as produced by
        :func:`src.data.parallel.build_parallel_corpus`.
    output_dir:
        Directory where the best checkpoint is saved.
    num_epochs:
        Number of training epochs.
    batch_size:
        Per-device batch size (each item is a translation pair).
    lr:
        AdamW peak learning rate.
    max_length:
        Tokeniser maximum sequence length.
    val_frac:
        Fraction of source IDs reserved for validation.
    max_train_rows:
        If set, subsample train pairs to at most this many.
    consistency_weight:
        Weight α for the KL consistency term.
    seed:
        Random seed.
    cache_dir:
        HuggingFace model cache directory.

    Returns
    -------
    Path
        Directory containing the saved best model and tokenizer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    _cfg = MODEL_REGISTRY.get(model_key, model_key)
    model_name = _cfg.get("model_id", model_key) if isinstance(_cfg, dict) else _cfg
    print(f"[consistency] model    : {model_name}")
    print(f"[consistency] warm start: {warm_start_from or '(none — base weights)'}")
    print(f"[consistency] device   : {_DEVICE}")
    print(f"[consistency] α        : {consistency_weight}")
    print(f"[consistency] output   : {output_dir}")

    # ------------------------------------------------------------------
    # 1. Split by source_id
    # ------------------------------------------------------------------
    source_ids = parallel_df["source_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(source_ids)
    n_val = max(1, int(len(shuffled) * val_frac))
    val_sids = set(shuffled[:n_val].tolist())

    train_df = parallel_df[~parallel_df["source_id"].isin(val_sids)].reset_index(drop=True)
    val_df   = parallel_df[ parallel_df["source_id"].isin(val_sids)].reset_index(drop=True)

    print(f"[consistency] train source IDs : {len(source_ids) - n_val}")
    print(f"[consistency] val   source IDs : {n_val}")

    # ------------------------------------------------------------------
    # 2. Tokeniser
    # ------------------------------------------------------------------
    print(f"\n[consistency] Loading tokeniser for {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_str)

    # ------------------------------------------------------------------
    # 3. Pair datasets
    # ------------------------------------------------------------------
    print("[consistency] Building pair datasets …")
    train_ds: Dataset = _ConsistencyPairDataset(train_df, tokenizer, max_length)
    val_ds   = _ConsistencyPairDataset(val_df,   tokenizer, max_length)

    print(f"[consistency] train pairs before subsample : {len(train_ds):,}")
    print(f"[consistency] val   pairs                  : {len(val_ds):,}")

    # ------------------------------------------------------------------
    # 4. Optional subsample of train pairs
    # ------------------------------------------------------------------
    if max_train_rows is not None and len(train_ds) > max_train_rows:
        indices = rng.choice(len(train_ds), size=max_train_rows, replace=False).tolist()
        train_ds = Subset(train_ds, indices)
        print(f"[consistency] train pairs after  subsample : {len(train_ds):,}")

    # Cap val pairs for fast per-epoch monitoring (val CLCS uses val_df separately).
    max_val_pairs = 2000
    if len(val_ds) > max_val_pairs:
        val_indices = rng.choice(len(val_ds), size=max_val_pairs, replace=False).tolist()
        val_ds_monitor: Dataset = Subset(val_ds, val_indices)
        print(f"[consistency] val   pairs (capped for monitoring) : {max_val_pairs:,}")
    else:
        val_ds_monitor = val_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds_monitor, batch_size=batch_size * 2, shuffle=False)

    # ------------------------------------------------------------------
    # 5. Model (3-class: neg / neu / pos)
    # ------------------------------------------------------------------
    load_from = str(warm_start_from) if warm_start_from is not None else model_name
    print(f"[consistency] Loading model from {load_from} …")
    model = AutoModelForSequenceClassification.from_pretrained(
        load_from,
        num_labels=3,
        cache_dir=cache_dir_str,
        ignore_mismatched_sizes=True,
    )
    model.to(_DEVICE)

    # ------------------------------------------------------------------
    # 6. Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        # --- train ---
        model.train()
        train_ce_sum = 0.0
        train_kl_sum = 0.0

        for step, batch in enumerate(train_loader, 1):
            labels = batch["labels"].to(_DEVICE)

            out_a = model(
                input_ids=batch["input_ids_a"].to(_DEVICE),
                attention_mask=batch["attention_mask_a"].to(_DEVICE),
                labels=labels,
            )
            out_b = model(
                input_ids=batch["input_ids_b"].to(_DEVICE),
                attention_mask=batch["attention_mask_b"].to(_DEVICE),
                labels=labels,
            )

            loss_ce = (out_a.loss + out_b.loss) / 2

            log_pa = F.log_softmax(out_a.logits, dim=-1)
            log_pb = F.log_softmax(out_b.logits, dim=-1)
            pa = torch.softmax(out_a.logits, dim=-1)
            pb = torch.softmax(out_b.logits, dim=-1)
            loss_kl = (
                F.kl_div(log_pa, pb, reduction="batchmean")
                + F.kl_div(log_pb, pa, reduction="batchmean")
            ) / 2

            loss = min_ce_weight * loss_ce + consistency_weight * loss_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_ce_sum += loss_ce.item()
            train_kl_sum += loss_kl.item()

            if step % 50 == 0:
                print(
                    f"  epoch {epoch}/{num_epochs}  step {step}/{len(train_loader)}"
                    f"  ce={train_ce_sum/step:.4f}  kl={train_kl_sum/step:.4f}",
                    flush=True,
                )

        avg_train_ce = train_ce_sum / len(train_loader)
        avg_train_kl = train_kl_sum / len(train_loader)

        # --- val loss ---
        model.eval()
        val_ce_sum = 0.0
        val_kl_sum = 0.0
        val_pred_counts = {0: 0, 1: 0, 2: 0}
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(_DEVICE)
                out_a = model(
                    input_ids=batch["input_ids_a"].to(_DEVICE),
                    attention_mask=batch["attention_mask_a"].to(_DEVICE),
                    labels=labels,
                )
                out_b = model(
                    input_ids=batch["input_ids_b"].to(_DEVICE),
                    attention_mask=batch["attention_mask_b"].to(_DEVICE),
                    labels=labels,
                )
                val_ce_sum += ((out_a.loss + out_b.loss) / 2).item()
                log_pa = F.log_softmax(out_a.logits, dim=-1)
                log_pb = F.log_softmax(out_b.logits, dim=-1)
                pa = torch.softmax(out_a.logits, dim=-1)
                pb = torch.softmax(out_b.logits, dim=-1)
                val_kl_sum += (
                    (
                        F.kl_div(log_pa, pb, reduction="batchmean")
                        + F.kl_div(log_pb, pa, reduction="batchmean")
                    )
                    / 2
                ).item()
                # Collect label distribution from text_a predictions
                for pred in out_a.logits.argmax(dim=-1).cpu().tolist():
                    val_pred_counts[pred] = val_pred_counts.get(pred, 0) + 1

        avg_val_ce  = val_ce_sum / len(val_loader)
        avg_val_kl  = val_kl_sum / len(val_loader)
        avg_val_loss = avg_val_ce + consistency_weight * avg_val_kl

        # --- label distribution check ---
        total_val_preds = sum(val_pred_counts.values())
        label_dist_str = "  ".join(
            f"{l}={val_pred_counts.get(l,0)/total_val_preds*100:.1f}%"
            for l in [0, 1, 2]
        )
        collapse_warning = ""
        if total_val_preds > 0 and max(val_pred_counts.values()) / total_val_preds > 0.60:
            dominant = max(val_pred_counts, key=val_pred_counts.get)
            collapse_warning = f"  ⚠️  label collapse risk (label {dominant} dominant)"

        # --- val CLCS ---
        val_clcs = _compute_val_clcs(
            model, tokenizer, val_df, max_length, batch_size * 2, _DEVICE
        )

        print(
            f"\nepoch {epoch}/{num_epochs}  "
            f"train_ce={avg_train_ce:.4f}  train_kl={avg_train_kl:.4f}  "
            f"val_ce={avg_val_ce:.4f}  val_kl={avg_val_kl:.4f}  "
            f"val_loss={avg_val_loss:.4f}  val_clcs={val_clcs:.4f}",
            flush=True,
        )
        print(f"  val label dist: {label_dist_str}{collapse_warning}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  [saved] best checkpoint at epoch {epoch} → {output_dir}", flush=True)

    print(
        f"\n[consistency] done. Best epoch: {best_epoch}  val_loss: {best_val_loss:.4f}"
    )
    return output_dir
