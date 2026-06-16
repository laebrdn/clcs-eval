#!/usr/bin/env python3
"""
XLM-R consistency-augmented fine-tuning.

    Total Loss = CE Loss + λ × Consistency Penalty

- CE Loss: cross-entropy on sentiment-labeled data (Brand24/mms if available;
  falls back to the locally-cached MMS parquet shard in data/raw/downloads/)
- Consistency Penalty: symmetric KL divergence between model output distributions
  for parallel pairs (same source text, two different languages), sampled
  on-the-fly from data/processed/parallel_corpus_train.csv
- λ (--lambda): consistency weight; sweep {0.1, 0.5, 1.0}

Data note: Brand24/mms is gated on HuggingFace (requires HF_TOKEN). When
unavailable the script auto-detects the locally-cached parquet shard(s) in
data/raw/downloads/ and uses those for CE training. The parallel corpus is
kept as KL source + CLCS eval only (no CE overlap → clean evaluation).

Warm-start: by default loads models/xlmr-base_sentiment/ (the sentiment
checkpoint) so consistency training preserves multilingual accuracy.

Checkpoints are saved every epoch to:
    results/checkpoints/xlmr_lambda{λ}/epoch_{n}/

Usage
-----
    python scripts/train_consistency.py --lambda 0.5
    python scripts/train_consistency.py --lambda 0.1 --epochs 5
    python scripts/train_consistency.py --lambda 1.0 --no-warm-start
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_MODEL_ID = "xlm-roberta-base"
_SENTIMENT_CKPT = _REPO_ROOT / "models" / "xlmr-base_sentiment"
_PARALLEL_CORPUS_TRAIN_PATH = _REPO_ROOT / "data" / "processed" / "parallel_corpus_train.csv"
_PARALLEL_CORPUS_EVAL_PATH  = _REPO_ROOT / "data" / "processed" / "parallel_corpus_eval.csv"
_CHECKPOINTS_DIR = _REPO_ROOT / "results" / "checkpoints"
_CACHE_DIR = _REPO_ROOT / "data" / "raw"
_DOWNLOADS_DIR = _REPO_ROOT / "data" / "raw" / "downloads"
_NUM_LABELS = 3  # 0=negative, 1=neutral, 2=positive


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# CE dataset (Brand24/mms)
# ---------------------------------------------------------------------------


class _MmsDataset(Dataset):
    """Flat sentiment dataset backed by Brand24/mms rows for CE training."""

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
# Parallel pair sampler (on-the-fly KL penalty)
# ---------------------------------------------------------------------------


class _ParallelPairSampler:
    """Provides random (text_a, text_b) pairs from the parallel corpus.

    For each call to sample_batch(n), picks n source IDs at random and
    samples 2 distinct language translations from each.  The pairs are
    returned as tokenized tensors, ready for a forward pass.
    """

    def __init__(
        self,
        parallel_df: pd.DataFrame,
        tokenizer,
        max_length: int,
        seed: int,
    ) -> None:
        """Build per-source-ID text groups from the parallel corpus.

        Parameters
        ----------
        parallel_df:
            DataFrame with columns ``[source_id, translated_text]`` from
            ``parallel_corpus_train.csv``.
        tokenizer:
            HuggingFace tokenizer used in :meth:`sample_batch`.
        max_length:
            Tokenisation max length.
        seed:
            Random seed for reproducible pair sampling.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._rng = random.Random(seed)

        # Build list-of-texts per source_id; keep only groups with ≥ 2 entries.
        self._groups: list[list[str]] = [
            grp["translated_text"].dropna().tolist()
            for _, grp in parallel_df.groupby("source_id")
            if grp["translated_text"].notna().sum() >= 2
        ]
        if not self._groups:
            raise RuntimeError("Parallel corpus has no source_id with ≥ 2 translations.")

    def sample_batch(self, n: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return two tokenized batches of n parallel texts (same source IDs)."""
        texts_a: list[str] = []
        texts_b: list[str] = []
        for _ in range(n):
            group = self._rng.choice(self._groups)
            a, b = self._rng.sample(group, 2)
            texts_a.append(a)
            texts_b.append(b)

        enc_a = self.tokenizer(
            texts_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            texts_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return dict(enc_a), dict(enc_b)


# ---------------------------------------------------------------------------
# CLCS evaluation on the full parallel corpus
# ---------------------------------------------------------------------------


def _compute_clcs(
    model,
    tokenizer,
    parallel_df: pd.DataFrame,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Hard CLCS (pairwise label agreement) across all source IDs and language pairs."""
    model.eval()
    texts = parallel_df["translated_text"].fillna("").tolist()
    sids = parallel_df["source_id"].tolist()
    langs = parallel_df["lang"].fillna("en").tolist()

    all_preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            all_preds.extend(model(**enc).logits.argmax(dim=-1).cpu().tolist())

    pred_map: dict[int, dict[str, int]] = {}
    for sid, lang, pred in zip(sids, langs, all_preds):
        pred_map.setdefault(int(sid), {})[str(lang)] = pred

    agree = total = 0
    for lang_preds in pred_map.values():
        vals = list(lang_preds.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                total += 1
                agree += int(vals[i] == vals[j])

    return agree / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# CE data loader (Brand24/mms preferred; cached parquet fallback)
# ---------------------------------------------------------------------------


def _load_ce_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load CE training data from Brand24/mms or fall back to cached parquet shards."""
    import os

    # Try Brand24/mms via HuggingFace.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from datasets import load_dataset
            print("[CE] Trying Brand24/mms from HuggingFace Hub …")
            mms = load_dataset("Brand24/mms", cache_dir=str(_CACHE_DIR), token=token)
            ds = mms["train"] if "train" in mms else next(iter(mms.values()))
            df = ds.to_pandas()
            for col in ("sentence", "content", "review"):
                if col in df.columns:
                    df = df.rename(columns={col: "text"})
                    break
            df = df.dropna(subset=["text", "label"])
            df["label"] = df["label"].astype(int)
            df.attrs["source"] = "Brand24/mms (HuggingFace)"
            return df
        except Exception as e:
            print(f"[CE] Brand24/mms unavailable ({e}); falling back to cached shards.")

    # Fallback: read all .parquet-like blobs from data/raw/downloads/.
    # These are the MMS parquet shards downloaded on the original machine.
    parquet_files = [
        p for p in _DOWNLOADS_DIR.iterdir()
        if not p.name.endswith(".json") and not p.name.endswith(".lock")
    ]
    if not parquet_files:
        raise RuntimeError(
            "Brand24/mms is not accessible (set HF_TOKEN) and no cached parquet "
            f"shards were found in {_DOWNLOADS_DIR}."
        )

    print(f"[CE] Loading {len(parquet_files)} cached parquet shard(s) from data/raw/downloads/ …")
    dfs = []
    for p in sorted(parquet_files):
        try:
            shard = pd.read_parquet(p)
            if "text" in shard.columns and "label" in shard.columns:
                dfs.append(shard)
                lang = shard["language"].iloc[0] if "language" in shard.columns else "?"
                print(f"  {p.name[:16]}…  {len(shard):,} rows  lang={lang}")
        except Exception:
            pass  # skip non-parquet blobs (translation model weights, etc.)

    if not dfs:
        raise RuntimeError(f"No readable parquet shards found in {_DOWNLOADS_DIR}.")

    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df.attrs["source"] = f"cached MMS shard(s) — {df['language'].unique().tolist() if 'language' in df.columns else '?'}"
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the full consistency-augmented XLM-R training loop.

    Each epoch alternates between CE steps on stratified Brand24/mms samples
    and KL-penalty steps on translation pairs from ``parallel_corpus_train.csv``.
    After every epoch, the model is evaluated for CLCS on the held-out
    ``parallel_corpus_eval.csv`` and accuracy on the CE validation set.
    Checkpoints and a per-epoch metrics JSON are saved to
    ``results/checkpoints/xlmr_lambda{λ}/``.

    Parameters
    ----------
    args:
        Parsed CLI arguments (see :func:`_parse_args`).
    """
    device = _get_device()
    lam: float = args.lam

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = _CHECKPOINTS_DIR / f"xlmr_lambda{lam}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device          : {device}")
    print(f"[train] λ               : {lam}")
    print(f"[train] base model      : {_MODEL_ID}")
    print(f"[train] epochs          : {args.epochs}")
    print(f"[train] batch size      : {args.batch_size}")
    print(f"[train] lr              : {args.lr}")
    print(f"[train] checkpoints     : {checkpoint_dir}")

    # ------------------------------------------------------------------
    # 1. CE training data
    #    Preferred: Brand24/mms via HuggingFace (requires HF_TOKEN if gated).
    #    Fallback: locally-cached parquet shards in data/raw/downloads/.
    #    The parallel corpus is intentionally excluded from CE data to avoid
    #    eval contamination in the per-epoch CLCS measurement.
    # ------------------------------------------------------------------
    df_mms = _load_ce_data(args)
    n_langs = df_mms["language"].nunique() if "language" in df_mms.columns else "?"
    print(f"[train] CE source       : {df_mms.attrs.get('source', 'unknown')}")
    print(f"[train] CE rows         : {len(df_mms):,}  ({n_langs} languages)")

    if args.max_ce_rows and len(df_mms) > args.max_ce_rows:
        # Proportional stratified sample by language so no single language dominates.
        n_target = args.max_ce_rows
        total = len(df_mms)
        parts = []
        for lang, grp in df_mms.groupby("language"):
            quota = max(1, round(n_target * len(grp) / total))
            quota = min(quota, len(grp))
            parts.append(grp.sample(n=quota, random_state=args.seed))
        df_mms = (
            pd.concat(parts)
            .sample(frac=1, random_state=args.seed)
            .reset_index(drop=True)
            .iloc[:n_target]
            .reset_index(drop=True)
        )
        n_langs_sub = df_mms["language"].nunique() if "language" in df_mms.columns else "?"
        print(f"[train] CE subsampled   : {len(df_mms):,}  (stratified, {n_langs_sub} languages)")

    # 90/10 CE train/val split (stratified by label).
    val_parts, train_parts = [], []
    for lbl in sorted(df_mms["label"].unique()):
        grp = df_mms[df_mms["label"] == lbl].sample(frac=1, random_state=args.seed)
        n_val = max(1, int(len(grp) * 0.1))
        val_parts.append(grp.iloc[:n_val])
        train_parts.append(grp.iloc[n_val:])
    df_ce_val = pd.concat(val_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_ce_train = pd.concat(train_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"[train] CE train        : {len(df_ce_train):,}")
    print(f"[train] CE val          : {len(df_ce_val):,}")

    # ------------------------------------------------------------------
    # 2. Parallel corpus — separate train/eval splits to avoid contamination
    # ------------------------------------------------------------------
    print(f"\n[train] Loading parallel corpus splits …")
    df_parallel_train = pd.read_csv(_PARALLEL_CORPUS_TRAIN_PATH)
    df_parallel_eval  = pd.read_csv(_PARALLEL_CORPUS_EVAL_PATH)
    print(f"[train] KL train pairs  : {len(df_parallel_train):,}  ({df_parallel_train['source_id'].nunique()} source IDs)")
    print(f"[train] CLCS eval       : {len(df_parallel_eval):,}  ({df_parallel_eval['source_id'].nunique()} source IDs, held-out)")

    # ------------------------------------------------------------------
    # 3. Tokenizer
    # ------------------------------------------------------------------
    print(f"\n[train] Loading tokenizer for {_MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, cache_dir=str(_CACHE_DIR))

    # ------------------------------------------------------------------
    # 4. Datasets and dataloaders
    # ------------------------------------------------------------------
    ce_train_ds = _MmsDataset(
        df_ce_train["text"].tolist(),
        df_ce_train["label"].tolist(),
        tokenizer,
        args.max_length,
    )
    ce_val_ds = _MmsDataset(
        df_ce_val["text"].tolist(),
        df_ce_val["label"].tolist(),
        tokenizer,
        args.max_length,
    )
    ce_train_loader = DataLoader(
        ce_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    ce_val_loader = DataLoader(
        ce_val_ds, batch_size=args.batch_size * 2, shuffle=False
    )

    pair_sampler = _ParallelPairSampler(df_parallel_train, tokenizer, args.max_length, args.seed)
    print(f"[train] Parallel groups : {len(pair_sampler._groups):,} source IDs with ≥2 translations")

    # ------------------------------------------------------------------
    # 5. Model — resume checkpoint > warm-start > base model
    # ------------------------------------------------------------------
    import re as _re

    start_epoch = 1
    resume_path = Path(args.resume_from).resolve() if args.resume_from else None

    if resume_path is not None:
        if not (resume_path / "config.json").exists():
            raise FileNotFoundError(f"--resume-from: no config.json found in {resume_path}")
        load_from = str(resume_path)
        # Infer completed epoch count from dir name patterns: epoch_N or epoch_N_step_M
        m = _re.search(r"epoch_(\d+)(?:_step_\d+)?$", resume_path.name)
        if m:
            start_epoch = int(m.group(1)) + 1
        print(f"\n[train] Resuming from   : {resume_path.name}")
        print(f"[train] Starting epoch  : {start_epoch}")
    elif not args.no_warm_start and (_SENTIMENT_CKPT / "config.json").exists():
        load_from = str(_SENTIMENT_CKPT)
        print(f"\n[train] Warm-starting from {_SENTIMENT_CKPT.name} …")
    else:
        load_from = _MODEL_ID
        print(f"\n[train] Loading {_MODEL_ID} from scratch …")

    model = AutoModelForSequenceClassification.from_pretrained(
        load_from,
        num_labels=_NUM_LABELS,
        cache_dir=str(_CACHE_DIR),
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    print(f"[train] model loaded to {device}")

    # ------------------------------------------------------------------
    # 6. AdamW + linear warmup/decay
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(ce_train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    if resume_path is not None:
        state_file = resume_path / "training_state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            print(f"[train] Restored optimizer + scheduler from {state_file.name}")
        else:
            # Fast-forward scheduler to the LR position at the end of the last completed epoch.
            completed_steps = (start_epoch - 1) * len(ce_train_loader)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(completed_steps):
                    scheduler.step()
            print(f"[train] Fast-forwarded scheduler {completed_steps:,} steps (no training_state.pt found)")

    print(f"\n[train] total steps     : {total_steps:,}  (warmup: {warmup_steps})")
    print("[train] starting training …\n")

    # ------------------------------------------------------------------
    # 7. Training loop
    #
    # Each step:
    #   loss = CE(mms_batch) + λ × KL_sym(parallel_pair_batch)
    #
    # KL_sym(p_a, p_b) = [KL(p_a || p_b) + KL(p_b || p_a)] / 2
    #
    # where p_a, p_b are softmax distributions over 3 sentiment classes
    # for two translations of the same source text.
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_ce = epoch_kl = 0.0

        for step, ce_batch in enumerate(ce_train_loader, 1):
            # --- CE forward pass (Brand24/mms batch) ---
            ce_batch = {k: v.to(device) for k, v in ce_batch.items()}
            ce_out = model(**ce_batch)
            loss_ce = ce_out.loss  # cross-entropy from HF's built-in labels forwarding

            # --- Consistency forward pass (parallel pairs, on-the-fly) ---
            n_pairs = len(ce_batch["input_ids"])
            enc_a, enc_b = pair_sampler.sample_batch(n_pairs)
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}

            logits_a = model(**enc_a).logits
            logits_b = model(**enc_b).logits

            # Symmetric KL: (KL(p_a||p_b) + KL(p_b||p_a)) / 2
            pa = torch.softmax(logits_a, dim=-1)
            pb = torch.softmax(logits_b, dim=-1)
            loss_kl = (
                F.kl_div(F.log_softmax(logits_a, dim=-1), pb, reduction="batchmean")
                + F.kl_div(F.log_softmax(logits_b, dim=-1), pa, reduction="batchmean")
            ) / 2

            # Total loss = CE + λ × KL_sym
            loss = loss_ce + lam * loss_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_ce += loss_ce.item()
            epoch_kl += loss_kl.item()

            if step % 100 == 0:
                print(
                    f"  epoch {epoch}/{args.epochs}  step {step}/{len(ce_train_loader)}"
                    f"  ce={epoch_ce/step:.4f}  kl={epoch_kl/step:.4f}"
                    f"  total={epoch_ce/step + lam * epoch_kl/step:.4f}",
                    flush=True,
                )

            # Mid-epoch checkpoint every 2000 steps so interruptions don't lose all progress.
            if step % 2000 == 0:
                mid_dir = checkpoint_dir / f"epoch_{epoch}_step_{step}"
                mid_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(mid_dir)
                tokenizer.save_pretrained(mid_dir)
                print(f"  [saved] mid-epoch checkpoint → {mid_dir}", flush=True)

        avg_ce = epoch_ce / len(ce_train_loader)
        avg_kl = epoch_kl / len(ce_train_loader)

        # --- Val CE loss + accuracy ---
        model.eval()
        val_ce_sum = val_correct = 0
        with torch.no_grad():
            for batch in ce_val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_ce_sum += out.loss.item()
                val_correct += (out.logits.argmax(dim=-1) == batch["labels"]).sum().item()
        avg_val_ce = val_ce_sum / len(ce_val_loader)
        val_acc = val_correct / len(ce_val_ds)

        # --- CLCS on held-out eval split only ---
        val_clcs = _compute_clcs(
            model, tokenizer, df_parallel_eval, args.max_length, args.batch_size * 2, device
        )

        print(
            f"\nepoch {epoch}/{args.epochs}  "
            f"train_ce={avg_ce:.4f}  train_kl={avg_kl:.4f}  "
            f"train_total={avg_ce + lam * avg_kl:.4f}  "
            f"val_ce={avg_val_ce:.4f}  val_acc={val_acc:.4f}  "
            f"val_clcs={val_clcs:.4f}",
            flush=True,
        )

        # --- Save per-epoch metrics to JSON ---
        import json
        _RESULTS_DIR = _REPO_ROOT / "results" / "scores"
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = _RESULTS_DIR / f"train_lambda{lam}_metrics.json"
        existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"run": {"lambda": lam, "batch_size": args.batch_size, "lr": args.lr, "max_ce_rows": args.max_ce_rows, "seed": args.seed}, "epochs": {}}
        existing["epochs"][str(epoch)] = {
            "train_ce": round(avg_ce, 4),
            "train_kl": round(avg_kl, 4),
            "train_total": round(avg_ce + lam * avg_kl, 4),
            "val_ce": round(avg_val_ce, 4),
            "val_acc": round(val_acc, 4),
            "val_clcs": round(val_clcs, 4),
        }
        metrics_path.write_text(json.dumps(existing, indent=2))
        print(f"  [saved] metrics → {metrics_path}", flush=True)

        # --- Checkpoint every epoch (model + optimizer/scheduler state) ---
        epoch_dir = checkpoint_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        torch.save(
            {"epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            epoch_dir / "training_state.pt",
        )
        print(f"  [saved] epoch {epoch} → {epoch_dir}", flush=True)

    print(f"\n[train] done. Checkpoints at: {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="XLM-R consistency-augmented fine-tuning: CE(mms) + λ·KL_sym(parallel)."
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.5,
        dest="lam",
        metavar="λ",
        help="Consistency weight λ for the KL penalty (default: 0.5). Sweep {0.1, 0.5, 1.0}.",
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
        help="Tokenizer max sequence length (default: 128).",
    )
    parser.add_argument(
        "--max-ce-rows",
        type=int,
        default=None,
        dest="max_ce_rows",
        help="Subsample Brand24/mms to at most N rows before training (default: all rows).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        dest="no_warm_start",
        help="Load xlm-roberta-base weights instead of the sentiment checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        dest="resume_from",
        metavar="CHECKPOINT_DIR",
        help=(
            "Resume training from a checkpoint directory (e.g. results/checkpoints/xlmr_lambda0.5/epoch_2). "
            "Loads model weights and, if training_state.pt is present, restores optimizer and scheduler state. "
            "The epoch number is inferred from the directory name; training continues from the next epoch."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
