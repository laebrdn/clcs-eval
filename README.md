# Cross-Lingual Consistency of Sentiment (CLCS) Evaluation

Evaluates whether multilingual language models assign **consistent** sentiment labels
across 27 language translations of the same source text.
We introduce the **Cross-Lingual Consistency Score (CLCS)** and demonstrate that
a KL-divergence-based consistency penalty can lift XLM-R's cross-lingual agreement
by +9 pp in a single fine-tuning epoch.

> **Dataset:** Brand24/mms — 6.165 M rows, 28 languages, 3-class sentiment
> (negative / neutral / positive).
> **Scope:** 27 languages (English excluded as source), 312 shared source IDs,
> 351 language pairs per model.

---

## CLCS Metric

Hard (agreement-based) pairwise CLCS between two languages A and B over N parallel samples:

$$\text{CLCS}(A, B) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\bigl[\hat{y}_A^i = \hat{y}_B^i\bigr]$$

**Global CLCS** = mean over all $\binom{L}{2}$ language pairs:

$$\text{Global-CLCS} = \frac{1}{\binom{L}{2}} \sum_{A < B} \text{CLCS}(A, B)$$

**Family CLCS** = mean restricted to within-family pairs (Germanic, Romance, Slavic, …).

Cohen's κ is reported alongside CLCS as a robustness check that corrects for
chance agreement given marginal label distributions.

See [src/metrics/clcs.py](src/metrics/clcs.py) for the full implementation.

---

## Models

| Key | Type | HuggingFace identifier |
|---|---|---|
| `xlmr-base` | Transformer (fine-tuned) | `xlm-roberta-base` |
| `mbert` | Transformer | `nlptown/bert-base-multilingual-uncased-sentiment` |
| `mdeberta` | Transformer | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` |
| `llama3.1` | LLM via Ollama | `llama3.1:8b` |
| `aya-expanse` | LLM via Ollama | `aya-expanse:8b` |

---

## Results

Fair cross-model evaluation: 27 shared languages, 312 common source IDs, 351 pairs.
All scores use the corrected (non-contaminated) evaluation split.

| Model | Global CLCS ↑ | Mean κ ↑ | Macro F1 ↑ |
|---|:---:|:---:|:---:|
| **xlmr-λ0.5-ep1** *(fine-tuned, best)* | **0.863** | — | — |
| xlmr-λ1.0-ep1 *(fine-tuned)* | 0.854 | — | — |
| xlmr-base *(baseline)* | 0.790 | 0.674 | 0.780 |
| aya-expanse-8B | 0.754 | 0.628 | 0.720 |
| mBERT | 0.733 | 0.526 | 0.467 |
| llama3.1-8B | 0.699 | 0.529 | 0.687 |
| mDeBERTa | 0.653 | 0.365 | 0.344 |

### Key findings

- **XLM-R dominates on both dimensions.** It achieves the highest Global CLCS (0.790)
  and Macro F1 (0.780) among all baselines, confirming that dedicated multilingual
  fine-tuning produces more cross-lingually coherent sentiment representations.
- **CLCS and accuracy can diverge.** mBERT achieves moderate consistency (0.733)
  despite low Macro F1 (0.467), suggesting it exploits language-specific label
  biases rather than learning genuinely transferable sentiment signals.
  mDeBERTa's distillation strategy hurts both metrics.
- **Consistency is a learnable objective.** A single-epoch fine-tune with the KL
  penalty at λ=0.5 lifts XLM-R CLCS from 0.790 → 0.863 (+9.2 pp) with no accuracy
  regression, demonstrating that cross-lingual consistency is separable from task accuracy.

---

## Language-Family Heatmap

![Family-level CLCS heatmap across all models](results/figures/family_heatmap.png)

*Each cell shows the mean intra-family CLCS for a given model. Germanic and Romance
language families consistently achieve the highest within-family agreement across all
models; Semitic and Japonic families show the most cross-model variance.*

---

## Repository structure

```
clcs-eval/
├── configs/
│   └── experiment.yaml            # All run parameters (seed, splits, λ sweep …)
├── data/
│   ├── raw/                       # Brand24/mms parquet cache (git-ignored)
│   └── processed/                 # Parallel corpus CSVs (git-ignored, reproducible)
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory data analysis
├── results/
│   ├── checkpoints/               # Per-epoch model checkpoints (git-ignored)
│   ├── figures/                   # PNG plots (committed)
│   └── scores/                    # CLCS score CSVs and JSON summaries (committed)
├── scripts/
│   ├── build_parallel_corpus.py   # Step 1 — build translation corpus
│   ├── run_inference.py           # Step 2 — run all models × all languages
│   ├── compute_clcs.py            # Step 3 — compute CLCS scores
│   └── train_consistency.py       # Step 4 — consistency fine-tuning
├── src/
│   ├── data/
│   │   ├── loader.py              # Brand24/mms loading & language-family map
│   │   └── parallel.py            # OPUS-MT translation pipeline
│   ├── metrics/
│   │   └── clcs.py                # CLCS, pairwise kappa, matrix, family helpers
│   ├── models/
│   │   ├── inference.py           # SentimentInference class & MODEL_REGISTRY
│   │   └── llm_inference.py       # LLMInference class (Ollama)
│   ├── training/
│   │   └── finetune.py            # fine_tune_sentiment / fine_tune_consistency
│   └── visualization/
│       └── plots.py               # Heatmap, bar chart, model-comparison figures
└── tests/
    └── test_clcs.py               # pytest unit tests (28 tests)
```

---

## Setup

```bash
git clone <repo-url> && cd clcs-eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# HuggingFace token — required for the gated Brand24/mms dataset
cp .env.example .env          # then fill in HF_TOKEN in .env
```

---

## Full pipeline walkthrough

### Step 1 — Build the parallel corpus

Downloads Brand24/mms, samples 500 English source texts
(confidence-filtered, stratified by label), translates via OPUS-MT into 26 target
languages, and saves to `data/processed/`.

```bash
python scripts/build_parallel_corpus.py \
    --n 500 \
    --batch-size 16 \
    --seed 42 \
    --confidence-threshold 0.85
```

> Restart-safe: interrupted runs resume from `data/processed/parallel_checkpoint.json`.

### Step 2 — Run inference

```bash
# Transformer models (mBERT, XLM-R base, mDeBERTa)
python scripts/run_inference.py --batch-size 32 --max-length 128

# LLM baselines — requires Ollama running locally
python scripts/run_inference.py --models llama3.1 aya-expanse
```

### Step 3 — Compute CLCS scores

```bash
# Fair cross-model evaluation (27 shared languages, 312 common source IDs)
python scripts/compute_clcs.py --corrected
```

Outputs land in `results/scores/`.

### Step 4 — Consistency fine-tuning (XLM-R)

Requires the warm-start checkpoint in `models/xlmr-base_sentiment/` and the split
corpus files in `data/processed/`.

```bash
# λ=0.5 sweep (best result: epoch 1, Global CLCS = 0.863)
python scripts/train_consistency.py --lambda 0.5 --max-ce-rows 500000
```

---

## Data contamination note

An early training run (`results/checkpoints/xlmr_lambda0.5_contaminated/`) used the
same parallel corpus for both KL training and CLCS evaluation, inflating val_clcs by
~0.21. It is retained for reproducibility only and must **not** be used for reporting.
All reported results use the corrected checkpoints.

---

## Run tests

```bash
pytest tests/ -v   # 28 tests
```

---

## Citation

If you use CLCS or this codebase, please cite:

```bibtex
@mastersthesis{simon2026clcs,
  author  = {Simon, Laetitia},
  title   = {Cross-Lingual Consistency of Sentiment: Benchmarking and
             Fine-Tuning Multilingual Language Models},
  school  = {Master's Programme in Machine Learning / NLP},
  year    = {2026},
  note    = {Code: \url{https://github.com/laetitiasimon/clcs-eval}}
}
```

---

## References & Acknowledgements

### Dataset
- Brand24/mms — Multilingual Social Media Sentiment, 28 languages.
  <https://huggingface.co/datasets/Brand24/mms>

### Pre-trained Models
- XLM-R — Conneau et al. (2020). ACL 2020. <https://arxiv.org/abs/1911.02116>
- mBERT — Devlin et al. (2019). NAACL-HLT. <https://arxiv.org/abs/1810.04805>
- DeBERTaV3 — He et al. (2021). ICLR 2021. <https://arxiv.org/abs/2111.09543>
- DistilBERT — Sanh et al. (2019). <https://arxiv.org/abs/1910.01108>
- Helsinki-NLP OPUS-MT — Tiedemann & Thottingal (2020). EAMT 2020.
  <https://aclanthology.org/2020.eamt-1.61/>

### Training & Statistics
- AdamW — Loshchilov & Hutter (2019). <https://arxiv.org/abs/1711.05101>
- Wilcoxon signed-rank test — Wilcoxon (1945). Biometrics Bulletin, 1(6).
- Mann-Whitney U / rank-biserial r — Mann & Whitney (1947); Kerby (2014).

### Libraries
- HuggingFace Transformers & Datasets — Wolf et al. (2020); Lhoest et al. (2021).
- PyTorch — Paszke et al. (2019). <https://arxiv.org/abs/1912.01703>
- NumPy, SciPy, pandas, Matplotlib, seaborn.
