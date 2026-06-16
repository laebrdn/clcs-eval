"""
Cross-Lingual Consistency Score (CLCS) metric implementation.

CLCS measures how consistently a multilingual model assigns sentiment labels
to semantically equivalent inputs across different languages.

The CLCS metric (both hard agreement-based and soft KL-based variants) is
an original contribution of this work.

Soft CLCS uses symmetric KL divergence mapped to [0, 1] via exp(-KL_sym):
    KL(p||q) = sum(p * log(p/q))  — standard KL divergence (Kullback & Leibler, 1951)
    KL_sym = 0.5 * (KL(p||q) + KL(q||p))

Statistical tests used in evaluation:
    Wilcoxon signed-rank test: Wilcoxon (1945), doi:10.2307/3001968
    Mann-Whitney U test: Mann & Whitney (1947), doi:10.1214/aoms/1177730491
    Rank-biserial r: Kerby (2014), doi:10.2466/11.IT.3.1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# Pairwise CLCS (hard / agreement-based)
# ---------------------------------------------------------------------------


def pairwise_clcs(
    predictions: dict[str, list[int]],
    lang_a: str,
    lang_b: str,
) -> float:
    """Compute hard agreement-based CLCS between two languages.

    For each parallel sample *i*, the pair is considered consistent if
    ``predictions[lang_a][i] == predictions[lang_b][i]``.  The score is the
    fraction of consistent pairs over all samples.

    Parameters
    ----------
    predictions:
        Mapping ``{lang_code: [label_0, label_1, ...]}`` of integer label
        predictions aligned by sample index.
    lang_a:
        First language code (must be a key in *predictions*).
    lang_b:
        Second language code (must be a key in *predictions*).

    Returns
    -------
    float
        Agreement fraction in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If the prediction lists for *lang_a* and *lang_b* have different
        lengths.
    """
    preds_a = predictions[lang_a]
    preds_b = predictions[lang_b]
    if len(preds_a) != len(preds_b):
        raise ValueError(
            f"Prediction lists have different lengths: "
            f"{lang_a}={len(preds_a)}, {lang_b}={len(preds_b)}"
        )
    if len(preds_a) == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(preds_a, preds_b))
    return matches / len(preds_a)


# ---------------------------------------------------------------------------
# Per-pair Cohen's kappa (robustness check)
# ---------------------------------------------------------------------------


def pairwise_kappa(
    predictions: dict[str, list[int]],
    lang_a: str,
    lang_b: str,
) -> float:
    """Cohen's kappa between two languages as a robustness check alongside hard CLCS.

    Unlike hard CLCS (raw agreement fraction), kappa corrects for agreement
    expected by chance given the marginal label distributions.  Reported
    alongside CLCS to satisfy reviewers who may ask whether high agreement
    is inflated by label imbalance.

    Parameters
    ----------
    predictions:
        Mapping ``{lang_code: [label_0, label_1, ...]}`` of integer label
        predictions aligned by sample index.
    lang_a:
        First language code.
    lang_b:
        Second language code.

    Returns
    -------
    float
        Cohen's kappa in ``[-1.0, 1.0]``.  Returns ``1.0`` for perfect
        agreement and ``0.0`` for chance-level agreement.

    Raises
    ------
    ValueError
        If the prediction lists for *lang_a* and *lang_b* have different
        lengths.
    """
    preds_a = predictions[lang_a]
    preds_b = predictions[lang_b]
    if len(preds_a) != len(preds_b):
        raise ValueError(
            f"Prediction lists have different lengths: "
            f"{lang_a}={len(preds_a)}, {lang_b}={len(preds_b)}"
        )
    if len(preds_a) == 0:
        return 0.0
    return float(cohen_kappa_score(preds_a, preds_b))


# ---------------------------------------------------------------------------
# Soft CLCS (symmetric KL divergence on probability outputs)
# ---------------------------------------------------------------------------


def _sym_kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Symmetric KL divergence between two probability distributions.

    Parameters
    ----------
    p, q:
        Probability vectors (must sum to 1, same length).

    Returns
    -------
    float
        ``0.5 * (KL(p||q) + KL(q||p))``.
    """
    eps = 1e-10
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def soft_pairwise_clcs(
    probabilities: dict[str, NDArray[np.float64]],
    lang_a: str,
    lang_b: str,
) -> float:
    """Compute soft CLCS using symmetric KL divergence on softmax outputs.

    The per-sample symmetric KL divergence is mapped to ``[0, 1]`` via
    ``exp(-kl)`` and then averaged across samples.

    Parameters
    ----------
    probabilities:
        Mapping ``{lang_code: array of shape (N, num_classes)}`` containing
        softmax probability vectors for each sample.
    lang_a:
        First language code.
    lang_b:
        Second language code.

    Returns
    -------
    float
        Soft consistency score in ``[0.0, 1.0]``.  A score of ``1.0`` means
        the distributions are identical; lower values indicate divergence.

    Raises
    ------
    ValueError
        If the probability arrays for *lang_a* and *lang_b* have different
        numbers of rows.
    """
    # Soft CLCS unused — LLM APIs do not expose probability distributions; hard agreement preferred for computational efficiency and cross-model comparability.
    probs_a = np.asarray(probabilities[lang_a], dtype=np.float64)
    probs_b = np.asarray(probabilities[lang_b], dtype=np.float64)
    if probs_a.shape[0] != probs_b.shape[0]:
        raise ValueError(
            f"Probability arrays have different numbers of samples: "
            f"{lang_a}={probs_a.shape[0]}, {lang_b}={probs_b.shape[0]}"
        )
    if probs_a.shape[0] == 0:
        return 0.0
    scores = [
        np.exp(-_sym_kl(probs_a[i], probs_b[i])) for i in range(probs_a.shape[0])
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# All-pairs computation
# ---------------------------------------------------------------------------


def compute_all_pairwise(
    predictions: dict[str, list[int]],
    languages: list[str],
    soft: bool = False,
    probabilities: dict[str, NDArray[np.float64]] | None = None,
) -> pd.DataFrame:
    """Compute CLCS for all ordered language pairs.

    Parameters
    ----------
    predictions:
        Mapping ``{lang_code: [label_0, label_1, ...]}`` of integer label
        predictions aligned by sample index.
    languages:
        Ordered list of language codes to include.
    soft:
        If ``True``, use :func:`soft_pairwise_clcs` (requires *probabilities*).
        If ``False`` (default), use :func:`pairwise_clcs`.
    probabilities:
        Required when *soft* is ``True``.  Mapping
        ``{lang_code: array (N, C)}`` of softmax outputs.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["lang_a", "lang_b", "clcs"]`` covering all
        unique unordered pairs ``(lang_a < lang_b)`` by list index.
    """
    if soft and probabilities is None:
        raise ValueError("probabilities must be provided when soft=True")

    rows: list[dict[str, object]] = []
    for i, la in enumerate(languages):
        for lb in languages[i + 1 :]:
            if soft:
                score = soft_pairwise_clcs(probabilities, la, lb)  # type: ignore[arg-type]
            else:
                score = pairwise_clcs(predictions, la, lb)
            rows.append({"lang_a": la, "lang_b": lb, "clcs": score})

    return pd.DataFrame(rows, columns=["lang_a", "lang_b", "clcs"])


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def family_clcs(
    pairwise_df: pd.DataFrame,
    language_families: dict[str, str],
) -> pd.DataFrame:
    """Aggregate pairwise CLCS scores by language family.

    Only intra-family pairs (where both languages belong to the same family)
    are included in each family's mean.

    Parameters
    ----------
    pairwise_df:
        Output of :func:`compute_all_pairwise` with columns
        ``["lang_a", "lang_b", "clcs"]``.
    language_families:
        Mapping ``{lang_code: family_name}`` (e.g. from
        :data:`src.data.loader.LANGUAGE_FAMILIES`).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["family", "mean_clcs", "num_pairs"]``
        sorted by ``mean_clcs`` descending.
    """
    df = pairwise_df.copy()
    df["family_a"] = df["lang_a"].map(language_families)
    df["family_b"] = df["lang_b"].map(language_families)
    same_family = df[df["family_a"] == df["family_b"]].copy()
    same_family = same_family.rename(columns={"family_a": "family"})
    agg = (
        same_family.groupby("family")["clcs"]
        .agg(mean_clcs="mean", num_pairs="count")
        .reset_index()
        .sort_values("mean_clcs", ascending=False)
    )
    return agg.reset_index(drop=True)


def global_clcs(pairwise_df: pd.DataFrame) -> float:
    """Compute the global mean CLCS across all language pairs.

    Parameters
    ----------
    pairwise_df:
        Output of :func:`compute_all_pairwise`.

    Returns
    -------
    float
        Mean CLCS score across all pairs.
    """
    return float(pairwise_df["clcs"].mean())


def clcs_matrix(
    pairwise_df: pd.DataFrame,
    languages: list[str],
) -> pd.DataFrame:
    """Build a symmetric N×N CLCS matrix suitable for heatmap plotting.

    The diagonal is set to ``1.0`` (perfect self-consistency).

    Parameters
    ----------
    pairwise_df:
        Output of :func:`compute_all_pairwise`.
    languages:
        Ordered list of language codes defining row/column order.

    Returns
    -------
    pd.DataFrame
        Symmetric DataFrame of shape ``(N, N)`` indexed and columned by
        *languages*.
    """
    n = len(languages)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    idx = {lang: i for i, lang in enumerate(languages)}

    for _, row in pairwise_df.iterrows():
        i, j = idx[row["lang_a"]], idx[row["lang_b"]]
        mat[i, j] = row["clcs"]
        mat[j, i] = row["clcs"]

    return pd.DataFrame(mat, index=languages, columns=languages)
