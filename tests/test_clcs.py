"""
Unit tests for src/metrics/clcs.py.

Covers:
- pairwise_clcs: perfect, zero, partial consistency; mismatched lengths
- soft_pairwise_clcs: score in [0, 1]; identical distributions give 1.0;
  mismatched lengths raise ValueError
- compute_all_pairwise: correct number of pairs; soft mode
- clcs_matrix: symmetry; diagonal equals 1.0
- family_clcs: correct grouping
- global_clcs: single scalar
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running tests from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.clcs import (
    clcs_matrix,
    compute_all_pairwise,
    family_clcs,
    global_clcs,
    pairwise_clcs,
    pairwise_kappa,
    soft_pairwise_clcs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LANGS = ["en", "fr", "de"]


@pytest.fixture()
def perfect_preds() -> dict[str, list[int]]:
    """All languages predict the exact same labels."""
    labels = [0, 1, 2, 0, 1]
    return {lang: labels[:] for lang in LANGS}


@pytest.fixture()
def zero_preds() -> dict[str, list[int]]:
    """'en' and 'fr' never agree."""
    return {
        "en": [0, 0, 0, 0],
        "fr": [1, 1, 1, 1],
    }


@pytest.fixture()
def partial_preds() -> dict[str, list[int]]:
    """'en' and 'fr' agree on exactly half the samples."""
    return {
        "en": [0, 1, 2, 0],
        "fr": [0, 0, 2, 1],
    }


@pytest.fixture()
def identical_probs() -> dict[str, np.ndarray]:
    """Both languages have identical probability distributions."""
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    return {"en": probs.copy(), "fr": probs.copy()}


@pytest.fixture()
def diff_probs() -> dict[str, np.ndarray]:
    """Two clearly different probability distributions."""
    return {
        "en": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "fr": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    }


# ---------------------------------------------------------------------------
# pairwise_clcs
# ---------------------------------------------------------------------------


class TestPairwiseClcs:
    def test_perfect_consistency(self, perfect_preds: dict) -> None:
        score = pairwise_clcs(perfect_preds, "en", "fr")
        assert score == pytest.approx(1.0)

    def test_zero_consistency(self, zero_preds: dict) -> None:
        score = pairwise_clcs(zero_preds, "en", "fr")
        assert score == pytest.approx(0.0)

    def test_partial_consistency(self, partial_preds: dict) -> None:
        score = pairwise_clcs(partial_preds, "en", "fr")
        assert score == pytest.approx(0.5)

    def test_symmetric(self, partial_preds: dict) -> None:
        assert pairwise_clcs(partial_preds, "en", "fr") == pairwise_clcs(
            partial_preds, "fr", "en"
        )

    def test_mismatched_lengths_raises(self) -> None:
        preds = {"en": [0, 1, 2], "fr": [0, 1]}
        with pytest.raises(ValueError, match="different lengths"):
            pairwise_clcs(preds, "en", "fr")

    def test_same_language(self, perfect_preds: dict) -> None:
        assert pairwise_clcs(perfect_preds, "en", "en") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# soft_pairwise_clcs
# ---------------------------------------------------------------------------


class TestSoftPairwiseClcs:
    def test_identical_distributions_give_one(self, identical_probs: dict) -> None:
        score = soft_pairwise_clcs(identical_probs, "en", "fr")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_score_in_zero_one(self, diff_probs: dict) -> None:
        score = soft_pairwise_clcs(diff_probs, "en", "fr")
        assert 0.0 <= score <= 1.0

    def test_divergent_distributions_below_one(self, diff_probs: dict) -> None:
        score = soft_pairwise_clcs(diff_probs, "en", "fr")
        assert score < 0.5

    def test_mismatched_lengths_raises(self) -> None:
        probs = {
            "en": np.array([[0.5, 0.3, 0.2]]),
            "fr": np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]),
        }
        with pytest.raises(ValueError, match="different numbers of samples"):
            soft_pairwise_clcs(probs, "en", "fr")

    def test_symmetric(self, identical_probs: dict) -> None:
        assert soft_pairwise_clcs(
            identical_probs, "en", "fr"
        ) == soft_pairwise_clcs(identical_probs, "fr", "en")


# ---------------------------------------------------------------------------
# compute_all_pairwise
# ---------------------------------------------------------------------------


class TestComputeAllPairwise:
    def test_pair_count(self, perfect_preds: dict) -> None:
        df = compute_all_pairwise(perfect_preds, LANGS)
        expected_pairs = len(LANGS) * (len(LANGS) - 1) // 2
        assert len(df) == expected_pairs

    def test_columns(self, perfect_preds: dict) -> None:
        df = compute_all_pairwise(perfect_preds, LANGS)
        assert list(df.columns) == ["lang_a", "lang_b", "clcs"]

    def test_all_perfect(self, perfect_preds: dict) -> None:
        df = compute_all_pairwise(perfect_preds, LANGS)
        assert all(v == pytest.approx(1.0) for v in df["clcs"])

    def test_soft_mode(self, identical_probs: dict) -> None:
        df = compute_all_pairwise(
            {},
            ["en", "fr"],
            soft=True,
            probabilities=identical_probs,
        )
        assert len(df) == 1
        assert df["clcs"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_soft_requires_probabilities(self, perfect_preds: dict) -> None:
        with pytest.raises(ValueError, match="probabilities must be provided"):
            compute_all_pairwise(perfect_preds, LANGS, soft=True)


# ---------------------------------------------------------------------------
# clcs_matrix
# ---------------------------------------------------------------------------


class TestClcsMatrix:
    def _pairwise_df(self, preds: dict, langs: list[str]):
        return compute_all_pairwise(preds, langs)

    def test_diagonal_is_one(self, perfect_preds: dict) -> None:
        df = self._pairwise_df(perfect_preds, LANGS)
        mat = clcs_matrix(df, LANGS)
        for lang in LANGS:
            assert mat.loc[lang, lang] == pytest.approx(1.0)

    def test_symmetry(self, partial_preds: dict) -> None:
        langs = ["en", "fr"]
        df = compute_all_pairwise(partial_preds, langs)
        mat = clcs_matrix(df, langs)
        assert mat.loc["en", "fr"] == pytest.approx(mat.loc["fr", "en"])

    def test_shape(self, perfect_preds: dict) -> None:
        df = self._pairwise_df(perfect_preds, LANGS)
        mat = clcs_matrix(df, LANGS)
        assert mat.shape == (len(LANGS), len(LANGS))


# ---------------------------------------------------------------------------
# family_clcs
# ---------------------------------------------------------------------------


class TestFamilyClcs:
    def test_family_grouping(self, perfect_preds: dict) -> None:
        df = compute_all_pairwise(perfect_preds, LANGS)
        families = {"en": "Germanic", "fr": "Romance", "de": "Germanic"}
        fam_df = family_clcs(df, families)
        # Only Germanic family has an intra-family pair (en-de).
        assert set(fam_df["family"]) == {"Germanic"}

    def test_columns(self, perfect_preds: dict) -> None:
        df = compute_all_pairwise(perfect_preds, LANGS)
        families = {"en": "Germanic", "fr": "Germanic", "de": "Germanic"}
        fam_df = family_clcs(df, families)
        assert "family" in fam_df.columns
        assert "mean_clcs" in fam_df.columns
        assert "num_pairs" in fam_df.columns


# ---------------------------------------------------------------------------
# pairwise_kappa
# ---------------------------------------------------------------------------


class TestPairwiseKappa:
    def test_perfect_agreement_is_one(self, perfect_preds: dict) -> None:
        kappa = pairwise_kappa(perfect_preds, "en", "fr")
        assert kappa == pytest.approx(1.0)

    def test_kappa_in_range(self, partial_preds: dict) -> None:
        kappa = pairwise_kappa(partial_preds, "en", "fr")
        assert -1.0 <= kappa <= 1.0

    def test_zero_agreement_not_positive(self, zero_preds: dict) -> None:
        # en=[0,0,0,0] vs fr=[1,1,1,1]: P_e=0 so kappa=(0-0)/1=0, not negative.
        # Kappa is negative only when P_e > 0 but P_o < P_e.
        kappa = pairwise_kappa(zero_preds, "en", "fr")
        assert kappa <= 0.0

    def test_mismatched_lengths_raises(self) -> None:
        preds = {"en": [0, 1, 2], "fr": [0, 1]}
        with pytest.raises(ValueError, match="different lengths"):
            pairwise_kappa(preds, "en", "fr")

    def test_symmetric(self, partial_preds: dict) -> None:
        assert pairwise_kappa(partial_preds, "en", "fr") == pytest.approx(
            pairwise_kappa(partial_preds, "fr", "en")
        )


# ---------------------------------------------------------------------------
# global_clcs
# ---------------------------------------------------------------------------


class TestGlobalClcs:
    def test_returns_float(self, perfect_preds: dict) -> None:
        df = compute_all_pairwise(perfect_preds, LANGS)
        score = global_clcs(df)
        assert isinstance(score, float)

    def test_perfect_global_is_one(self, perfect_preds: dict) -> None:
        df = compute_all_pairwise(perfect_preds, LANGS)
        assert global_clcs(df) == pytest.approx(1.0)
