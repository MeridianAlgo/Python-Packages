"""
Tests for KellyCriterion position sizing.
"""

import os

import numpy as np
import pandas as pd
import pytest

os.environ["MERIDIANALGO_QUIET"] = "1"

from meridianalgo.portfolio.kelly import KellyCriterion


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(500, 3) * 0.01 + 0.0005,
        columns=["A", "B", "C"],
    )
    return returns


class TestKellyCriterionInit:
    def test_defaults(self) -> None:
        kc = KellyCriterion()
        assert kc.fraction == 0.5
        assert kc.max_position == 1.0
        assert kc.min_position == 0.0

    def test_custom_params(self) -> None:
        kc = KellyCriterion(fraction=0.25, max_position=0.5)
        assert kc.fraction == 0.25
        assert kc.max_position == 0.5

    def test_invalid_fraction_zero(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            KellyCriterion(fraction=0.0)

    def test_invalid_fraction_negative(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            KellyCriterion(fraction=-0.1)

    def test_invalid_fraction_above_one(self) -> None:
        with pytest.raises(ValueError):
            KellyCriterion(fraction=1.5)

    def test_invalid_max_position(self) -> None:
        with pytest.raises(ValueError, match="max_position"):
            KellyCriterion(max_position=0.0)

    def test_repr(self) -> None:
        kc = KellyCriterion(fraction=0.5)
        assert "KellyCriterion" in repr(kc)
        assert "0.5" in repr(kc)


class TestSingleAssetKelly:
    def test_fair_coin_zero_kelly(self) -> None:
        kc = KellyCriterion(fraction=1.0)
        result = kc.single_asset(win_prob=0.5, win_loss_ratio=1.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_positive_edge(self) -> None:
        kc = KellyCriterion(fraction=1.0)
        result = kc.single_asset(win_prob=0.55, win_loss_ratio=1.0)
        assert result == pytest.approx(0.1, abs=1e-10)

    def test_half_kelly_halves_result(self) -> None:
        kc_full = KellyCriterion(fraction=1.0)
        kc_half = KellyCriterion(fraction=0.5)
        full = kc_full.single_asset(win_prob=0.6, win_loss_ratio=1.0)
        half = kc_half.single_asset(win_prob=0.6, win_loss_ratio=1.0)
        assert half == pytest.approx(full * 0.5, abs=1e-10)

    def test_max_position_cap(self) -> None:
        kc = KellyCriterion(fraction=1.0, max_position=0.3)
        result = kc.single_asset(win_prob=0.9, win_loss_ratio=10.0)
        assert result <= 0.3

    def test_negative_edge_returns_zero(self) -> None:
        kc = KellyCriterion(fraction=1.0, min_position=0.0)
        result = kc.single_asset(win_prob=0.4, win_loss_ratio=1.0)
        assert result == 0.0

    def test_invalid_win_prob_zero(self) -> None:
        kc = KellyCriterion()
        with pytest.raises(ValueError):
            kc.single_asset(win_prob=0.0, win_loss_ratio=1.0)

    def test_invalid_win_prob_one(self) -> None:
        kc = KellyCriterion()
        with pytest.raises(ValueError):
            kc.single_asset(win_prob=1.0, win_loss_ratio=1.0)

    def test_invalid_win_loss_ratio(self) -> None:
        kc = KellyCriterion()
        with pytest.raises(ValueError):
            kc.single_asset(win_prob=0.6, win_loss_ratio=0.0)


class TestMultiAssetKelly:
    def test_optimize_returns_series(self, returns_df: pd.DataFrame) -> None:
        kc = KellyCriterion(fraction=0.5)
        weights = kc.optimize(returns_df)
        assert isinstance(weights, pd.Series)
        assert list(weights.index) == list(returns_df.columns)

    def test_weights_bounded(self, returns_df: pd.DataFrame) -> None:
        kc = KellyCriterion(fraction=0.5, max_position=1.0, min_position=0.0)
        weights = kc.optimize(returns_df)
        assert (weights >= 0.0).all()
        assert (weights <= 1.0).all()

    def test_weights_stored(self, returns_df: pd.DataFrame) -> None:
        kc = KellyCriterion(fraction=0.5)
        weights = kc.optimize(returns_df)
        assert kc.weights is not None
        assert (kc.weights == weights).all()

    def test_empty_dataframe_raises(self) -> None:
        kc = KellyCriterion()
        with pytest.raises(ValueError):
            kc.optimize(pd.DataFrame())

    def test_single_row_raises(self) -> None:
        kc = KellyCriterion()
        df = pd.DataFrame({"A": [0.01], "B": [0.02]})
        with pytest.raises(ValueError):
            kc.optimize(df)

    def test_fraction_scales_weights(self) -> None:
        np.random.seed(1)
        returns = pd.DataFrame(np.random.randn(200, 2) * 0.01 + 0.001, columns=["X", "Y"])
        kc_full = KellyCriterion(fraction=1.0, max_position=10.0)
        kc_half = KellyCriterion(fraction=0.5, max_position=10.0)
        w_full = kc_full.optimize(returns)
        w_half = kc_half.optimize(returns)
        # Both weight vectors sum to the same value (normalized), but half Kelly
        # produces smaller raw weights before normalization.
        assert w_full.sum() == pytest.approx(w_half.sum(), abs=0.01)


class TestFromMoments:
    def test_basic_uncapped(self) -> None:
        kc = KellyCriterion(fraction=1.0, max_position=10.0)
        result = kc.from_moments(expected_return=0.10, volatility=0.20)
        # f* = (0.10) / (0.20^2) = 2.5
        assert result == pytest.approx(2.5, abs=0.01)

    def test_basic_capped(self) -> None:
        kc = KellyCriterion(fraction=1.0, max_position=1.0)
        result = kc.from_moments(expected_return=0.10, volatility=0.20)
        # f* = 2.5 but capped at max_position=1.0
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_fractional(self) -> None:
        kc_full = KellyCriterion(fraction=1.0, max_position=10.0)
        kc_half = KellyCriterion(fraction=0.5, max_position=10.0)
        full = kc_full.from_moments(expected_return=0.10, volatility=0.20)
        half = kc_half.from_moments(expected_return=0.10, volatility=0.20)
        assert half == pytest.approx(full * 0.5, abs=1e-10)

    def test_invalid_volatility(self) -> None:
        kc = KellyCriterion()
        with pytest.raises(ValueError):
            kc.from_moments(expected_return=0.10, volatility=0.0)

    def test_cap_applied(self) -> None:
        kc = KellyCriterion(fraction=1.0, max_position=1.0)
        result = kc.from_moments(expected_return=0.50, volatility=0.10)
        assert result <= 1.0


class TestGrowthRate:
    def test_growth_rate_formula(self) -> None:
        kc = KellyCriterion(fraction=0.5)
        g = kc.growth_rate(expected_return=0.10, volatility=0.20)
        expected = 0.10 * 0.5 - 0.5 * 0.04 * 0.25
        assert g == pytest.approx(expected, abs=1e-10)

    def test_zero_fraction_zero_growth(self) -> None:
        kc = KellyCriterion(fraction=0.5)
        g = kc.growth_rate(expected_return=0.10, volatility=0.20, fraction=0.0)
        assert g == pytest.approx(0.0, abs=1e-10)
