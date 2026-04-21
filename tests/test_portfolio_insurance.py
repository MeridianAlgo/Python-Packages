"""Tests for portfolio insurance strategies."""

import numpy as np
import pandas as pd
import pytest

from meridianalgo.portfolio.insurance import CPPI, TimeInvariantCPPI


@pytest.fixture
def equity_returns():
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.standard_normal(n) * 0.01 + 0.0003
    returns[150:160] = -0.05
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(returns, index=dates)


@pytest.fixture
def bear_returns():
    rng = np.random.default_rng(42)
    n = 252
    returns = rng.standard_normal(n) * 0.015 - 0.001
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(returns, index=dates)


class TestCPPI:
    def test_run_returns_result(self, equity_returns):
        cppi = CPPI(multiplier=3.0, floor_pct=0.80)
        result = cppi.run(equity_returns, initial_value=100_000)
        assert result.total_return is not None
        assert result.max_drawdown <= 0

    def test_portfolio_always_above_zero(self, equity_returns):
        cppi = CPPI(multiplier=3.0, floor_pct=0.80)
        result = cppi.run(equity_returns, initial_value=100_000)
        assert result.portfolio_value.min() > 0

    def test_floor_approximately_held_in_normal_markets(self, equity_returns):
        cppi = CPPI(multiplier=3.0, floor_pct=0.80)
        result = cppi.run(equity_returns, initial_value=100_000)
        assert result.floor_breaches <= len(equity_returns) * 0.01

    def test_higher_multiplier_higher_return_in_pure_bull(self):
        rng = np.random.default_rng(0)
        n = 252
        pure_bull = pd.Series(
            rng.standard_normal(n) * 0.008 + 0.0005,
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        cppi_low = CPPI(multiplier=1.0, floor_pct=0.80)
        cppi_high = CPPI(multiplier=5.0, floor_pct=0.80)
        r_low = cppi_low.run(pure_bull, initial_value=100_000).total_return
        r_high = cppi_high.run(pure_bull, initial_value=100_000).total_return
        assert r_high >= r_low

    def test_risky_weight_between_zero_and_max_leverage(self, equity_returns):
        cppi = CPPI(multiplier=3.0, floor_pct=0.80, max_leverage=1.0)
        result = cppi.run(equity_returns, initial_value=100_000)
        assert result.risky_weight.min() >= -0.01
        assert result.risky_weight.max() <= 1.01

    def test_invalid_multiplier_raises(self):
        with pytest.raises(ValueError):
            CPPI(multiplier=-1.0)

    def test_invalid_floor_raises(self):
        with pytest.raises(ValueError):
            CPPI(floor_pct=1.5)

    def test_total_return_is_float(self, equity_returns):
        cppi = CPPI()
        result = cppi.run(equity_returns)
        assert isinstance(result.total_return, float)

    def test_sensitivity_analysis_shape(self, equity_returns):
        cppi = CPPI()
        df = cppi.sensitivity_analysis(
            equity_returns,
            multipliers=[1.0, 3.0, 5.0],
            floor_pcts=[0.80, 0.90],
        )
        assert len(df) == 6
        assert "total_return" in df.columns
        assert "max_drawdown" in df.columns

    def test_floor_protects_in_bear_market(self, bear_returns):
        cppi = CPPI(multiplier=3.0, floor_pct=0.80)
        result = cppi.run(bear_returns, initial_value=100_000)
        final_value = result.portfolio_value.iloc[-1]
        assert final_value >= 100_000 * 0.80 * 0.95

    def test_with_safe_returns(self, equity_returns):
        safe = pd.Series(0.02 / 252, index=equity_returns.index)
        cppi = CPPI(multiplier=3.0, floor_pct=0.80)
        result = cppi.run(equity_returns, safe_returns=safe)
        assert result.total_return is not None


class TestTimeInvariantCPPI:
    def test_run_returns_result(self, equity_returns):
        tipp = TimeInvariantCPPI(multiplier=3.0, floor_pct=0.80)
        result = tipp.run(equity_returns, initial_value=100_000)
        assert result.total_return is not None

    def test_floor_ratchets_up(self, equity_returns):
        tipp = TimeInvariantCPPI(multiplier=3.0, floor_pct=0.80)
        result = tipp.run(equity_returns, initial_value=100_000)
        final_floor = result.floor_value.iloc[-1]
        initial_floor = 100_000 * 0.80
        assert final_floor >= initial_floor

    def test_portfolio_always_above_zero(self, equity_returns):
        tipp = TimeInvariantCPPI()
        result = tipp.run(equity_returns, initial_value=100_000)
        assert result.portfolio_value.min() > 0

    def test_multiplier_stored(self, equity_returns):
        tipp = TimeInvariantCPPI(multiplier=4.0, floor_pct=0.75)
        result = tipp.run(equity_returns)
        assert result.multiplier == 4.0
        assert result.floor_rate == 0.75
