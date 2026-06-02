"""Tests for the top-level metrics helpers."""

import numpy as np
import pandas as pd
import pytest

import meridianalgo as m


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.01, 500))


def test_summary_stats_returns_dict(returns):
    stats = m.summary_stats(returns)
    assert isinstance(stats, dict)
    for key in (
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_volatility",
    ):
        assert key in stats


def test_summary_stats_as_series(returns):
    stats = m.summary_stats(returns, as_series=True)
    assert isinstance(stats, pd.Series)
    assert "sharpe_ratio" in stats.index


def test_summary_stats_with_benchmark(returns):
    benchmark = returns * 0.8
    stats = m.summary_stats(returns, benchmark=benchmark)
    assert stats["beta"] is not None
    assert stats["alpha"] is not None


def test_tearsheet_is_string(returns):
    report = m.tearsheet(returns)
    assert isinstance(report, str)
    assert "Performance Summary" in report
    assert "Sharpe Ratio" in report


def test_metrics_registered():
    assert m.ModuleRegistry.is_available("metrics")
