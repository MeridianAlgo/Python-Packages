"""Tests for benchmark analytics module."""

import numpy as np
import pandas as pd
import pytest

from meridianalgo.analytics.benchmark import (
    ActiveShare,
    BenchmarkAnalytics,
    BrinsonAttribution,
)


@pytest.fixture
def returns_pair():
    rng = np.random.default_rng(42)
    n = 500
    bench = rng.standard_normal(n) * 0.01 + 0.0003
    alpha_signal = rng.standard_normal(n) * 0.002
    port = bench + alpha_signal

    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(port, index=dates), pd.Series(bench, index=dates)


@pytest.fixture
def negative_alpha_pair():
    rng = np.random.default_rng(123)
    n = 300
    bench = rng.standard_normal(n) * 0.01 + 0.0003
    port = bench - 0.0005

    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(port, index=dates), pd.Series(bench, index=dates)


class TestBenchmarkAnalytics:
    def test_active_returns_length(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        ar = analytics.active_returns()
        assert len(ar) == len(port)

    def test_tracking_error_positive(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        te = analytics.tracking_error()
        assert te > 0

    def test_information_ratio_sign_reflects_alpha(self, returns_pair, negative_alpha_pair):
        port_pos, bench_pos = returns_pair
        a_pos = BenchmarkAnalytics(port_pos, bench_pos)
        ir_pos = a_pos.information_ratio()

        port_neg, bench_neg = negative_alpha_pair
        a_neg = BenchmarkAnalytics(port_neg, bench_neg)
        ir_neg = a_neg.information_ratio()

        assert ir_pos > ir_neg

    def test_up_capture_positive(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        uc = analytics.up_capture_ratio()
        assert uc > 0

    def test_down_capture_positive(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        dc = analytics.down_capture_ratio()
        assert dc > 0

    def test_batting_average_between_zero_one(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        ba = analytics.batting_average()
        assert 0 <= ba <= 1

    def test_beta_alpha_r2(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        beta, alpha, r2 = analytics.beta_alpha()
        assert isinstance(beta, float)
        assert isinstance(alpha, float)
        assert 0 <= r2 <= 1

    def test_active_metrics_returns_dataclass(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        m = analytics.active_metrics()
        assert m.tracking_error > 0
        assert isinstance(m.information_ratio, float)
        assert isinstance(m.batting_average, float)

    def test_max_active_drawdown_non_positive(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        madd = analytics.max_active_drawdown()
        assert madd <= 0

    def test_rolling_information_ratio_length(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        rir = analytics.rolling_information_ratio(window=63)
        assert len(rir) == len(port)

    def test_rolling_beta_length(self, returns_pair):
        port, bench = returns_pair
        analytics = BenchmarkAnalytics(port, bench)
        rb = analytics.rolling_beta(window=63)
        assert len(rb) == len(port)

    def test_insufficient_data_raises(self):
        short = pd.Series([0.01, -0.02, 0.01])
        with pytest.raises(ValueError, match="Insufficient"):
            BenchmarkAnalytics(short, short)


class TestActiveShare:
    def test_zero_active_share_identical(self):
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        assert ActiveShare.compute(w, w) == pytest.approx(0.0)

    def test_max_active_share_no_overlap(self):
        port = pd.Series({"A": 0.6, "B": 0.4})
        bench = pd.Series({"C": 0.7, "D": 0.3})
        assert ActiveShare.compute(port, bench) == pytest.approx(1.0)

    def test_active_share_between_zero_one(self):
        port = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
        bench = pd.Series({"A": 0.3, "B": 0.4, "D": 0.3})
        result = ActiveShare.compute(port, bench)
        assert 0 <= result <= 1

    def test_categorize_concentrated(self):
        assert ActiveShare.categorize(0.95) == "concentrated_active"

    def test_categorize_moderately_active(self):
        assert ActiveShare.categorize(0.75) == "moderately_active"

    def test_categorize_closet_indexer(self):
        assert ActiveShare.categorize(0.40) == "closet_indexer"

    def test_categorize_index_fund(self):
        assert ActiveShare.categorize(0.05) == "index_fund"


class TestBrinsonAttribution:
    def setup_method(self):
        self.pw = pd.Series({"Tech": 0.40, "Finance": 0.30, "Energy": 0.30})
        self.bw = pd.Series({"Tech": 0.30, "Finance": 0.35, "Energy": 0.35})
        self.pr = pd.Series({"Tech": 0.05, "Finance": 0.02, "Energy": -0.01})
        self.br = pd.Series({"Tech": 0.04, "Finance": 0.025, "Energy": -0.005})

    def test_attribution_sums_match_total(self):
        attr = BrinsonAttribution(self.pw, self.bw, self.pr, self.br)
        result = attr.compute()
        total = result.total_allocation + result.total_selection + result.total_interaction
        assert abs(total - result.total_active_return) < 1e-10

    def test_allocation_effect_series(self):
        attr = BrinsonAttribution(self.pw, self.bw, self.pr, self.br)
        result = attr.compute()
        assert isinstance(result.allocation_effect, pd.Series)
        assert len(result.allocation_effect) == 3

    def test_selection_effect_series(self):
        attr = BrinsonAttribution(self.pw, self.bw, self.pr, self.br)
        result = attr.compute()
        assert isinstance(result.selection_effect, pd.Series)

    def test_interaction_effect_series(self):
        attr = BrinsonAttribution(self.pw, self.bw, self.pr, self.br)
        result = attr.compute()
        assert isinstance(result.interaction_effect, pd.Series)
