import numpy as np
import pandas as pd
import pytest

from meridianalgo.quant.advanced_signals import (
    calculate_z_score,
    fractional_difference,
    get_half_life,
    hurst_exponent,
    information_coefficient,
)


class TestAdvancedSignals:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 1000  # More data for stability
        # Random walk
        rw = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        # Mean reverting series (OU process like)
        mr = [100]
        for i in range(n - 1):
            mr.append(mr[-1] + 0.1 * (100 - mr[-1]) + np.random.randn())
        mr = pd.Series(mr)

        # Trending series
        tr = pd.Series(np.linspace(100, 200, n) + np.random.randn(n) * 2)
        return {"rw": rw, "mr": mr, "tr": tr}

    def test_hurst_exponent(self, sample_data):
        h_rw = hurst_exponent(sample_data["rw"])
        h_mr = hurst_exponent(sample_data["mr"])

        # Check they return valid numbers
        assert isinstance(h_rw, (float, np.float64, np.float32))
        assert isinstance(h_mr, (float, np.float64, np.float32))
        # Mean reverting should be fairly low
        assert h_mr < 0.5

    def test_fractional_difference(self, sample_data):
        diffed = fractional_difference(sample_data["rw"], d=0.5)
        assert isinstance(diffed, pd.Series)
        assert len(diffed) > 0

    def test_calculate_z_score(self, sample_data):
        z = calculate_z_score(sample_data["rw"], window=20)
        assert isinstance(z, pd.Series)

    def test_get_half_life(self, sample_data):
        hl_mr = get_half_life(sample_data["mr"])
        assert hl_mr > 0
        assert hl_mr < 100  # Should revert quickly with 0.1 theta

    def test_information_coefficient(self, sample_data):
        ic = information_coefficient(sample_data["rw"], sample_data["rw"])
        assert ic > 0.99
