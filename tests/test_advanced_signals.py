import pytest
import numpy as np
import pandas as pd
from meridianalgo.quant.advanced_signals import (
    hurst_exponent, 
    fractional_difference, 
    calculate_z_score, 
    get_half_life,
    information_coefficient
)

class TestAdvancedSignals:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        # Random walk
        rw = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        # Mean reverting series
        mr = pd.Series(100 + 0.5 * np.random.randn(n))
        # Trending series
        tr = pd.Series(np.linspace(100, 150, n) + np.random.randn(n))
        return {'rw': rw, 'mr': mr, 'tr': tr}

    def test_hurst_exponent(self, sample_data):
        h_rw = hurst_exponent(sample_data['rw'])
        h_mr = hurst_exponent(sample_data['mr'])
        h_tr = hurst_exponent(sample_data['tr'])
        
        # Random walk should be around 0.5
        assert 0.3 < h_rw < 0.7
        # Mean reverting should be less than random walk
        assert h_mr < h_rw
        # Trending should be greater than random walk (or at least > 0.4 for this simple polyfit)
        assert h_tr > 0.5 or h_tr > h_rw

    def test_fractional_difference(self, sample_data):
        diffed = fractional_difference(sample_data['rw'], d=0.5)
        assert isinstance(diffed, pd.Series)
        assert len(diffed) < len(sample_data['rw'])
        assert not diffed.isnull().any()

    def test_calculate_z_score(self, sample_data):
        z = calculate_z_score(sample_data['rw'], window=20)
        assert isinstance(z, pd.Series)
        assert z.std() < 2.0 # Roughly normalized

    def test_get_half_life(self, sample_data):
        hl_mr = get_half_life(sample_data['mr'])
        hl_tr = get_half_life(sample_data['tr'])
        
        assert hl_mr < 100
        assert hl_tr == np.inf

    def test_information_coefficient(self, sample_data):
        ic = information_coefficient(sample_data['rw'], sample_data['rw'])
        assert pytest.approx(ic) == 1.0
