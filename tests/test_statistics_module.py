"""
Unit tests for the statistics module functionality.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestStatisticsModule(unittest.TestCase):
    """Test cases for statistics module functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dates = pd.date_range("2023-01-01", periods=1000, freq="D")

        # Create correlated time series for statistical arbitrage testing
        base_returns = np.random.normal(0.001, 0.02, 1000)
        self.data = pd.DataFrame(
            {
                "AAPL": np.cumprod(1 + base_returns),
                "MSFT": np.cumprod(
                    1 + base_returns * 0.8 + np.random.normal(0, 0.01, 1000)
                ),
                "GOOG": np.cumprod(
                    1 + base_returns * 0.6 + np.random.normal(0, 0.015, 1000)
                ),
            },
            index=self.dates,
        )

        self.returns = self.data.pct_change().dropna()

    def test_statistical_arbitrage_initialization(self):
        """Test StatisticalArbitrage initialization."""
        from meridianalgo import StatisticalArbitrage

        arb = StatisticalArbitrage(self.data)
        self.assertIsInstance(arb, StatisticalArbitrage)
        self.assertEqual(len(arb.data), len(self.data))

    def test_rolling_correlation(self):
        """Test rolling correlation calculation."""
        from meridianalgo import StatisticalArbitrage

        arb = StatisticalArbitrage(self.data)
        corr = arb.calculate_rolling_correlation(window=20)

        self.assertIsInstance(corr, pd.DataFrame)
        # Should have correlation between all pairs
        self.assertGreater(len(corr), 0)

    def test_cointegration_test(self):
        """Test cointegration testing."""
        from meridianalgo import StatisticalArbitrage

        arb = StatisticalArbitrage(self.data)

        # Test with two series
        x = self.data["AAPL"]
        y = self.data["MSFT"]

        try:
            result = arb.test_cointegration(x, y)
            self.assertIn("test_statistic", result)
            self.assertIn("p_value", result)
            self.assertIn("is_cointegrated", result)
            self.assertIsInstance(bool(result["is_cointegrated"]), bool)
        except ImportError:
            self.skipTest("statsmodels not available for cointegration testing")

    def test_value_at_risk(self):
        """Test Value at Risk calculation."""
        from meridianalgo import calculate_value_at_risk

        var_95 = calculate_value_at_risk(self.returns["AAPL"], confidence_level=0.95)
        var_99 = calculate_value_at_risk(self.returns["AAPL"], confidence_level=0.99)

        self.assertIsInstance(var_95, float)
        self.assertIsInstance(var_99, float)
        # 99% VaR should be more extreme (more negative) than 95% VaR
        self.assertLess(var_99, var_95)

    def test_value_at_risk_invalid_confidence(self):
        """Test VaR with invalid confidence level."""
        from meridianalgo import calculate_value_at_risk

        with self.assertRaises(ValueError):
            calculate_value_at_risk(self.returns["AAPL"], confidence_level=1.5)

        with self.assertRaises(ValueError):
            calculate_value_at_risk(self.returns["AAPL"], confidence_level=-0.1)

    def test_expected_shortfall(self):
        """Test Expected Shortfall calculation."""
        from meridianalgo import calculate_expected_shortfall, calculate_value_at_risk

        es_95 = calculate_expected_shortfall(
            self.returns["AAPL"], confidence_level=0.95
        )
        es_99 = calculate_expected_shortfall(
            self.returns["AAPL"], confidence_level=0.99
        )

        self.assertIsInstance(es_95, float)
        self.assertIsInstance(es_99, float)
        # ES should be more extreme than VaR
        var_95 = calculate_value_at_risk(self.returns["AAPL"], confidence_level=0.95)
        self.assertLess(es_95, var_95)

    def test_expected_shortfall_invalid_confidence(self):
        """Test ES with invalid confidence level."""
        from meridianalgo import calculate_expected_shortfall

        with self.assertRaises(ValueError):
            calculate_expected_shortfall(self.returns["AAPL"], confidence_level=1.5)

    def test_hurst_exponent(self):
        """Test Hurst exponent calculation."""
        from meridianalgo import hurst_exponent

        # Test with random walk (should be close to 0.5)
        random_walk = np.cumsum(np.random.randn(1000))
        hurst = hurst_exponent(random_walk)

        self.assertIsInstance(hurst, float)
        # For a random walk, Hurst exponent should be close to 0.5
        self.assertGreater(hurst, 0.3)
        self.assertLess(hurst, 0.7)

    def test_autocorrelation(self):
        """Test autocorrelation calculation."""
        from meridianalgo import calculate_autocorrelation

        # Test with AR(1) process
        ar1 = np.zeros(1000)
        ar1[0] = np.random.randn()
        for i in range(1, 1000):
            ar1[i] = 0.7 * ar1[i - 1] + np.random.randn()

        series = pd.Series(ar1)
        autocorr = calculate_autocorrelation(series, lag=1)

        self.assertIsInstance(autocorr, float)
        # AR(1) with coefficient 0.7 should have positive autocorrelation
        self.assertGreater(autocorr, 0)

    def test_rolling_volatility(self):
        """Test rolling volatility calculation."""
        from meridianalgo import rolling_volatility

        vol = rolling_volatility(self.returns["AAPL"], window=21, annualized=True)

        self.assertIsInstance(vol, pd.Series)
        # Rolling volatility should have NaN values for the first window-1 periods
        self.assertEqual(len(vol), len(self.returns))  # Same length as input
        self.assertEqual(vol.notna().sum(), len(self.returns) - 20)  # 21-period window
        self.assertTrue(vol.notna().sum() > 0)

        # Annualized volatility should be positive (excluding NaN values)
        self.assertTrue((vol.dropna() > 0).all())


if __name__ == "__main__":
    unittest.main()
