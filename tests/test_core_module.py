"""
Unit tests for the core module functionality.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestCoreModule(unittest.TestCase):
    """Test cases for core module functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            index=self.dates,
            name="Close",
        )
        self.returns = self.prices.pct_change().dropna()

        # Create multi-asset returns for portfolio optimization
        self.returns_df = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 1000),
                "MSFT": np.random.normal(0.0008, 0.018, 1000),
                "GOOG": np.random.normal(0.0012, 0.022, 1000),
            }
        )

    def test_portfolio_optimizer_initialization(self):
        """Test PortfolioOptimizer initialization."""
        from meridianalgo import PortfolioOptimizer

        optimizer = PortfolioOptimizer(self.returns_df)
        self.assertIsInstance(optimizer, PortfolioOptimizer)
        self.assertEqual(optimizer.returns.shape, self.returns_df.shape)
        self.assertTrue(optimizer.cov_matrix is not None)

    def test_portfolio_optimizer_efficient_frontier(self):
        """Test efficient frontier calculation."""
        from meridianalgo import PortfolioOptimizer

        optimizer = PortfolioOptimizer(self.returns_df)
        frontier = optimizer.calculate_efficient_frontier(num_portfolios=100)

        self.assertIn("volatility", frontier)
        self.assertIn("returns", frontier)
        self.assertIn("sharpe", frontier)
        self.assertIn("weights", frontier)
        self.assertEqual(len(frontier["volatility"]), 100)

    def test_time_series_analyzer_initialization(self):
        """Test TimeSeriesAnalyzer initialization."""
        from meridianalgo import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.prices)
        self.assertIsInstance(analyzer, TimeSeriesAnalyzer)
        self.assertEqual(len(analyzer.data), len(self.prices))

    def test_time_series_analyzer_returns(self):
        """Test returns calculation."""
        from meridianalgo import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.prices)
        returns = analyzer.calculate_returns()

        self.assertEqual(len(returns), len(self.prices) - 1)
        self.assertTrue(returns.index.equals(self.prices.index[1:]))

    def test_time_series_analyzer_log_returns(self):
        """Test log returns calculation."""
        from meridianalgo import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.prices)
        log_returns = analyzer.calculate_returns(log_returns=True)

        self.assertEqual(len(log_returns), len(self.prices) - 1)
        # Log returns should be approximately equal to simple returns for small values
        simple_returns = analyzer.calculate_returns()
        np.testing.assert_allclose(log_returns, np.log(1 + simple_returns), rtol=1e-10)

    def test_time_series_analyzer_volatility(self):
        """Test volatility calculation."""
        from meridianalgo import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.prices)
        volatility = analyzer.calculate_volatility(window=21, annualized=True)

        # Rolling volatility should have NaN values for the first window-1 periods
        # Note: calculate_volatility calls calculate_returns() which drops NaN, so length is reduced by 1
        self.assertEqual(
            len(volatility), len(self.prices) - 1
        )  # Same length as returns
        self.assertEqual(
            volatility.notna().sum(), len(self.prices) - 1 - 20
        )  # 21-period window
        self.assertTrue(volatility.notna().sum() > 0)

    def test_get_market_data(self):
        """Test market data fetching."""
        from meridianalgo import get_market_data

        try:
            data = get_market_data(
                ["AAPL"], start_date="2023-01-01", end_date="2023-01-10"
            )
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            self.assertIn("AAPL", data.columns)
        except Exception as e:
            self.skipTest(f"Skipping market data test: {str(e)}")

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        from meridianalgo import calculate_metrics

        metrics = calculate_metrics(self.returns)
        required_metrics = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]

        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        from meridianalgo import calculate_max_drawdown

        max_dd = calculate_max_drawdown(self.returns)
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0)  # Drawdown should be negative or zero

    def test_calculate_metrics_with_risk_free_rate(self):
        """Test metrics calculation with risk-free rate."""
        from meridianalgo import calculate_metrics

        risk_free_rate = 0.02
        metrics = calculate_metrics(self.returns, risk_free_rate=risk_free_rate)

        self.assertIn("sharpe_ratio", metrics)
        # Sharpe ratio should be adjusted for risk-free rate
        self.assertIsInstance(metrics["sharpe_ratio"], (int, float))


if __name__ == "__main__":
    unittest.main()
