"""
Tests for core MeridianAlgo functionality.
"""

import numpy as np
import pandas as pd
import pytest

from meridianalgo.core import (
    PortfolioOptimizer,
    StatisticalArbitrage,
    TimeSeriesAnalyzer,
    calculate_calmar_ratio,
    calculate_correlation_matrix,
    calculate_expected_shortfall,
    calculate_half_life,
    calculate_hurst_exponent,
    calculate_max_drawdown,
    calculate_metrics,
    calculate_rolling_correlation,
    calculate_sortino_ratio,
    calculate_value_at_risk,
    get_market_data,
)


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (252, 3)),
        index=dates,
        columns=["AAPL", "MSFT", "GOOGL"],
    )
    return returns


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    prices = pd.DataFrame(
        100 * np.cumprod(1 + np.random.normal(0.001, 0.02, (252, 3)), axis=0),
        index=dates,
        columns=["AAPL", "MSFT", "GOOGL"],
    )
    return prices


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer class."""

    def test_initialization(self, sample_returns):
        """Test PortfolioOptimizer initialization."""
        optimizer = PortfolioOptimizer(sample_returns)
        assert optimizer.returns.equals(sample_returns)
        assert isinstance(optimizer.cov_matrix, pd.DataFrame)
        assert optimizer.cov_matrix.shape == (3, 3)

    def test_calculate_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation."""
        optimizer = PortfolioOptimizer(sample_returns)
        result = optimizer.calculate_efficient_frontier(num_portfolios=10)

        assert "volatility" in result
        assert "returns" in result
        assert "sharpe" in result
        assert "weights" in result

        assert len(result["volatility"]) == 10
        assert len(result["returns"]) == 10
        assert len(result["sharpe"]) == 10
        assert result["weights"].shape == (10, 3)

    def test_optimize_portfolio(self, sample_returns):
        """Test portfolio optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        result = optimizer.optimize_portfolio()

        assert "weights" in result
        assert "return" in result
        assert "volatility" in result
        assert "sharpe" in result

        # Check weights sum to 1
        assert np.isclose(sum(result["weights"]), 1.0, atol=0.01)

        # Check all weights are non-negative
        assert all(w >= 0 for w in result["weights"])


class TestTimeSeriesAnalyzer:
    """Test TimeSeriesAnalyzer class."""

    def test_initialization(self, sample_prices):
        """Test TimeSeriesAnalyzer initialization."""
        analyzer = TimeSeriesAnalyzer(sample_prices)
        assert analyzer.data.equals(sample_prices)

    def test_calculate_returns(self, sample_prices):
        """Test returns calculation."""
        analyzer = TimeSeriesAnalyzer(sample_prices["AAPL"])
        returns = analyzer.calculate_returns()

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(sample_prices) - 1
        assert returns.isna().sum() == 0

    def test_calculate_log_returns(self, sample_prices):
        """Test log returns calculation."""
        analyzer = TimeSeriesAnalyzer(sample_prices["AAPL"])
        log_returns = analyzer.calculate_returns(log_returns=True)

        assert isinstance(log_returns, pd.Series)
        assert len(log_returns) == len(sample_prices) - 1

    def test_calculate_volatility(self, sample_prices):
        """Test volatility calculation."""
        analyzer = TimeSeriesAnalyzer(sample_prices["AAPL"])
        vol = analyzer.calculate_volatility(window=21)

        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_prices) - 1

    def test_calculate_moving_average(self, sample_prices):
        """Test moving average calculation."""
        analyzer = TimeSeriesAnalyzer(sample_prices["AAPL"])
        sma = analyzer.calculate_moving_average(window=20, ma_type="sma")

        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_prices)

        ema = analyzer.calculate_moving_average(window=20, ma_type="ema")
        assert isinstance(ema, pd.Series)

    def test_calculate_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands calculation."""
        analyzer = TimeSeriesAnalyzer(sample_prices["AAPL"])
        bb = analyzer.calculate_bollinger_bands(window=20, num_std=2)

        assert "middle" in bb
        assert "upper" in bb
        assert "lower" in bb

        assert isinstance(bb["middle"], pd.Series)
        assert isinstance(bb["upper"], pd.Series)
        assert isinstance(bb["lower"], pd.Series)


class TestRiskMetrics:
    """Test risk metric functions."""

    def test_calculate_metrics(self, sample_returns):
        """Test comprehensive metrics calculation."""
        returns = sample_returns["AAPL"]
        metrics = calculate_metrics(returns)

        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "annualized_volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics

    def test_calculate_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        returns = sample_returns["AAPL"]
        max_dd = calculate_max_drawdown(returns)

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero

    def test_calculate_value_at_risk(self, sample_returns):
        """Test VaR calculation."""
        returns = sample_returns["AAPL"]
        var = calculate_value_at_risk(returns)

        assert isinstance(var, float)
        assert var < 0

    def test_calculate_expected_shortfall(self, sample_returns):
        """Test Expected Shortfall calculation."""
        returns = sample_returns["AAPL"]
        es = calculate_expected_shortfall(returns, confidence_level=0.95)

        assert isinstance(es, float)
        assert es < 0  # ES should be negative

    def test_calculate_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        returns = sample_returns["AAPL"]
        sortino = calculate_sortino_ratio(returns)

        assert isinstance(sortino, float)

    def test_calculate_calmar_ratio(self, sample_returns):
        """Test Calmar ratio calculation."""
        returns = sample_returns["AAPL"]
        calmar = calculate_calmar_ratio(returns)

        assert isinstance(calmar, float)


class TestStatisticalArbitrage:
    """Test StatisticalArbitrage class."""

    def test_initialization(self, sample_prices):
        """Test StatisticalArbitrage initialization."""
        stat_arb = StatisticalArbitrage(sample_prices)
        assert stat_arb.data.equals(sample_prices)

    def test_calculate_zscore(self, sample_prices):
        """Test z-score calculation."""
        stat_arb = StatisticalArbitrage(sample_prices)
        zscore = stat_arb.calculate_zscore(window=20)

        assert isinstance(zscore, pd.DataFrame)
        assert zscore.shape == sample_prices.shape

    def test_calculate_cointegration(self, sample_prices):
        """Test cointegration calculation."""
        stat_arb = StatisticalArbitrage(sample_prices)
        x = sample_prices["AAPL"]
        y = sample_prices["MSFT"]

        result = stat_arb.calculate_cointegration(x, y)

        assert "score" in result
        assert "pvalue" in result
        assert "is_cointegrated" in result

        assert isinstance(result["score"], float)
        assert isinstance(result["pvalue"], float)
        assert isinstance(result["is_cointegrated"], bool)


class TestStatisticalFunctions:
    """Test statistical analysis functions."""

    def test_calculate_correlation_matrix(self, sample_returns):
        """Test correlation matrix calculation."""
        corr = calculate_correlation_matrix(sample_returns)

        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)
        assert np.allclose(np.diag(corr), 1.0)  # Diagonal should be 1

    def test_calculate_rolling_correlation(self, sample_returns):
        """Test rolling correlation calculation."""
        rolling_corr = calculate_rolling_correlation(sample_returns, window=21)

        assert isinstance(rolling_corr, pd.DataFrame)

    def test_calculate_hurst_exponent(self, sample_prices):
        """Test Hurst exponent calculation."""
        hurst = calculate_hurst_exponent(sample_prices["AAPL"])

        assert isinstance(hurst, float)
        assert 0 <= hurst <= 1  # Hurst exponent should be between 0 and 1

    def test_calculate_half_life(self, sample_prices):
        """Test half-life calculation."""
        half_life = calculate_half_life(sample_prices["AAPL"])

        assert isinstance(half_life, float)
        assert half_life > 0  # Half-life should be positive


class TestMarketData:
    """Test market data functions."""

    @pytest.mark.skip(reason="Requires internet connection")
    def test_get_market_data(self):
        """Test market data fetching."""
        # This test is skipped by default as it requires internet
        data = get_market_data(["AAPL"], "2020-01-01", "2020-01-31")

        assert isinstance(data, pd.DataFrame)
        assert "AAPL" in data.columns
        assert len(data) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        empty_returns = pd.DataFrame()

        with pytest.raises(ValueError):
            PortfolioOptimizer(empty_returns)

    def test_single_asset(self):
        """Test with single asset."""
        single_asset = pd.DataFrame({"AAPL": np.random.normal(0.001, 0.02, 100)})

        optimizer = PortfolioOptimizer(single_asset)
        result = optimizer.optimize_portfolio()

        assert result["weights"][0] == 1.0

    def test_constant_returns(self):
        """Test with constant returns."""
        constant_returns = pd.DataFrame({"AAPL": [0.01] * 100, "MSFT": [0.005] * 100})

        optimizer = PortfolioOptimizer(constant_returns)
        result = optimizer.optimize_portfolio()

        # Should still work but with zero volatility
        assert "weights" in result


if __name__ == "__main__":
    pytest.main([__file__])
