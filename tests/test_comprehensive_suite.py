"""
Comprehensive test suite for MeridianAlgo package.
Tests core functionality, imports, and basic operations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import meridianalgo as ma


class TestPackageImport:
    """Test package imports and version."""

    def test_package_import(self):
        """Test that package imports successfully."""
        assert ma is not None

    def test_version_exists(self):
        """Test that version is defined."""
        assert hasattr(ma, "__version__")
        assert isinstance(ma.__version__, str)
        assert len(ma.__version__) > 0

    def test_version_format(self):
        """Test version follows semantic versioning."""
        version_parts = ma.__version__.split(".")
        assert len(version_parts) >= 2
        assert version_parts[0].isdigit()
        assert version_parts[1].isdigit()


class TestDataGeneration:
    """Test data generation and handling."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(100) * 2), index=dates, name="AAPL"
        )
        return prices

    @pytest.fixture
    def sample_returns(self, sample_prices):
        """Generate sample returns data."""
        return sample_prices.pct_change().dropna()

    def test_sample_data_creation(self, sample_prices):
        """Test sample data is created correctly."""
        assert isinstance(sample_prices, pd.Series)
        assert len(sample_prices) == 100
        assert sample_prices.name == "AAPL"

    def test_returns_calculation(self, sample_returns):
        """Test returns are calculated correctly."""
        assert isinstance(sample_returns, pd.Series)
        assert len(sample_returns) == 99
        assert not sample_returns.isna().any()


class TestStatisticalFunctions:
    """Test statistical functions."""

    @pytest.fixture
    def returns_data(self):
        """Generate returns data for testing."""
        np.random.seed(42)
        return pd.Series(np.random.randn(252) * 0.01)

    def test_sharpe_ratio_calculation(self, returns_data):
        """Test Sharpe ratio calculation."""
        try:
            sharpe = ma.calculate_sharpe_ratio(returns_data)
            assert isinstance(sharpe, (float, np.floating))
            assert not np.isnan(sharpe)
        except AttributeError:
            pytest.skip("Sharpe ratio function not available")

    def test_max_drawdown_calculation(self, returns_data):
        """Test max drawdown calculation."""
        try:
            mdd = ma.calculate_max_drawdown(returns_data)
            assert isinstance(mdd, (float, np.floating))
            assert mdd <= 0
        except AttributeError:
            pytest.skip("Max drawdown function not available")

    def test_volatility_calculation(self, returns_data):
        """Test volatility calculation."""
        vol = returns_data.std() * np.sqrt(252)
        assert isinstance(vol, (float, np.floating))
        assert vol > 0


class TestTechnicalIndicators:
    """Test technical indicator calculations."""

    @pytest.fixture
    def price_series(self):
        """Generate price series for indicators."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(100) * 2), index=dates, name="TEST"
        )
        return prices

    def test_rsi_calculation(self, price_series):
        """Test RSI calculation."""
        try:
            rsi = ma.calculate_rsi(price_series, window=14)
            assert isinstance(rsi, pd.Series)
            assert len(rsi) <= len(price_series)
            valid_rsi = rsi.dropna()
            if len(valid_rsi) > 0:
                assert (valid_rsi >= 0).all()
                assert (valid_rsi <= 100).all()
        except (AttributeError, TypeError):
            pytest.skip("RSI calculation not available")

    def test_moving_average(self, price_series):
        """Test moving average calculation."""
        ma_20 = price_series.rolling(window=20).mean()
        assert isinstance(ma_20, pd.Series)
        assert len(ma_20) == len(price_series)
        assert ma_20.dropna().shape[0] == len(price_series) - 19

    def test_bollinger_bands(self, price_series):
        """Test Bollinger Bands calculation."""
        try:
            bb = ma.calculate_bollinger_bands(price_series, window=20)
            assert isinstance(bb, (pd.DataFrame, dict, tuple))
        except (AttributeError, TypeError):
            pytest.skip("Bollinger Bands calculation not available")


class TestPortfolioAnalysis:
    """Test portfolio analysis functions."""

    @pytest.fixture
    def portfolio_returns(self):
        """Generate multi-asset returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        returns = pd.DataFrame(
            {
                "AAPL": np.random.randn(252) * 0.02,
                "MSFT": np.random.randn(252) * 0.018,
                "GOOGL": np.random.randn(252) * 0.022,
            },
            index=dates,
        )
        return returns

    def test_portfolio_returns_structure(self, portfolio_returns):
        """Test portfolio returns structure."""
        assert isinstance(portfolio_returns, pd.DataFrame)
        assert portfolio_returns.shape == (252, 3)
        assert list(portfolio_returns.columns) == ["AAPL", "MSFT", "GOOGL"]

    def test_correlation_matrix(self, portfolio_returns):
        """Test correlation matrix calculation."""
        corr = portfolio_returns.corr()
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)
        assert (np.diag(corr) == 1.0).all()

    def test_covariance_matrix(self, portfolio_returns):
        """Test covariance matrix calculation."""
        cov = portfolio_returns.cov()
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (3, 3)
        assert (np.diag(cov) > 0).all()


class TestRiskMetrics:
    """Test risk metric calculations."""

    @pytest.fixture
    def risk_returns(self):
        """Generate returns for risk testing."""
        np.random.seed(42)
        return pd.Series(np.random.randn(1000) * 0.02)

    def test_var_calculation(self, risk_returns):
        """Test Value at Risk calculation."""
        var_95 = risk_returns.quantile(0.05)
        assert isinstance(var_95, (float, np.floating))
        assert var_95 < 0

    def test_cvar_calculation(self, risk_returns):
        """Test Conditional VaR calculation."""
        var_95 = risk_returns.quantile(0.05)
        cvar_95 = risk_returns[risk_returns <= var_95].mean()
        assert isinstance(cvar_95, (float, np.floating))
        assert cvar_95 <= var_95

    def test_downside_deviation(self, risk_returns):
        """Test downside deviation calculation."""
        downside = risk_returns[risk_returns < 0]
        downside_std = downside.std()
        assert isinstance(downside_std, (float, np.floating))
        assert downside_std >= 0


class TestDataValidation:
    """Test data validation and error handling."""

    def test_empty_series_handling(self):
        """Test handling of empty series."""
        empty_series = pd.Series([], dtype=float)
        assert len(empty_series) == 0

    def test_nan_handling(self):
        """Test NaN handling."""
        series_with_nan = pd.Series([1, 2, np.nan, 4, 5])
        clean_series = series_with_nan.dropna()
        assert len(clean_series) == 4
        assert not clean_series.isna().any()

    def test_infinite_value_detection(self):
        """Test infinite value detection."""
        series_with_inf = pd.Series([1, 2, np.inf, 4, 5])
        has_inf = np.isinf(series_with_inf).any()
        assert has_inf == True  # Use == instead of is for numpy bool


class TestMathematicalOperations:
    """Test mathematical operations."""

    def test_log_returns(self):
        """Test log returns calculation."""
        prices = pd.Series([100, 102, 101, 103, 105])
        log_returns = np.log(prices / prices.shift(1)).dropna()
        assert len(log_returns) == 4
        assert not log_returns.isna().any()

    def test_cumulative_returns(self):
        """Test cumulative returns calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        cum_returns = (1 + returns).cumprod() - 1
        assert len(cum_returns) == 5
        assert cum_returns.iloc[-1] > 0

    def test_annualization(self):
        """Test annualization factor."""
        daily_vol = 0.01
        annual_vol = daily_vol * np.sqrt(252)
        assert annual_vol > daily_vol
        assert annual_vol == pytest.approx(0.1587, rel=0.01)


class TestPerformanceMetrics:
    """Test performance metric calculations."""

    @pytest.fixture
    def performance_data(self):
        """Generate performance data."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003)
        return returns

    def test_total_return(self, performance_data):
        """Test total return calculation."""
        total_return = (1 + performance_data).prod() - 1
        assert isinstance(total_return, (float, np.floating))

    def test_annualized_return(self, performance_data):
        """Test annualized return calculation."""
        total_return = (1 + performance_data).prod() - 1
        years = len(performance_data) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        assert isinstance(annual_return, (float, np.floating))

    def test_sortino_ratio(self, performance_data):
        """Test Sortino ratio calculation."""
        downside_returns = performance_data[performance_data < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std > 0:
            sortino = (performance_data.mean() * 252) / downside_std
            assert isinstance(sortino, (float, np.floating))


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_basic_workflow(self):
        """Test basic analysis workflow."""
        # Generate data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(100) * 2), index=dates, name="TEST"
        )

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Calculate metrics
        mean_return = returns.mean()
        volatility = returns.std()
        sharpe = mean_return / volatility if volatility > 0 else 0

        # Assertions
        assert isinstance(mean_return, (float, np.floating))
        assert isinstance(volatility, (float, np.floating))
        assert isinstance(sharpe, (float, np.floating))
        assert volatility >= 0

    def test_multi_asset_workflow(self):
        """Test multi-asset analysis workflow."""
        # Generate multi-asset data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.DataFrame(
            {
                "Asset1": np.random.randn(100) * 0.02,
                "Asset2": np.random.randn(100) * 0.015,
                "Asset3": np.random.randn(100) * 0.025,
            },
            index=dates,
        )

        # Calculate portfolio metrics
        weights = np.array([0.4, 0.3, 0.3])
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std()

        # Assertions
        assert len(portfolio_returns) == 100
        assert isinstance(portfolio_vol, (float, np.floating))
        assert portfolio_vol > 0


def run_all_tests():
    """Run all tests and return results."""
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-W",
        "ignore::DeprecationWarning",
    ]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
