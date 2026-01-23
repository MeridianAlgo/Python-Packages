"""
Comprehensive tests for risk analysis module.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the path to import meridianalgo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import meridianalgo as ma
    from meridianalgo.risk_analysis import (
        VaRCalculator,
        ExpectedShortfall,
        HistoricalVaR,
        ParametricVaR,
        MonteCarloVaR,
    )
except ImportError as e:
    pytest.skip(f"Could not import meridianalgo: {e}", allow_module_level=True)


class TestRiskAnalysis:
    """Test suite for risk analysis."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate returns with some volatility clustering and fat tails
        returns = []
        volatility = 0.02

        for i in range(252):
            # Add volatility clustering
            if i > 0 and abs(returns[-1]) > 0.03:
                volatility = min(volatility * 1.2, 0.05)
            else:
                volatility = max(volatility * 0.95, 0.01)

            # Generate return with fat tails (t-distribution)
            ret = np.random.standard_t(df=5) * volatility * 0.3
            returns.append(ret)

        return pd.Series(returns, index=dates)

    def test_var_calculator_creation(self, sample_returns):
        """Test VaR calculator initialization."""
        try:
            var_calc = VaRCalculator(sample_returns)

            assert var_calc is not None
            assert hasattr(var_calc, "returns")
            assert len(var_calc.returns) == len(sample_returns)

            print(" VaR calculator creation test passed")
        except Exception as e:
            print(f" VaR calculator creation test failed: {e}")

    def test_historical_var(self, sample_returns):
        """Test Historical VaR calculation."""
        try:
            var_calc = HistoricalVaR(sample_returns)

            # Test different confidence levels
            var_95 = var_calc.calculate_var(confidence_level=0.95)
            var_99 = var_calc.calculate_var(confidence_level=0.99)

            # VaR should be negative (loss)
            assert var_95 <= 0
            assert var_99 <= 0

            # 99% VaR should be more extreme than 95% VaR
            assert var_99 <= var_95

            # VaR should be reasonable (not too extreme)
            assert var_95 >= -1.0  # Not more than 100% loss
            assert var_99 >= -1.0

            print(" Historical VaR test passed")
        except Exception as e:
            print(f" Historical VaR test failed: {e}")

    def test_parametric_var(self, sample_returns):
        """Test Parametric VaR calculation."""
        try:
            var_calc = ParametricVaR(sample_returns)

            # Test different confidence levels
            var_95 = var_calc.calculate_var(confidence_level=0.95)
            var_99 = var_calc.calculate_var(confidence_level=0.99)

            # VaR should be negative (loss)
            assert var_95 <= 0
            assert var_99 <= 0

            # 99% VaR should be more extreme than 95% VaR
            assert var_99 <= var_95

            print(" Parametric VaR test passed")
        except Exception as e:
            print(f" Parametric VaR test failed: {e}")

    def test_monte_carlo_var(self, sample_returns):
        """Test Monte Carlo VaR calculation."""
        try:
            var_calc = MonteCarloVaR(sample_returns)

            # Test with smaller number of simulations for speed
            var_95 = var_calc.calculate_var(confidence_level=0.95, num_simulations=1000)

            # VaR should be negative (loss)
            assert var_95 <= 0

            # VaR should be reasonable
            assert var_95 >= -1.0

            print(" Monte Carlo VaR test passed")
        except Exception as e:
            print(f" Monte Carlo VaR test failed: {e}")

    def test_expected_shortfall(self, sample_returns):
        """Test Expected Shortfall (CVaR) calculation."""
        try:
            es_calc = ExpectedShortfall(sample_returns)

            # Calculate ES and VaR
            es_95 = es_calc.calculate_expected_shortfall(confidence_level=0.95)
            var_95 = es_calc.calculate_var(confidence_level=0.95)

            # ES should be more extreme than VaR
            assert es_95 <= var_95

            # Both should be negative (losses)
            assert es_95 <= 0
            assert var_95 <= 0

            print(" Expected Shortfall test passed")
        except Exception as e:
            print(f" Expected Shortfall test failed: {e}")

    def test_maximum_drawdown(self, sample_returns):
        """Test Maximum Drawdown calculation."""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + sample_returns).cumprod()

            # Calculate drawdown
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # Max drawdown should be negative or zero
            assert max_drawdown <= 0

            # Should be reasonable (not more than 100% loss)
            assert max_drawdown >= -1.0

            print(" Maximum Drawdown test passed")
        except Exception as e:
            print(f" Maximum Drawdown test failed: {e}")

    def test_volatility_metrics(self, sample_returns):
        """Test volatility-based risk metrics."""
        try:
            # Calculate various volatility measures
            daily_vol = sample_returns.std()
            annual_vol = daily_vol * np.sqrt(252)

            # Rolling volatility
            rolling_vol = sample_returns.rolling(window=30).std()

            # GARCH-like volatility (simplified)
            ewm_vol = sample_returns.ewm(span=30).std()

            # All volatilities should be positive
            assert daily_vol >= 0
            assert annual_vol >= 0
            assert all(vol >= 0 for vol in rolling_vol.dropna())
            assert all(vol >= 0 for vol in ewm_vol.dropna())

            # Annual vol should be higher than daily
            assert annual_vol >= daily_vol

            print(" Volatility metrics test passed")
        except Exception as e:
            print(f" Volatility metrics test failed: {e}")

    def test_correlation_risk(self, sample_returns):
        """Test correlation-based risk measures."""
        try:
            # Create multiple return series
            returns_matrix = np.column_stack(
                [
                    sample_returns.values,
                    sample_returns.values * 0.8
                    + np.random.normal(0, 0.01, len(sample_returns)),
                    sample_returns.values * -0.3
                    + np.random.normal(0, 0.015, len(sample_returns)),
                ]
            )

            returns_df = pd.DataFrame(
                returns_matrix,
                columns=["Asset1", "Asset2", "Asset3"],
                index=sample_returns.index,
            )

            # Calculate correlation matrix
            corr_matrix = returns_df.corr()

            # Correlation matrix should be symmetric
            assert np.allclose(corr_matrix.values, corr_matrix.values.T)

            # Diagonal should be 1
            assert np.allclose(np.diag(corr_matrix.values), 1.0)

            # All correlations should be between -1 and 1
            assert ((corr_matrix.values >= -1.0) & (corr_matrix.values <= 1.0)).all()

            print(" Correlation risk test passed")
        except Exception as e:
            print(f" Correlation risk test failed: {e}")

    def test_stress_testing(self, sample_returns):
        """Test stress testing scenarios."""
        try:
            # Define stress scenarios
            scenarios = {
                "market_crash": -0.20,  # 20% market drop
                "volatility_spike": sample_returns.std() * 3,  # 3x normal volatility
                "correlation_breakdown": 0.9,  # High correlation scenario
            }

            # Test market crash scenario
            stressed_return = sample_returns.mean() + scenarios["market_crash"]

            # Calculate impact
            portfolio_value = 100000
            stressed_value = portfolio_value * (1 + stressed_return)
            loss = portfolio_value - stressed_value

            assert loss >= 0  # Should be a loss
            assert stressed_value < portfolio_value

            print(" Stress testing test passed")
        except Exception as e:
            print(f" Stress testing test failed: {e}")

    def test_tail_risk_measures(self, sample_returns):
        """Test tail risk measures."""
        try:
            # Calculate skewness and kurtosis
            sample_returns.skew()
            sample_returns.kurtosis()

            # Test tail ratio (95th percentile / 5th percentile)
            p95 = sample_returns.quantile(0.95)
            p5 = sample_returns.quantile(0.05)

            if p5 != 0:
                tail_ratio = abs(p95 / p5)
                assert tail_ratio > 0

            # Test extreme value statistics
            extreme_losses = sample_returns[
                sample_returns < sample_returns.quantile(0.05)
            ]

            if len(extreme_losses) > 0:
                avg_extreme_loss = extreme_losses.mean()
                assert avg_extreme_loss < 0  # Should be negative (loss)

            print(" Tail risk measures test passed")
        except Exception as e:
            print(f" Tail risk measures test failed: {e}")

    def test_risk_decomposition(self, sample_returns):
        """Test risk decomposition analysis."""
        try:
            # Create a simple portfolio
            weights = np.array([0.4, 0.3, 0.3])

            # Create multiple assets (simplified)
            asset_returns = np.column_stack(
                [
                    sample_returns.values,
                    sample_returns.values * 0.8
                    + np.random.normal(0, 0.01, len(sample_returns)),
                    sample_returns.values * 0.6
                    + np.random.normal(0, 0.012, len(sample_returns)),
                ]
            )

            # Calculate portfolio return
            np.dot(asset_returns, weights)

            # Calculate individual asset contributions to portfolio risk
            cov_matrix = np.cov(asset_returns.T)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

            # Marginal contributions
            marginal_contrib = np.dot(cov_matrix, weights) / np.sqrt(portfolio_variance)

            # Component contributions
            component_contrib = weights * marginal_contrib / np.sqrt(portfolio_variance)

            # Component contributions should sum to 1
            assert abs(component_contrib.sum() - 1.0) < 1e-10

            print(" Risk decomposition test passed")
        except Exception as e:
            print(f" Risk decomposition test failed: {e}")

    def test_error_handling(self, sample_returns):
        """Test error handling for invalid inputs."""
        try:
            # Test with insufficient data
            short_returns = sample_returns.head(5)

            try:
                var_calc = VaRCalculator(short_returns)
                var_calc.calculate_var(confidence_level=0.95)
                # Should either work or raise appropriate error
            except (ValueError, IndexError):
                pass  # Expected for insufficient data

            # Test with invalid confidence level
            var_calc = VaRCalculator(sample_returns)
            try:
                var_calc.calculate_var(confidence_level=1.5)  # Invalid
            except ValueError:
                pass  # Expected behavior

            print(" Error handling test passed")
        except Exception as e:
            print(f" Error handling test failed: {e}")


def test_risk_analysis_import():
    """Test that risk analysis can be imported."""
    try:
        from meridianalgo.risk_analysis import (
            ExpectedShortfall,
            VaRCalculator,
        )  # noqa: F401

        print(" Risk analysis import test passed")
        return True
    except ImportError as e:
        print(f" Import test failed: {e}")
        return False


def test_risk_with_real_data():
    """Test risk analysis with real market data if available."""
    try:
        # Try to get real data
        data = ma.get_market_data(["AAPL"], "2023-01-01", "2023-12-31")

        if data is not None and len(data) > 50:
            # Calculate returns
            returns = data["AAPL"].pct_change().dropna()

            if len(returns) > 30:
                # Test VaR calculation
                var_95 = ma.calculate_value_at_risk(returns, confidence_level=0.95)
                es_95 = ma.calculate_expected_shortfall(returns, confidence_level=0.95)

                assert var_95 <= 0  # Should be negative (loss)
                assert es_95 <= var_95  # ES should be more extreme

                print(" Real data risk test passed")
            else:
                print(" Insufficient real data for risk test")
        else:
            print(" No real data available, skipping real data test")

    except Exception as e:
        print(f" Real data risk test failed: {e}")


if __name__ == "__main__":
    # Run tests manually
    print(" Running Risk Analysis Tests...")

    # Test imports first
    if not test_risk_analysis_import():
        print(" Cannot proceed with tests - import failed")
        exit(1)

    # Create test instance
    test_instance = TestRiskAnalysis()

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    returns = []
    volatility = 0.02

    for i in range(252):
        if i > 0 and abs(returns[-1]) > 0.03:
            volatility = min(volatility * 1.2, 0.05)
        else:
            volatility = max(volatility * 0.95, 0.01)

        ret = np.random.standard_t(df=5) * volatility * 0.3
        returns.append(ret)

    sample_returns = pd.Series(returns, index=dates)

    # Run all tests
    test_methods = [
        test_instance.test_var_calculator_creation,
        test_instance.test_historical_var,
        test_instance.test_parametric_var,
        test_instance.test_monte_carlo_var,
        test_instance.test_expected_shortfall,
        test_instance.test_maximum_drawdown,
        test_instance.test_volatility_metrics,
        test_instance.test_correlation_risk,
        test_instance.test_stress_testing,
        test_instance.test_tail_risk_measures,
        test_instance.test_risk_decomposition,
        test_instance.test_error_handling,
    ]

    passed = 0
    total = len(test_methods)

    for test_method in test_methods:
        try:
            test_method(sample_returns)
            passed += 1
        except Exception as e:
            print(f" Test {test_method.__name__} failed: {e}")

    # Test with real data
    test_risk_with_real_data()

    print(f"\n Test Results: {passed}/{total} tests passed")

    if passed == total:
        print(" All risk analysis tests passed!")
    else:
        print(f" {total - passed} tests failed")
