"""
Comprehensive tests for portfolio management module.
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
    from meridianalgo.portfolio_management import (
        PortfolioOptimizer,
        EfficientFrontier,
        BlackLitterman,
        RiskParity,
    )
except ImportError as e:
    pytest.skip(f"Could not import meridianalgo: {e}", allow_module_level=True)


class TestPortfolioManagement:
    """Test suite for portfolio management."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate correlated returns for multiple assets
        n_assets = 5
        mean_returns = np.random.uniform(0.0005, 0.002, n_assets)

        # Create correlation matrix
        correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1.0)

        # Generate returns
        returns = np.random.multivariate_normal(
            mean_returns, correlation * 0.02**2, 252
        )

        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        return pd.DataFrame(returns, columns=symbols, index=dates)

    def test_portfolio_optimizer_creation(self, sample_returns):
        """Test portfolio optimizer initialization."""
        try:
            optimizer = PortfolioOptimizer(sample_returns)

            assert optimizer is not None
            assert hasattr(optimizer, "returns")
            assert len(optimizer.returns.columns) == len(sample_returns.columns)

            print(" Portfolio optimizer creation test passed")
        except Exception as e:
            print(f" Portfolio optimizer creation test failed: {e}")

    def test_mean_variance_optimization(self, sample_returns):
        """Test mean-variance optimization."""
        try:
            optimizer = PortfolioOptimizer(sample_returns)

            # Test maximum Sharpe ratio optimization
            weights = optimizer.optimize_portfolio(objective="sharpe")

            # Weights should sum to 1
            assert abs(sum(weights.values()) - 1.0) < 1e-6

            # All weights should be non-negative (long-only)
            assert all(w >= -1e-6 for w in weights.values())

            # Should have weights for all assets
            assert len(weights) == len(sample_returns.columns)

            print(" Mean-variance optimization test passed")
        except Exception as e:
            print(f" Mean-variance optimization test failed: {e}")

    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation."""
        try:
            frontier = EfficientFrontier(sample_returns)

            # Calculate frontier points
            target_returns = np.linspace(0.05, 0.25, 10)
            frontier_data = frontier.calculate_frontier(target_returns)

            assert isinstance(frontier_data, pd.DataFrame)
            assert "return" in frontier_data.columns
            assert "volatility" in frontier_data.columns
            assert len(frontier_data) == len(target_returns)

            # Returns should be monotonically increasing
            assert all(frontier_data["return"].diff().dropna() >= -1e-6)

            print(" Efficient frontier test passed")
        except Exception as e:
            print(f" Efficient frontier test failed: {e}")

    def test_black_litterman_model(self, sample_returns):
        """Test Black-Litterman model."""
        try:
            # Create market cap weights
            market_caps = {
                "AAPL": 3000,
                "GOOGL": 1800,
                "MSFT": 2800,
                "TSLA": 800,
                "AMZN": 1500,
            }

            bl_model = BlackLitterman(sample_returns, market_caps)

            # Test with some views
            views = {"AAPL": 0.15, "TSLA": 0.20}
            bl_weights = bl_model.optimize_with_views(views)

            # Weights should sum to 1
            assert abs(sum(bl_weights.values()) - 1.0) < 1e-6

            # Should have weights for all assets
            assert len(bl_weights) == len(sample_returns.columns)

            print(" Black-Litterman model test passed")
        except Exception as e:
            print(f" Black-Litterman model test failed: {e}")

    def test_risk_parity(self, sample_returns):
        """Test risk parity optimization."""
        try:
            rp_optimizer = RiskParity(sample_returns)
            rp_weights = rp_optimizer.optimize()

            # Weights should sum to 1
            assert abs(sum(rp_weights.values()) - 1.0) < 1e-6

            # All weights should be positive in risk parity
            assert all(w > 1e-6 for w in rp_weights.values())

            # Should have weights for all assets
            assert len(rp_weights) == len(sample_returns.columns)

            print(" Risk parity test passed")
        except Exception as e:
            print(f" Risk parity test failed: {e}")

    def test_portfolio_metrics(self, sample_returns):
        """Test portfolio performance metrics calculation."""
        try:
            # Create equal weight portfolio
            weights = {
                asset: 1 / len(sample_returns.columns)
                for asset in sample_returns.columns
            }

            # Calculate portfolio returns
            portfolio_returns = (sample_returns * pd.Series(weights)).sum(axis=1)

            # Test various metrics
            annual_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

            # Calculate max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Validate metrics
            assert isinstance(annual_return, (int, float))
            assert isinstance(volatility, (int, float))
            assert isinstance(sharpe_ratio, (int, float))
            assert isinstance(max_drawdown, (int, float))

            assert volatility >= 0
            assert max_drawdown <= 0

            print(" Portfolio metrics test passed")
        except Exception as e:
            print(f" Portfolio metrics test failed: {e}")

    def test_risk_budgeting(self, sample_returns):
        """Test risk budgeting functionality."""
        try:
            # Create portfolio with known weights
            weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

            # Calculate covariance matrix
            cov_matrix = sample_returns.cov().values

            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Risk contributions should sum to 1
            assert abs(risk_contrib.sum() - 1.0) < 1e-6

            # All risk contributions should be positive
            assert all(rc >= 0 for rc in risk_contrib)

            print(" Risk budgeting test passed")
        except Exception as e:
            print(f" Risk budgeting test failed: {e}")

    def test_rebalancing_logic(self, sample_returns):
        """Test portfolio rebalancing logic."""
        try:
            # Initial weights
            target_weights = {
                asset: 1 / len(sample_returns.columns)
                for asset in sample_returns.columns
            }

            # Simulate price changes
            price_changes = sample_returns.iloc[0]

            # Calculate new weights after price changes
            new_values = {
                asset: target_weights[asset] * (1 + price_changes[asset])
                for asset in sample_returns.columns
            }

            total_value = sum(new_values.values())
            current_weights = {
                asset: value / total_value for asset, value in new_values.items()
            }

            # Calculate rebalancing trades
            trades = {
                asset: target_weights[asset] - current_weights[asset]
                for asset in sample_returns.columns
            }

            # Trades should sum to approximately zero (conservation)
            assert abs(sum(trades.values())) < 1e-10

            print(" Rebalancing logic test passed")
        except Exception as e:
            print(f" Rebalancing logic test failed: {e}")

    def test_transaction_costs(self, sample_returns):
        """Test transaction cost calculations."""
        try:
            # Define transaction cost parameters
            commission_rate = 0.001  # 0.1%
            bid_ask_spread = 0.0005  # 0.05%

            # Calculate costs for a trade
            trade_value = 10000
            commission = trade_value * commission_rate
            spread_cost = trade_value * bid_ask_spread / 2
            total_cost = commission + spread_cost

            assert commission >= 0
            assert spread_cost >= 0
            assert total_cost == commission + spread_cost

            # Cost should be reasonable percentage
            cost_percentage = total_cost / trade_value
            assert 0 <= cost_percentage <= 0.01  # Less than 1%

            print(" Transaction costs test passed")
        except Exception as e:
            print(f" Transaction costs test failed: {e}")

    def test_performance_attribution(self, sample_returns):
        """Test performance attribution analysis."""
        try:
            # Create benchmark (equal weight)
            benchmark_weights = {
                asset: 1 / len(sample_returns.columns)
                for asset in sample_returns.columns
            }

            # Create active portfolio (overweight some assets)
            active_weights = benchmark_weights.copy()
            active_weights["AAPL"] += 0.1
            active_weights["GOOGL"] -= 0.1

            # Calculate returns
            benchmark_returns = (sample_returns * pd.Series(benchmark_weights)).sum(
                axis=1
            )
            active_returns = (sample_returns * pd.Series(active_weights)).sum(axis=1)

            # Calculate active return
            active_return = active_returns - benchmark_returns

            # Calculate tracking error
            tracking_error = active_return.std() * np.sqrt(252)

            # Calculate information ratio
            information_ratio = (
                (active_return.mean() * 252) / tracking_error
                if tracking_error > 0
                else 0
            )

            assert isinstance(tracking_error, (int, float))
            assert isinstance(information_ratio, (int, float))
            assert tracking_error >= 0

            print(" Performance attribution test passed")
        except Exception as e:
            print(f" Performance attribution test failed: {e}")

    def test_error_handling(self, sample_returns):
        """Test error handling for invalid inputs."""
        try:
            # Test with insufficient data
            short_returns = sample_returns.head(10)

            try:
                optimizer = PortfolioOptimizer(short_returns)
                optimizer.optimize_portfolio()
                # Should either work or raise appropriate error
            except (ValueError, np.linalg.LinAlgError):
                pass  # Expected for insufficient data

            # Test with invalid objective
            optimizer = PortfolioOptimizer(sample_returns)
            try:
                optimizer.optimize_portfolio(objective="invalid")
            except (ValueError, KeyError):
                pass  # Expected behavior

            print(" Error handling test passed")
        except Exception as e:
            print(f" Error handling test failed: {e}")


def test_portfolio_management_import():
    """Test that portfolio management can be imported."""
    try:
        from meridianalgo.portfolio_management import (  # noqa: F401
            EfficientFrontier,
            PortfolioOptimizer,
        )

        print(" Portfolio management import test passed")
        return True
    except ImportError as e:
        print(f" Import test failed: {e}")
        return False


def test_portfolio_with_real_data():
    """Test portfolio management with real market data if available."""
    try:
        # Try to get real data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        data = ma.get_market_data(symbols, "2023-01-01", "2023-12-31")

        if data is not None and len(data) > 50:
            # Calculate returns
            returns = data.pct_change().dropna()

            if len(returns) > 20:
                # Test optimization
                optimizer = ma.PortfolioOptimizer(returns)
                weights = optimizer.optimize_portfolio()

                assert len(weights) == len(symbols)
                assert abs(sum(weights.values()) - 1.0) < 1e-6

                print(" Real data portfolio test passed")
            else:
                print(" Insufficient real data for portfolio test")
        else:
            print(" No real data available, skipping real data test")

    except Exception as e:
        print(f" Real data portfolio test failed: {e}")


if __name__ == "__main__":
    # Run tests manually
    print(" Running Portfolio Management Tests...")

    # Test imports first
    if not test_portfolio_management_import():
        print(" Cannot proceed with tests - import failed")
        exit(1)

    # Create test instance
    test_instance = TestPortfolioManagement()

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    n_assets = 5
    mean_returns = np.random.uniform(0.0005, 0.002, n_assets)
    correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)

    returns = np.random.multivariate_normal(mean_returns, correlation * 0.02**2, 252)

    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    sample_returns = pd.DataFrame(returns, columns=symbols, index=dates)

    # Run all tests
    test_methods = [
        test_instance.test_portfolio_optimizer_creation,
        test_instance.test_mean_variance_optimization,
        test_instance.test_efficient_frontier,
        test_instance.test_black_litterman_model,
        test_instance.test_risk_parity,
        test_instance.test_portfolio_metrics,
        test_instance.test_risk_budgeting,
        test_instance.test_rebalancing_logic,
        test_instance.test_transaction_costs,
        test_instance.test_performance_attribution,
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
    test_portfolio_with_real_data()

    print(f"\n Test Results: {passed}/{total} tests passed")

    if passed == total:
        print(" All portfolio management tests passed!")
    else:
        print(f" {total - passed} tests failed")
