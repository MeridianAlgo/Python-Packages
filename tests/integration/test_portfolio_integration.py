"""
Integration test for portfolio management and performance analytics.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the package to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def create_sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")

    # Generate returns with positive drift and some volatility
    returns = pd.Series(
        np.random.normal(
            0.0008, 0.015, len(dates)
        ),  # ~20% annual return, 15% volatility
        index=dates,
    )

    return returns


def create_sample_portfolio_data():
    """Create sample portfolio data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    assets = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    # Generate correlated returns
    n_assets = len(assets)
    n_periods = len(dates)

    # Create correlation matrix
    correlation = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)

    # Generate returns
    returns = np.random.multivariate_normal(
        mean=[0.0008] * n_assets, cov=correlation * 0.015**2, size=n_periods
    )

    return pd.DataFrame(returns, index=dates, columns=assets)


def test_performance_analytics():
    """Test performance analytics system."""
    print("Testing Performance Analytics...")

    try:
        from meridianalgo.backtesting.performance_analytics import PerformanceAnalyzer

        returns = create_sample_returns()
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze_returns(returns)

        # Basic checks
        assert metrics.total_return != 0, "Total return should not be zero"
        assert metrics.volatility > 0, "Volatility should be positive"
        assert metrics.trading_days == len(returns), (
            "Trading days should match data length"
        )

        print(" Performance analytics working correctly")
        print(f"  Total Return: {metrics.total_return:.2%}")
        print(f"  Annualized Return: {metrics.annualized_return:.2%}")
        print(f"  Volatility: {metrics.volatility:.2%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

        return True

    except Exception as e:
        print(f" Performance analytics test failed: {e}")
        return False


def test_portfolio_optimization():
    """Test portfolio optimization."""
    print("\nTesting Portfolio Optimization...")

    try:
        from meridianalgo.portfolio.optimization import PortfolioOptimizer

        returns_data = create_sample_portfolio_data()
        optimizer = PortfolioOptimizer()

        # Test mean-variance optimization
        weights = optimizer.optimize(
            returns_data, method="mean_variance", target_return=0.12
        )

        assert len(weights) == len(returns_data.columns), (
            "Weights should match number of assets"
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"
        assert all(w >= 0 for w in weights.values()), "Weights should be non-negative"

        print(" Portfolio optimization working correctly")
        print(f"  Optimized weights: {dict(weights)}")

        return True

    except Exception as e:
        print(f" Portfolio optimization test failed: {e}")
        return False


def test_risk_management():
    """Test risk management system."""
    print("\nTesting Risk Management...")

    try:
        from meridianalgo.portfolio.risk_management import RiskManager

        returns_data = create_sample_portfolio_data()
        risk_manager = RiskManager()

        # Test VaR calculation
        var_95 = risk_manager.calculate_var(returns_data, confidence_level=0.95)
        var_99 = risk_manager.calculate_var(returns_data, confidence_level=0.99)

        assert var_95 < 0, "VaR should be negative (loss)"
        assert var_99 < var_95, "99% VaR should be more negative than 95% VaR"

        # Test Expected Shortfall
        es_95 = risk_manager.calculate_expected_shortfall(
            returns_data, confidence_level=0.95
        )

        assert es_95 < var_95, "Expected Shortfall should be more negative than VaR"

        print(" Risk management working correctly")
        print(f"  VaR (95%): {var_95:.4f}")
        print(f"  VaR (99%): {var_99:.4f}")
        print(f"  ES (95%): {es_95:.4f}")

        return True

    except Exception as e:
        print(f" Risk management test failed: {e}")
        return False


def test_transaction_costs():
    """Test transaction cost optimization."""
    print("\nTesting Transaction Cost Optimization...")

    try:
        from meridianalgo.portfolio.transaction_costs import TransactionCostOptimizer

        optimizer = TransactionCostOptimizer()

        # Create sample portfolio data
        current_weights = pd.Series(
            [0.4, 0.3, 0.2, 0.1], index=["AAPL", "GOOGL", "MSFT", "TSLA"]
        )
        target_weights = pd.Series(
            [0.3, 0.3, 0.3, 0.1], index=["AAPL", "GOOGL", "MSFT", "TSLA"]
        )

        # Test cost calculation
        total_cost = optimizer.calculate_total_cost(
            current_weights, target_weights, portfolio_value=1000000
        )

        assert total_cost >= 0, "Transaction costs should be non-negative"

        print(" Transaction cost optimization working correctly")
        print(f"  Total transaction cost: ${total_cost:.2f}")

        return True

    except Exception as e:
        print(f" Transaction cost optimization test failed: {e}")
        return False


def main():
    """Run all portfolio integration tests."""
    print("Portfolio Management Integration Tests")
    print("=" * 50)

    tests = [
        test_performance_analytics,
        test_portfolio_optimization,
        test_risk_management,
        test_transaction_costs,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'=' * 50}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print(" All portfolio integration tests passed!")
        return True
    else:
        print(" Some portfolio integration tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
