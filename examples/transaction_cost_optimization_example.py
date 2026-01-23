"""
Example demonstrating transaction cost optimization and execution algorithms.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from meridianalgo.portfolio.optimization import (
    PortfolioOptimizer,
    RebalancingOptimizer,
    TransactionCostAwareOptimizer,
)
from meridianalgo.portfolio.transaction_costs import (
    LinearImpactModel,
    SquareRootImpactModel,
    TransactionCostOptimizer,
)


def demonstrate_execution_algorithms():
    """Demonstrate different execution algorithms."""
    print("=== Execution Algorithms Demonstration ===\n")

    # Create market data
    market_data = {
        "volume": 2000000,  # 2M shares daily volume
        "volatility": 0.30,  # 30% annual volatility
        "price": 150.0,  # $150 per share
        "bid_ask_spread": 0.02,
    }

    target_quantity = 50000  # Want to buy 50,000 shares
    execution_horizon = 20  # Over 20 time periods

    optimizer = TransactionCostOptimizer()

    algorithms = ["twap_linear", "vwap_linear", "is_square_root"]
    results = {}

    for algo in algorithms:
        print(f"Testing {algo.upper()} algorithm:")

        result = optimizer.optimize_execution(
            target_quantity=target_quantity,
            market_data=market_data,
            execution_horizon=execution_horizon,
            algorithm=algo,
        )

        results[algo] = result

        print(f"  Total Cost: ${result.total_cost:,.2f}")
        print(f"  Market Impact: {result.market_impact:.6f}")
        print(f"  Timing Risk: {result.timing_risk:.6f}")
        print(f"  Success: {result.success}")
        print()

    return results


def demonstrate_market_impact_models():
    """Demonstrate different market impact models."""
    print("=== Market Impact Models Comparison ===\n")

    # Test different trade sizes
    trade_sizes = np.array([1000, 5000, 10000, 25000, 50000, 100000])
    volume = 1000000  # 1M daily volume
    volatility = 0.25
    price = 100.0

    linear_model = LinearImpactModel()
    sqrt_model = SquareRootImpactModel()

    linear_impacts = []
    sqrt_impacts = []

    for size in trade_sizes:
        linear_impact = linear_model.calculate_impact(size, volume, volatility)
        sqrt_impact = sqrt_model.calculate_impact(size, volume, volatility, price)

        linear_impacts.append(linear_impact)
        sqrt_impacts.append(sqrt_impact / price)  # Convert to percentage

    # Create comparison DataFrame
    comparison = pd.DataFrame(
        {
            "Trade_Size": trade_sizes,
            "Participation_Rate": trade_sizes / volume * 100,
            "Linear_Impact_%": np.array(linear_impacts) * 100,
            "SquareRoot_Impact_%": np.array(sqrt_impacts) * 100,
        }
    )

    print(comparison.round(4))
    print()

    return comparison


def demonstrate_tax_loss_harvesting():
    """Demonstrate tax-loss harvesting optimization."""
    print("=== Tax-Loss Harvesting Demonstration ===\n")

    # Create portfolio with some losses
    portfolio = pd.DataFrame(
        {
            "asset": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "quantity": [100, 50, 200, 75, 30],
            "cost_basis": [
                180.0,
                2900.0,
                320.0,
                900.0,
                3500.0,
            ],  # Higher than current prices
            "purchase_date": [
                "2023-01-15",
                "2023-03-20",
                "2023-02-10",
                "2023-04-05",
                "2023-05-12",
            ],
        }
    )

    # Current prices (some losses, some gains)
    current_prices = pd.DataFrame(
        {
            "AAPL": [150.0],  # Loss
            "GOOGL": [2700.0],  # Loss
            "MSFT": [310.0],  # Gain
            "TSLA": [700.0],  # Loss
            "AMZN": [3600.0],  # Gain
        }
    )

    optimizer = TransactionCostOptimizer()

    result = optimizer.optimize_tax_harvesting(
        portfolio=portfolio,
        prices=current_prices,
        tax_rate=0.25,
        min_loss_threshold=500.0,
    )

    print("Tax Harvesting Results:")
    print(f"  Total Tax Savings: ${result.total_tax_savings:,.2f}")
    print(f"  Realized Losses: ${result.realized_losses:,.2f}")
    print(f"  Realized Gains: ${result.realized_gains:,.2f}")
    print(f"  Number of Trades: {len(result.trades)}")
    print(f"  Wash Sale Violations: {len(result.wash_sale_violations)}")
    print()

    if len(result.trades) > 0:
        print("Recommended Trades:")
        print(
            result.trades[["asset", "action", "quantity", "pnl", "tax_savings"]].round(
                2
            )
        )
        print()

    return result


def demonstrate_transaction_cost_aware_optimization():
    """Demonstrate portfolio optimization with transaction costs."""
    print("=== Transaction-Cost-Aware Portfolio Optimization ===\n")

    # Create sample data
    assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    expected_returns = pd.Series([0.12, 0.15, 0.10, 0.18, 0.14], index=assets)

    # Create covariance matrix
    np.random.seed(42)
    corr_matrix = np.random.uniform(0.1, 0.5, (5, 5))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)

    volatilities = np.array([0.25, 0.30, 0.22, 0.40, 0.28])
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    cov_df = pd.DataFrame(cov_matrix, index=assets, columns=assets)

    # Current portfolio (need to rebalance)
    current_weights = pd.Series([0.50, 0.10, 0.20, 0.05, 0.15], index=assets)

    # Market data
    market_data = {
        "AAPL": {"volume": 50000000, "volatility": 0.25, "price": 150.0},
        "GOOGL": {"volume": 25000000, "volatility": 0.30, "price": 2800.0},
        "MSFT": {"volume": 40000000, "volatility": 0.22, "price": 300.0},
        "TSLA": {"volume": 30000000, "volatility": 0.40, "price": 800.0},
        "AMZN": {"volume": 35000000, "volatility": 0.28, "price": 3500.0},
    }

    portfolio_value = 10000000  # $10M portfolio

    # Compare standard optimization vs transaction-cost-aware
    print("Standard Portfolio Optimization:")
    standard_optimizer = PortfolioOptimizer()
    standard_result = standard_optimizer.optimize(
        expected_returns, cov_df, objective="max_sharpe"
    )

    print(f"  Expected Return: {standard_result.expected_return:.4f}")
    print(f"  Volatility: {standard_result.volatility:.4f}")
    print(f"  Sharpe Ratio: {standard_result.sharpe_ratio:.4f}")
    print(f"  Weights: {dict(zip(assets, standard_result.weights.round(4)))}")

    # Calculate transaction costs for standard optimization
    cost_optimizer = TransactionCostOptimizer()
    standard_costs = cost_optimizer.calculate_rebalancing_costs(
        current_weights, standard_result.weights, portfolio_value, market_data
    )
    print(
        f"  Transaction Costs: ${standard_costs['total_cost']:,.2f} ({standard_costs['cost_percentage']:.3f}%)"
    )
    print()

    print("Transaction-Cost-Aware Optimization:")
    tc_optimizer = TransactionCostAwareOptimizer()
    tc_result = tc_optimizer.optimize(
        expected_returns,
        cov_df,
        current_weights=current_weights,
        portfolio_value=portfolio_value,
        market_data=market_data,
        transaction_cost_penalty=5.0,
        objective="max_sharpe",
    )

    print(f"  Expected Return: {tc_result.expected_return:.4f}")
    print(f"  Volatility: {tc_result.volatility:.4f}")
    print(f"  Sharpe Ratio: {tc_result.sharpe_ratio:.4f}")
    print(f"  Weights: {dict(zip(assets, tc_result.weights.round(4)))}")
    print(
        f"  Transaction Costs: ${tc_result.metadata['transaction_costs']['total_cost']:,.2f}"
    )
    print(f"  Turnover: {tc_result.metadata['turnover']:.4f}")
    print()

    return standard_result, tc_result


def demonstrate_rebalancing_frequency_optimization():
    """Demonstrate optimal rebalancing frequency analysis."""
    print("=== Rebalancing Frequency Optimization ===\n")

    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    assets = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    # Generate correlated returns
    returns_data = pd.DataFrame(
        np.random.multivariate_normal(
            [0.0005, 0.0006, 0.0004, 0.0008],  # Daily expected returns
            [
                [0.0004, 0.0001, 0.0001, 0.0002],
                [0.0001, 0.0009, 0.0002, 0.0003],
                [0.0001, 0.0002, 0.0003, 0.0001],
                [0.0002, 0.0003, 0.0001, 0.0016],
            ],  # Covariance matrix
            len(dates),
        ),
        index=dates,
        columns=assets,
    )

    # Target weights (equal weight)
    target_weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=assets)

    # Market data
    market_data = {
        "AAPL": {"volume": 50000000, "volatility": 0.25, "price": 150.0},
        "GOOGL": {"volume": 25000000, "volatility": 0.30, "price": 2800.0},
        "MSFT": {"volume": 40000000, "volatility": 0.22, "price": 300.0},
        "TSLA": {"volume": 30000000, "volatility": 0.40, "price": 800.0},
    }

    rebalancer = RebalancingOptimizer()

    result = rebalancer.optimize_rebalancing_frequency(
        returns_data=returns_data,
        target_weights=target_weights,
        rebalancing_frequencies=[1, 5, 10, 20, 30, 60, 90],
        market_data=market_data,
        portfolio_value=5000000,  # $5M portfolio
    )

    print(f"Optimal Rebalancing Frequency: {result['optimal_frequency']} days")
    print()

    print("Frequency Analysis:")
    freq_df = pd.DataFrame(result["results_by_frequency"]).T
    print(
        freq_df[
            [
                "frequency_days",
                "cost_percentage",
                "avg_tracking_error",
                "num_rebalances",
            ]
        ].round(4)
    )
    print()

    return result


def create_visualization_plots(execution_results, impact_comparison):
    """Create visualization plots for the results."""
    try:
        import matplotlib.pyplot as plt

        # Plot 1: Execution Algorithm Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Total costs comparison
        algorithms = list(execution_results.keys())
        costs = [execution_results[algo].total_cost for algo in algorithms]

        ax1.bar(algorithms, costs)
        ax1.set_title("Total Execution Costs by Algorithm")
        ax1.set_ylabel("Cost ($)")
        ax1.tick_params(axis="x", rotation=45)

        # Market impact comparison
        impacts = [execution_results[algo].market_impact for algo in algorithms]
        ax2.bar(algorithms, impacts)
        ax2.set_title("Market Impact by Algorithm")
        ax2.set_ylabel("Impact (fraction)")
        ax2.tick_params(axis="x", rotation=45)

        # Impact model comparison
        ax3.plot(
            impact_comparison["Trade_Size"],
            impact_comparison["Linear_Impact_%"],
            label="Linear Model",
            marker="o",
        )
        ax3.plot(
            impact_comparison["Trade_Size"],
            impact_comparison["SquareRoot_Impact_%"],
            label="Square Root Model",
            marker="s",
        )
        ax3.set_title("Market Impact Models Comparison")
        ax3.set_xlabel("Trade Size (shares)")
        ax3.set_ylabel("Market Impact (%)")
        ax3.legend()
        ax3.grid(True)

        # Participation rate vs impact
        ax4.plot(
            impact_comparison["Participation_Rate"],
            impact_comparison["Linear_Impact_%"],
            label="Linear Model",
            marker="o",
        )
        ax4.plot(
            impact_comparison["Participation_Rate"],
            impact_comparison["SquareRoot_Impact_%"],
            label="Square Root Model",
            marker="s",
        )
        ax4.set_title("Participation Rate vs Market Impact")
        ax4.set_xlabel("Participation Rate (%)")
        ax4.set_ylabel("Market Impact (%)")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig("transaction_cost_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    except ImportError:
        print("Matplotlib not available for plotting")


def main():
    """Run all transaction cost optimization demonstrations."""
    print("Transaction Cost Optimization and Execution Algorithms Demo")
    print("=" * 60)
    print()

    # 1. Execution algorithms
    execution_results = demonstrate_execution_algorithms()

    # 2. Market impact models
    impact_comparison = demonstrate_market_impact_models()

    # 3. Tax-loss harvesting
    demonstrate_tax_loss_harvesting()

    # 4. Transaction-cost-aware optimization
    standard_result, tc_result = demonstrate_transaction_cost_aware_optimization()

    # 5. Rebalancing frequency optimization
    demonstrate_rebalancing_frequency_optimization()

    # 6. Create visualizations
    create_visualization_plots(execution_results, impact_comparison)

    print("=" * 60)
    print("Demo completed successfully!")
    print()
    print("Key Takeaways:")
    print("1. Different execution algorithms have varying cost/risk profiles")
    print("2. Market impact models help estimate transaction costs")
    print("3. Tax-loss harvesting can provide significant tax savings")
    print("4. Transaction-cost-aware optimization reduces turnover")
    print("5. Optimal rebalancing frequency balances costs and tracking error")


if __name__ == "__main__":
    main()
