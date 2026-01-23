"""
Advanced trading strategy example using MeridianAlgo.
"""

import numpy as np
import pandas as pd

import meridianalgo as ma


class TradingStrategy:
    """A simple mean-reversion trading strategy using statistical arbitrage."""

    def __init__(self, lookback_window=30, entry_threshold=2.0, exit_threshold=0.5):
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.positions = {}
        self.trades = []

    def calculate_zscore(self, prices):
        """Calculate rolling z-score for mean reversion."""
        rolling_mean = prices.rolling(window=self.lookback_window).mean()
        rolling_std = prices.rolling(window=self.lookback_window).std()
        return (prices - rolling_mean) / rolling_std

    def generate_signals(self, data):
        """Generate trading signals based on z-score."""
        signals = pd.DataFrame(index=data.index)

        for ticker in data.columns:
            prices = data[ticker]
            zscore = self.calculate_zscore(prices)

            # Generate signals
            signals[f"{ticker}_zscore"] = zscore
            signals[f"{ticker}_signal"] = 0

            # Entry signals (mean reversion)
            signals.loc[zscore < -self.entry_threshold, f"{ticker}_signal"] = 1  # Buy
            signals.loc[zscore > self.entry_threshold, f"{ticker}_signal"] = -1  # Sell

            # Exit signals
            signals.loc[
                (zscore > -self.exit_threshold) & (zscore < self.exit_threshold),
                f"{ticker}_signal",
            ] = 0

        return signals

    def backtest(self, data, initial_capital=100000):
        """Backtest the trading strategy."""
        signals = self.generate_signals(data)
        _ = data.pct_change()  # Calculate returns for potential use

        portfolio_value = initial_capital
        position_sizes = {}

        results = []

        for date in data.index[1:]:
            current_portfolio_value = portfolio_value

            for ticker in data.columns:
                signal = signals.loc[date, f"{ticker}_signal"]
                price = data.loc[date, ticker]

                if signal == 1 and ticker not in position_sizes:  # Buy signal
                    # Use 10% of portfolio for each position
                    position_value = portfolio_value * 0.1
                    position_sizes[ticker] = position_value / price
                    portfolio_value -= position_value

                elif signal == -1 and ticker in position_sizes:  # Sell signal
                    # Close position
                    position_value = position_sizes[ticker] * price
                    portfolio_value += position_value
                    del position_sizes[ticker]

            # Update portfolio value with current positions
            for ticker, shares in position_sizes.items():
                portfolio_value += shares * data.loc[date, ticker]

            # Calculate daily return
            daily_return = (
                portfolio_value - current_portfolio_value
            ) / current_portfolio_value

            results.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "daily_return": daily_return,
                    "positions": len(position_sizes),
                }
            )

        return pd.DataFrame(results).set_index("date")


def example_trading_strategy():
    """Example of implementing a trading strategy."""
    print("=== Trading Strategy Example ===")

    # Get data for multiple assets
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    data = ma.get_market_data(tickers, start_date="2023-01-01", end_date="2024-01-01")

    print(f"Retrieved data for {len(data)} days")
    print(f"Assets: {list(data.columns)}")

    # Create and run strategy
    strategy = TradingStrategy(lookback_window=30, entry_threshold=2.0)
    results = strategy.backtest(data)

    # Calculate performance metrics
    returns = results["daily_return"]
    metrics = ma.calculate_metrics(returns)

    print("\nStrategy Performance:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

    # Calculate risk metrics
    var_95 = ma.calculate_value_at_risk(returns, confidence_level=0.95)
    es_95 = ma.calculate_expected_shortfall(returns, confidence_level=0.95)

    print("\nRisk Metrics:")
    print(f"95% VaR: {var_95:.2%}")
    print(f"95% Expected Shortfall: {es_95:.2%}")

    return results


def example_portfolio_optimization_advanced():
    """Advanced portfolio optimization example."""
    print("\n=== Advanced Portfolio Optimization ===")

    # Get data for multiple assets
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    data = ma.get_market_data(tickers, start_date="2023-01-01", end_date="2024-01-01")
    returns = data.pct_change().dropna()

    # Create portfolio optimizer
    optimizer = ma.PortfolioOptimizer(returns)

    # Calculate efficient frontier
    frontier = optimizer.calculate_efficient_frontier(num_portfolios=2000)

    # Find portfolios with different risk levels
    low_risk_idx = np.argmin(frontier["volatility"])
    high_return_idx = np.argmax(frontier["returns"])
    max_sharpe_idx = np.argmax(frontier["sharpe"])

    portfolios = {
        "Low Risk": low_risk_idx,
        "High Return": high_return_idx,
        "Max Sharpe": max_sharpe_idx,
    }

    print("Portfolio Allocations:")
    for name, idx in portfolios.items():
        weights = frontier["weights"][idx]
        print(f"\n{name}:")
        print(f"  Return: {frontier['returns'][idx]:.2%}")
        print(f"  Volatility: {frontier['volatility'][idx]:.2%}")
        print(f"  Sharpe Ratio: {frontier['sharpe'][idx]:.2f}")
        print("  Weights:")
        for i, ticker in enumerate(tickers):
            if weights[i] > 0.01:  # Only show weights > 1%
                print(f"    {ticker}: {weights[i]:.2%}")


def example_risk_analysis():
    """Comprehensive risk analysis example."""
    print("\n=== Risk Analysis Example ===")

    # Get data for a single asset
    data = ma.get_market_data(["AAPL"], start_date="2023-01-01", end_date="2024-01-01")
    returns = data["AAPL"].pct_change().dropna()

    # Calculate various risk metrics
    print("Risk Metrics for AAPL:")

    # Basic metrics
    metrics = ma.calculate_metrics(returns)
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

    # VaR at different confidence levels
    for conf in [0.90, 0.95, 0.99]:
        var = ma.calculate_value_at_risk(returns, confidence_level=conf)
        es = ma.calculate_expected_shortfall(returns, confidence_level=conf)
        print(f"{conf*100:.0f}% VaR: {var:.2%}")
        print(f"{conf*100:.0f}% ES: {es:.2%}")

    # Rolling volatility
    rolling_vol = ma.rolling_volatility(returns, window=21, annualized=True)
    print(f"Average Rolling Volatility: {rolling_vol.mean():.2%}")
    print(f"Max Rolling Volatility: {rolling_vol.max():.2%}")

    # Hurst exponent
    hurst = ma.hurst_exponent(returns)
    print(f"Hurst Exponent: {hurst:.3f}")

    # Autocorrelation
    autocorr = ma.calculate_autocorrelation(returns, lag=1)
    print(f"1-day Autocorrelation: {autocorr:.3f}")


def main():
    """Run all advanced examples."""
    print("MeridianAlgo Advanced Examples")
    print("=" * 50)

    try:
        # Run trading strategy example
        example_trading_strategy()

        # Run advanced portfolio optimization
        example_portfolio_optimization_advanced()

        # Run risk analysis
        example_risk_analysis()

        print("\n" + "=" * 50)
        print("All advanced examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Please check your internet connection and dependencies.")


if __name__ == "__main__":
    main()
