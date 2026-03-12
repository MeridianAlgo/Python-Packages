"""
Basic Usage Examples for MeridianAlgo

This file demonstrates the fundamental capabilities of MeridianAlgo for quantitative finance.
Perfect for getting started with portfolio optimization, risk analysis, and time series analysis.

What you'll learn:
- How to fetch and analyze market data
- Portfolio optimization using Modern Portfolio Theory
- Time series analysis and performance metrics
- Risk measurement (VaR, Expected Shortfall, Hurst exponent)
- Statistical arbitrage and pairs trading
- Machine learning for price prediction
"""

import numpy as np

import meridianalgo as ma


def example_portfolio_optimization():
    """
    Portfolio Optimization Using Modern Portfolio Theory

    This example shows how to build an optimal portfolio that maximizes
    risk-adjusted returns (Sharpe ratio). We'll analyze multiple tech stocks
    and find the best allocation across them.

    The efficient frontier represents all possible portfolios with different
    risk-return profiles. We'll identify the one with the highest Sharpe ratio,
    which gives us the best return per unit of risk.
    """
    print("=== Portfolio Optimization Example ===")

    # Define the stocks we want to analyze
    # These are major tech companies with different risk profiles
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

    # Fetch historical price data for one year
    # This data will be used to calculate returns and correlations
    data = ma.get_market_data(tickers, start_date="2023-01-01", end_date="2024-01-01")

    # Convert prices to daily returns
    # Returns are more stationary than prices and better for analysis
    returns = data.pct_change().dropna()
    print(f"Retrieved data for {len(returns)} days")
    print(f"Assets: {list(returns.columns)}")

    # Initialize the portfolio optimizer with our returns data
    # This will calculate the covariance matrix and expected returns
    optimizer = ma.PortfolioOptimizer(returns)

    # Generate the efficient frontier by simulating 1000 random portfolios
    # Each portfolio has different weights, and we track their risk-return profiles
    frontier = optimizer.calculate_efficient_frontier(num_portfolios=1000)

    # Find the portfolio with the maximum Sharpe ratio
    # Sharpe ratio = (Return - Risk-free rate) / Volatility
    # Higher is better - it means more return for each unit of risk
    max_sharpe_idx = np.argmax(frontier["sharpe"])
    optimal_weights = frontier["weights"][max_sharpe_idx]

    print("\nOptimal Portfolio Weights:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {optimal_weights[i]:.2%}")

    # Display the expected performance of this optimal portfolio
    print(f"Expected Return: {frontier['returns'][max_sharpe_idx]:.2%}")
    print(f"Volatility: {frontier['volatility'][max_sharpe_idx]:.2%}")
    print(f"Sharpe Ratio: {frontier['sharpe'][max_sharpe_idx]:.2f}")


def example_time_series_analysis():
    """
    Time Series Analysis and Performance Metrics

    Time series analysis helps us understand how an asset behaves over time.
    We'll calculate key performance metrics that professional traders use
    to evaluate investments.

    Key metrics explained:
    - Total Return: How much the investment gained/lost overall
    - Annualized Return: Average yearly return (for comparing different time periods)
    - Volatility: How much the price fluctuates (higher = riskier)
    - Sharpe Ratio: Return per unit of risk (higher is better)
    - Max Drawdown: Largest peak-to-trough decline (worst loss period)
    """
    print("\n=== Time Series Analysis Example ===")

    # Fetch one year of Apple stock data
    data = ma.get_market_data(["AAPL"], start_date="2023-01-01", end_date="2024-01-01")
    prices = data["AAPL"]

    # Create a time series analyzer to process the price data
    analyzer = ma.TimeSeriesAnalyzer(prices)

    # Calculate daily returns (percentage change from day to day)
    returns = analyzer.calculate_returns()

    # Calculate rolling volatility over 21-day windows (about 1 trading month)
    # Annualized volatility is what most professionals report
    analyzer.calculate_volatility(window=21, annualized=True)

    # Compute comprehensive performance metrics
    # These metrics give us a complete picture of risk and return
    metrics = ma.calculate_metrics(returns)

    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")


def example_risk_metrics():
    """
    Advanced Risk Measurement Techniques

    Risk management is crucial for successful trading. This example demonstrates
    professional risk metrics used by hedge funds and institutional investors.

    Metrics explained:
    - Value at Risk (VaR): Maximum expected loss at a given confidence level
      Example: 95% VaR of 2% means there's only a 5% chance of losing more than 2%
    - Expected Shortfall (ES): Average loss when VaR is exceeded (tail risk)
    - Hurst Exponent: Measures if prices trend, mean-revert, or random walk
      > 0.5 = trending (momentum strategies work)
      < 0.5 = mean-reverting (pairs trading works)
      = 0.5 = random walk (efficient market)
    """
    print("\n=== Risk Metrics Example ===")

    # Fetch Apple stock data
    data = ma.get_market_data(["AAPL"], start_date="2023-01-01", end_date="2024-01-01")
    returns = data["AAPL"].pct_change().dropna()

    # Calculate Value at Risk at different confidence levels
    # 95% VaR: What's the maximum loss we'd expect 95% of the time?
    var_95 = ma.calculate_value_at_risk(returns, confidence_level=0.95)

    # 99% VaR: More conservative - what's the max loss 99% of the time?
    var_99 = ma.calculate_value_at_risk(returns, confidence_level=0.99)

    # Expected Shortfall: When things go really bad, how bad do they get?
    # This measures the average loss in the worst 5% of cases
    es_95 = ma.calculate_expected_shortfall(returns, confidence_level=0.95)

    print(f"95% Value at Risk: {var_95:.2%}")
    print(f"99% Value at Risk: {var_99:.2%}")
    print(f"95% Expected Shortfall: {es_95:.2%}")

    # Calculate the Hurst exponent to understand price behavior
    # This tells us if the stock trends or mean-reverts
    hurst = ma.hurst_exponent(returns)
    print(f"Hurst Exponent: {hurst:.3f}")

    # Interpret the Hurst exponent for trading strategy selection
    if hurst > 0.5:
        print("   Series shows trending behavior - momentum strategies may work well")
    elif hurst < 0.5:
        print("   Series shows mean-reverting behavior - pairs trading may work well")
    else:
        print("   Series shows random walk behavior - market is efficient")


def example_statistical_arbitrage():
    """
    Statistical Arbitrage and Pairs Trading

    Statistical arbitrage exploits temporary price divergences between related assets.
    When two stocks normally move together but temporarily diverge, we can profit
    by betting they'll converge again.

    This strategy is used by:
    - Hedge funds for market-neutral returns
    - High-frequency trading firms
    - Quantitative trading desks

    Key concepts:
    - Correlation: How closely two stocks move together (-1 to +1)
    - Cointegration: Whether two stocks have a stable long-term relationship
      (more important than correlation for pairs trading)
    """
    print("\n=== Statistical Arbitrage Example ===")

    # Fetch data for two tech giants that often move together
    # Apple and Microsoft are good candidates because they're both in tech
    data = ma.get_market_data(
        ["AAPL", "MSFT"], start_date="2023-01-01", end_date="2024-01-01"
    )

    # Initialize the statistical arbitrage analyzer
    arb = ma.StatisticalArbitrage(data)

    # Calculate how correlation changes over time using a 30-day rolling window
    # This helps us see if the relationship is stable or breaking down
    rolling_corr = arb.calculate_rolling_correlation(window=30)

    # Test for cointegration - the key test for pairs trading
    # Cointegration means the stocks have a stable long-term relationship
    # even if they diverge temporarily
    try:
        coint_result = arb.test_cointegration(data["AAPL"], data["MSFT"])
        print(f"Cointegration Test Statistic: {coint_result['test_statistic']:.3f}")
        print(f"P-value: {coint_result['p_value']:.3f}")
        print(f"Cointegrated: {coint_result['is_cointegrated']}")

        if coint_result["is_cointegrated"]:
            print("   These stocks are good candidates for pairs trading!")
        else:
            print("   These stocks may not be suitable for pairs trading")
    except ImportError:
        print("Statsmodels not available for cointegration testing")

    # Calculate the average correlation over the period
    avg_corr = rolling_corr.mean().mean()
    print(f"Average Rolling Correlation: {avg_corr:.3f}")


def example_machine_learning():
    """
    Machine Learning for Price Prediction

    Modern quantitative trading increasingly uses machine learning to find
    patterns in market data. This example shows how to:
    1. Engineer features from raw price data
    2. Train an LSTM neural network to predict returns
    3. Make predictions on new data

    Why LSTM (Long Short-Term Memory)?
    - LSTMs are designed for time series data
    - They can remember long-term patterns
    - They're used by many hedge funds for price prediction

    Feature engineering is crucial:
    - Raw prices aren't enough - we need derived features
    - Technical indicators, momentum, volatility all help
    - More informative features = better predictions
    """
    print("\n=== Machine Learning Example ===")

    # Fetch Apple stock data for training
    data = ma.get_market_data(["AAPL"], start_date="2023-01-01", end_date="2024-01-01")
    prices = data["AAPL"]

    # Create a feature engineer to automatically generate predictive features
    # This will create technical indicators, momentum features, and more
    engineer = ma.FeatureEngineer()
    features = engineer.create_features(prices)

    print(f"Created {len(features.columns)} features:")
    for col in features.columns:
        print(f"  - {col}")

    # Try to use LSTM for prediction if PyTorch is available
    try:
        import torch  # noqa: F401

        print("\nPyTorch available - testing LSTM predictor...")

        # Prepare the target variable (next day's return)
        # We're trying to predict tomorrow's return based on today's features
        target = prices.pct_change().shift(-1).dropna()
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]

        if len(X) > 100:
            # Split into training and testing sets (80/20 split)
            # We train on historical data and test on recent data
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, _y_test = y[:train_size], y[train_size:]

            # Scale features to have mean=0 and std=1
            # Neural networks work better with normalized data
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train an LSTM model
            # sequence_length=10 means it looks at 10 days of history
            # epochs=5 means it goes through the training data 5 times
            predictor = ma.LSTMPredictor(sequence_length=10, epochs=5)
            predictor.fit(X_train_scaled, y_train.values)

            # Make predictions on the test set
            predictions = predictor.predict(X_test_scaled)
            print(f"LSTM model trained and made {len(predictions)} predictions")
            print(
                "In production, you'd evaluate these predictions against actual returns"
            )
        else:
            print("Not enough data for LSTM training")

    except ImportError:
        print("PyTorch not available - skipping LSTM example")
        print("Install PyTorch to use deep learning features: pip install torch")


def main():
    """
    Run All Basic Examples

    This function executes all the examples in sequence, demonstrating
    the core capabilities of MeridianAlgo. Each example is self-contained
    and shows a different aspect of quantitative finance.

    Examples covered:
    1. Portfolio Optimization - Build optimal portfolios
    2. Time Series Analysis - Analyze asset performance
    3. Risk Metrics - Measure and manage risk
    4. Statistical Arbitrage - Find trading opportunities
    5. Machine Learning - Predict future returns
    """
    print("MeridianAlgo Examples")
    print("=" * 50)

    try:
        example_portfolio_optimization()
        example_time_series_analysis()
        example_risk_metrics()
        example_statistical_arbitrage()
        example_machine_learning()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("- Check out advanced_trading_strategy.py for backtesting")
        print("- See comprehensive_examples.py for more features")
        print("- Explore quant_examples.py for professional strategies")

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Please check your internet connection and dependencies.")


if __name__ == "__main__":
    main()
