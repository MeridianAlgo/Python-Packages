"""
Basic usage examples for MeridianAlgo package.
"""

import numpy as np

# Import MeridianAlgo modules
import meridianalgo as ma


def example_portfolio_optimization():
    """Example of portfolio optimization."""
    print("=== Portfolio Optimization Example ===")

    # Get market data for multiple assets
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    data = ma.get_market_data(tickers, start_date="2023-01-01", end_date="2024-01-01")

    # Calculate returns
    returns = data.pct_change().dropna()
    print(f"Retrieved data for {len(returns)} days")
    print(f"Assets: {list(returns.columns)}")

    # Create portfolio optimizer
    optimizer = ma.PortfolioOptimizer(returns)

    # Calculate efficient frontier
    frontier = optimizer.calculate_efficient_frontier(num_portfolios=1000)

    # Find optimal portfolio (highest Sharpe ratio)
    max_sharpe_idx = np.argmax(frontier["sharpe"])
    optimal_weights = frontier["weights"][max_sharpe_idx]

    print("\nOptimal Portfolio Weights:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {optimal_weights[i]:.2%}")

    print(f"Expected Return: {frontier['returns'][max_sharpe_idx]:.2%}")
    print(f"Volatility: {frontier['volatility'][max_sharpe_idx]:.2%}")
    print(f"Sharpe Ratio: {frontier['sharpe'][max_sharpe_idx]:.2f}")


def example_time_series_analysis():
    """Example of time series analysis."""
    print("\n=== Time Series Analysis Example ===")

    # Get data for a single asset
    data = ma.get_market_data(["AAPL"], start_date="2023-01-01", end_date="2024-01-01")
    prices = data["AAPL"]

    # Create time series analyzer
    analyzer = ma.TimeSeriesAnalyzer(prices)

    # Calculate returns and volatility
    returns = analyzer.calculate_returns()
    analyzer.calculate_volatility(window=21, annualized=True)

    # Calculate performance metrics
    metrics = ma.calculate_metrics(returns)

    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")


def example_risk_metrics():
    """Example of risk metrics calculation."""
    print("\n=== Risk Metrics Example ===")

    # Get data
    data = ma.get_market_data(["AAPL"], start_date="2023-01-01", end_date="2024-01-01")
    returns = data["AAPL"].pct_change().dropna()

    # Calculate risk metrics
    var_95 = ma.calculate_value_at_risk(returns, confidence_level=0.95)
    var_99 = ma.calculate_value_at_risk(returns, confidence_level=0.99)
    es_95 = ma.calculate_expected_shortfall(returns, confidence_level=0.95)

    print(f"95% Value at Risk: {var_95:.2%}")
    print(f"99% Value at Risk: {var_99:.2%}")
    print(f"95% Expected Shortfall: {es_95:.2%}")

    # Calculate Hurst exponent
    hurst = ma.hurst_exponent(returns)
    print(f"Hurst Exponent: {hurst:.3f}")

    if hurst > 0.5:
        print("   Series shows trending behavior")
    elif hurst < 0.5:
        print("   Series shows mean-reverting behavior")
    else:
        print("   Series shows random walk behavior")


def example_statistical_arbitrage():
    """Example of statistical arbitrage analysis."""
    print("\n=== Statistical Arbitrage Example ===")

    # Get data for two correlated assets
    data = ma.get_market_data(
        ["AAPL", "MSFT"], start_date="2023-01-01", end_date="2024-01-01"
    )

    # Create statistical arbitrage analyzer
    arb = ma.StatisticalArbitrage(data)

    # Calculate rolling correlation
    rolling_corr = arb.calculate_rolling_correlation(window=30)

    # Test for cointegration
    try:
        coint_result = arb.test_cointegration(data["AAPL"], data["MSFT"])
        print(f"Cointegration Test Statistic: {coint_result['test_statistic']:.3f}")
        print(f"P-value: {coint_result['p_value']:.3f}")
        print(f"Cointegrated: {coint_result['is_cointegrated']}")
    except ImportError:
        print("Statsmodels not available for cointegration testing")

    # Calculate average correlation
    avg_corr = rolling_corr.mean().mean()
    print(f"Average Rolling Correlation: {avg_corr:.3f}")


def example_machine_learning():
    """Example of machine learning features."""
    print("\n=== Machine Learning Example ===")

    # Get data
    data = ma.get_market_data(["AAPL"], start_date="2023-01-01", end_date="2024-01-01")
    prices = data["AAPL"]

    # Create feature engineer
    engineer = ma.FeatureEngineer()
    features = engineer.create_features(prices)

    print(f"Created {len(features.columns)} features:")
    for col in features.columns:
        print(f"  - {col}")

    # Check if PyTorch is available for LSTM
    try:
        import torch  # noqa: F401

        print("\nPyTorch available - testing LSTM predictor...")

        # Prepare data for LSTM
        target = prices.pct_change().shift(-1).dropna()
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]

        if len(X) > 100:
            # Split data
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, _y_test = y[:train_size], y[train_size:]

            # Scale features
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train LSTM model
            predictor = ma.LSTMPredictor(sequence_length=10, epochs=5)
            predictor.fit(X_train_scaled, y_train.values)

            # Make predictions
            predictions = predictor.predict(X_test_scaled)
            print(f"LSTM model trained and made {len(predictions)} predictions")
        else:
            print("Not enough data for LSTM training")

    except ImportError:
        print("PyTorch not available - skipping LSTM example")


def main():
    """Run all examples."""
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

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Please check your internet connection and dependencies.")


if __name__ == "__main__":
    main()
