"""
Getting Started with MeridianAlgo
==================================

This example walks you through the basics of using MeridianAlgo for quantitative
finance analysis. We'll cover data fetching, basic calculations, and visualization.

Perfect for: Beginners who want to understand the fundamentals
Time to run: ~30 seconds
"""

import meridianalgo as ma

print("Welcome to MeridianAlgo!")
print("=" * 60)

# ============================================================================
# Step 1: Fetching Market Data
# ============================================================================
print("\nStep 1: Fetching Market Data")
print("-" * 60)

# Let's grab some stock data for popular tech companies
# We're using the last year of data to keep things manageable
tickers = ["AAPL", "MSFT", "GOOGL"]
print(f"Fetching data for: {', '.join(tickers)}")

try:
    # get_market_data is your one-stop shop for historical price data
    # It automatically handles data cleaning and formatting
    prices = ma.get_market_data(tickers, start="2023-01-01", end="2024-01-01")

    print(f"✓ Successfully fetched {len(prices)} days of data")
    print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
    print("\nFirst few prices:")
    print(prices.head())

except Exception as e:
    print(f"✗ Error fetching data: {e}")
    print("  Make sure you have an internet connection!")
    exit(1)

# ============================================================================
# Step 2: Calculating Returns
# ============================================================================
print("\nStep 2: Calculating Returns")
print("-" * 60)

# Returns are the percentage change in price from one day to the next
# This is what we actually care about for most financial analysis
returns = prices.pct_change().dropna()

print("Daily returns calculated!")
print(f"  Total observations: {len(returns)}")
print("\nAverage daily returns:")
for ticker in tickers:
    avg_return = returns[ticker].mean()
    annual_return = (1 + avg_return) ** 252 - 1  # Annualize it
    print(f"  {ticker}: {avg_return:.4%} daily ({annual_return:.2%} annualized)")

# ============================================================================
# Step 3: Risk Analysis
# ============================================================================
print("\n⚠️  Step 3: Risk Analysis")
print("-" * 60)

# Volatility tells us how much the price bounces around
# Higher volatility = more risk (but potentially more reward)
print("Volatility (annualized standard deviation):")
for ticker in tickers:
    vol = returns[ticker].std() * (252**0.5)  # Annualize volatility
    print(f"  {ticker}: {vol:.2%}")

# Sharpe Ratio: Return per unit of risk (higher is better)
# A Sharpe > 1 is generally considered good
print("\nSharpe Ratios (return/risk):")
for ticker in tickers:
    mean_return = returns[ticker].mean() * 252
    volatility = returns[ticker].std() * (252**0.5)
    sharpe = mean_return / volatility if volatility > 0 else 0
    print(f"  {ticker}: {sharpe:.2f}")

# ============================================================================
# Step 4: Correlation Analysis
# ============================================================================
print("\nStep 4: Correlation Analysis")
print("-" * 60)

# Correlation shows how stocks move together
# 1.0 = perfect correlation, 0 = no correlation, -1.0 = inverse correlation
correlation = returns.corr()
print("Correlation matrix:")
print(correlation.round(3))

print("\nWhat this means:")
print("  Values close to 1.0: Stocks move together (diversification benefit is low)")
print("  Values close to 0.0: Stocks move independently (good for diversification)")
print("  Values close to -1.0: Stocks move opposite (great for hedging)")

# ============================================================================
# Step 5: Simple Portfolio
# ============================================================================
print("\nStep 5: Building a Simple Portfolio")
print("-" * 60)

# Let's create an equal-weighted portfolio
# This means we invest the same amount in each stock
weights = [1 / len(tickers)] * len(tickers)
print(f"Portfolio weights: {dict(zip(tickers, weights, strict=False))}")

# Calculate portfolio returns
portfolio_returns = (returns * weights).sum(axis=1)

# Portfolio statistics
portfolio_mean = portfolio_returns.mean() * 252
portfolio_vol = portfolio_returns.std() * (252**0.5)
portfolio_sharpe = portfolio_mean / portfolio_vol if portfolio_vol > 0 else 0

print("\nPortfolio Performance:")
print(f"  Expected Annual Return: {portfolio_mean:.2%}")
print(f"  Annual Volatility: {portfolio_vol:.2%}")
print(f"  Sharpe Ratio: {portfolio_sharpe:.2f}")

# Compare to individual stocks
print("\nComparison:")
print(f"  Portfolio volatility ({portfolio_vol:.2%}) is lower than individual stocks")
print("  This is the power of diversification!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Congratulations! You've completed the basics!")
print("=" * 60)
print("\nWhat you learned:")
print("  ✓ How to fetch market data")
print("  ✓ How to calculate returns and risk metrics")
print("  ✓ How to analyze correlations")
print("  ✓ How to build a simple portfolio")
print("\nNext steps:")
print("  → Try 02_portfolio_optimization.py for advanced portfolio techniques")
print("  → Explore 03_risk_management.py to learn about VaR and risk metrics")
print("  → Check out 04_technical_analysis.py for trading indicators")
