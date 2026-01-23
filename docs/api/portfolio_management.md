# Portfolio Management API

Advanced portfolio optimization and management tools.

##  Overview

The Portfolio Management module provides comprehensive tools for:
- **Portfolio Optimization**: Modern Portfolio Theory, Black-Litterman, Risk Parity
- **Risk Management**: VaR, Expected Shortfall, Maximum Drawdown
- **Performance Analysis**: Attribution analysis, benchmark comparison
- **Rebalancing**: Calendar and threshold-based rebalancing strategies

##  Quick Start

```python
import meridianalgo as ma
import pandas as pd
import numpy as np

# Get market data
data = ma.get_market_data(['AAPL', 'MSFT', 'GOOGL', 'TSLA'], start_date='2023-01-01')
returns = data.pct_change().dropna()

# Create portfolio optimizer
optimizer = ma.PortfolioOptimizer(returns)

# Optimize portfolio
optimal_portfolio = optimizer.optimize_portfolio(objective='sharpe')
print(f"Optimal weights: {optimal_portfolio['weights']}")
print(f"Expected return: {optimal_portfolio['return']:.2%}")
print(f"Volatility: {optimal_portfolio['volatility']:.2%}")
```

##  Portfolio Optimization

### PortfolioOptimizer

```python
optimizer = ma.PortfolioOptimizer(returns, risk_free_rate=0.02)
```

**Parameters:**
- `returns` (pd.DataFrame): Asset returns
- `risk_free_rate` (float): Risk-free rate (default: 0.02)

**Methods:**

#### Calculate Efficient Frontier

```python
frontier_data = optimizer.calculate_efficient_frontier(num_portfolios=1000)
```

**Parameters:**
- `num_portfolios` (int): Number of portfolios to generate (default: 1000)

**Returns:**
- `dict`: Dictionary with frontier data
  - `volatility`: Portfolio volatilities
  - `returns`: Portfolio returns
  - `sharpe`: Sharpe ratios
  - `weights`: Portfolio weights

**Example:**
```python
# Calculate efficient frontier
frontier = optimizer.calculate_efficient_frontier(num_portfolios=1000)

# Plot efficient frontier
import matplotlib.pyplot as plt
plt.scatter(frontier['volatility'], frontier['returns'], c=frontier['sharpe'], cmap='viridis')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.show()
```

#### Optimize Portfolio

```python
optimal = optimizer.optimize_portfolio(objective='sharpe', constraints=None)
```

**Parameters:**
- `objective` (str): Optimization objective ('sharpe', 'min_vol', 'max_return')
- `constraints` (dict): Additional constraints (optional)

**Returns:**
- `dict`: Optimal portfolio
  - `weights`: Optimal weights
  - `return`: Expected return
  - `volatility`: Portfolio volatility
  - `sharpe_ratio`: Sharpe ratio

**Example:**
```python
# Optimize for maximum Sharpe ratio
optimal_sharpe = optimizer.optimize_portfolio(objective='sharpe')

# Optimize for minimum volatility
optimal_min_vol = optimizer.optimize_portfolio(objective='min_vol')

# Optimize for maximum return
optimal_max_return = optimizer.optimize_portfolio(objective='max_return')

print(f"Sharpe-optimal weights: {optimal_sharpe['weights']}")
print(f"Min-vol weights: {optimal_min_vol['weights']}")
print(f"Max-return weights: {optimal_max_return['weights']}")
```

### EfficientFrontier

```python
frontier = ma.EfficientFrontier(returns)
```

**Parameters:**
- `returns` (pd.DataFrame): Asset returns

**Methods:**

#### Calculate Frontier

```python
frontier_data = frontier.calculate_frontier(target_returns)
```

**Parameters:**
- `target_returns` (np.ndarray): Array of target returns

**Returns:**
- `dict`: Frontier data
  - `weights`: Portfolio weights for each target return
  - `volatilities`: Portfolio volatilities
  - `returns`: Target returns

**Example:**
```python
# Create target returns
target_returns = np.linspace(0.05, 0.25, 20)

# Calculate frontier
frontier_data = frontier.calculate_frontier(target_returns)

# Plot frontier
plt.plot(frontier_data['volatilities'], frontier_data['returns'])
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.show()
```

### BlackLitterman

```python
bl_model = ma.BlackLitterman(returns, market_caps=None)
```

**Parameters:**
- `returns` (pd.DataFrame): Asset returns
- `market_caps` (pd.Series): Market capitalization weights (optional)

**Methods:**

#### Calculate Implied Returns

```python
implied_returns = bl_model.calculate_implied_returns(risk_aversion=3)
```

**Parameters:**
- `risk_aversion` (float): Risk aversion parameter (default: 3)

**Returns:**
- `pd.Series`: Implied equilibrium returns

#### Optimize with Views

```python
bl_portfolio = bl_model.optimize_with_views(views, confidence=0.1)
```

**Parameters:**
- `views` (dict): View specifications
- `confidence` (float): Confidence level for views (default: 0.1)

**Returns:**
- `dict`: Black-Litterman portfolio
  - `weights`: Portfolio weights
  - `implied_returns`: Implied returns

**Example:**
```python
# Define views
views = {
    'AAPL': 0.15,  # 15% expected return
    'MSFT': 0.12,  # 12% expected return
}

# Optimize with views
bl_portfolio = bl_model.optimize_with_views(views)
print(f"BL weights: {bl_portfolio['weights']}")
```

### RiskParity

```python
rp_optimizer = ma.RiskParity(returns)
```

**Parameters:**
- `returns` (pd.DataFrame): Asset returns

**Methods:**

#### Optimize

```python
rp_portfolio = rp_optimizer.optimize()
```

**Returns:**
- `dict`: Risk parity portfolio
  - `weights`: Portfolio weights
  - `risk_contributions`: Risk contributions

**Example:**
```python
# Optimize risk parity portfolio
rp_portfolio = rp_optimizer.optimize()

print(f"Risk parity weights: {rp_portfolio['weights']}")
print(f"Risk contributions: {rp_portfolio['risk_contributions']}")
```

##  Risk Management

### RiskManager

```python
risk_manager = ma.RiskManager(portfolio_returns)
```

**Parameters:**
- `portfolio_returns` (pd.Series): Portfolio returns

**Methods:**

#### Calculate VaR

```python
var = risk_manager.calculate_var(confidence_level=0.95)
```

**Parameters:**
- `confidence_level` (float): Confidence level (default: 0.95)

**Returns:**
- `float`: Value at Risk

#### Calculate Expected Shortfall

```python
es = risk_manager.calculate_expected_shortfall(confidence_level=0.95)
```

**Parameters:**
- `confidence_level` (float): Confidence level (default: 0.95)

**Returns:**
- `float`: Expected Shortfall

### VaRCalculator

```python
var_calc = ma.VaRCalculator(returns)
```

**Parameters:**
- `returns` (pd.Series): Asset returns

**Methods:**

#### Historical VaR

```python
var_hist = var_calc.historical_var(confidence_level=0.95)
```

**Parameters:**
- `confidence_level` (float): Confidence level (default: 0.95)

**Returns:**
- `float`: Historical VaR

#### Parametric VaR

```python
var_param = var_calc.parametric_var(confidence_level=0.95)
```

**Parameters:**
- `confidence_level` (float): Confidence level (default: 0.95)

**Returns:**
- `float`: Parametric VaR

#### Monte Carlo VaR

```python
var_mc = var_calc.monte_carlo_var(confidence_level=0.95, n_simulations=10000)
```

**Parameters:**
- `confidence_level` (float): Confidence level (default: 0.95)
- `n_simulations` (int): Number of simulations (default: 10000)

**Returns:**
- `float`: Monte Carlo VaR

### StressTester

```python
stress_tester = ma.StressTester(portfolio_returns)
```

**Parameters:**
- `portfolio_returns` (pd.Series): Portfolio returns

**Methods:**

#### Historical Stress Test

```python
stress_results = stress_tester.historical_stress_test(stress_periods)
```

**Parameters:**
- `stress_periods` (list): List of stress period names

**Returns:**
- `dict`: Stress test results

#### Scenario Stress Test

```python
scenario_results = stress_tester.scenario_stress_test(scenarios)
```

**Parameters:**
- `scenarios` (dict): Scenario specifications

**Returns:**
- `dict`: Scenario test results

**Example:**
```python
# Define stress scenarios
scenarios = {
    'market_crash': -0.20,  # 20% market decline
    'volatility_spike': 0.50,  # 50% volatility increase
    'liquidity_crisis': -0.15,  # 15% liquidity impact
}

# Run stress tests
stress_results = stress_tester.scenario_stress_test(scenarios)
print(f"Stress test results: {stress_results}")
```

##  Performance Analysis

### PerformanceAnalyzer

```python
analyzer = ma.PerformanceAnalyzer(portfolio_returns, benchmark_returns=None)
```

**Parameters:**
- `portfolio_returns` (pd.Series): Portfolio returns
- `benchmark_returns` (pd.Series): Benchmark returns (optional)

**Methods:**

#### Calculate Metrics

```python
metrics = analyzer.calculate_metrics()
```

**Returns:**
- `dict`: Performance metrics
  - `total_return`: Total return
  - `annualized_return`: Annualized return
  - `volatility`: Annualized volatility
  - `sharpe_ratio`: Sharpe ratio
  - `max_drawdown`: Maximum drawdown

**Example:**
```python
# Calculate performance metrics
metrics = analyzer.calculate_metrics()

print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Annualized Return: {metrics['annualized_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### AttributionAnalysis

```python
attribution = ma.AttributionAnalysis(portfolio_returns, factor_returns)
```

**Parameters:**
- `portfolio_returns` (pd.Series): Portfolio returns
- `factor_returns` (pd.DataFrame): Factor returns

**Methods:**

#### Analyze Attribution

```python
attribution_results = attribution.analyze_attribution()
```

**Returns:**
- `dict`: Attribution analysis results
  - `factor_contribution`: Factor contribution
  - `residual_return`: Residual return

### BenchmarkComparison

```python
comparison = ma.BenchmarkComparison(portfolio_returns, benchmark_returns)
```

**Parameters:**
- `portfolio_returns` (pd.Series): Portfolio returns
- `benchmark_returns` (pd.Series): Benchmark returns

**Methods:**

#### Compare Performance

```python
comparison_results = comparison.compare_performance()
```

**Returns:**
- `dict`: Comparison results
  - `excess_return`: Excess return
  - `tracking_error`: Tracking error
  - `information_ratio`: Information ratio

##  Rebalancing

### Rebalancer

```python
rebalancer = ma.Rebalancer(target_weights)
```

**Parameters:**
- `target_weights` (dict): Target portfolio weights

**Methods:**

#### Rebalance

```python
new_weights = rebalancer.rebalance(current_weights)
```

**Parameters:**
- `current_weights` (dict): Current portfolio weights

**Returns:**
- `dict`: New portfolio weights

### CalendarRebalancer

```python
calendar_rebalancer = ma.CalendarRebalancer(target_weights, frequency='monthly')
```

**Parameters:**
- `target_weights` (dict): Target portfolio weights
- `frequency` (str): Rebalancing frequency ('monthly', 'quarterly', 'annually')

**Methods:**

#### Should Rebalance

```python
should_rebalance = calendar_rebalancer.should_rebalance(last_rebalance)
```

**Parameters:**
- `last_rebalance` (datetime): Last rebalancing date

**Returns:**
- `bool`: Whether to rebalance

### ThresholdRebalancer

```python
threshold_rebalancer = ma.ThresholdRebalancer(target_weights, threshold=0.05)
```

**Parameters:**
- `target_weights` (dict): Target portfolio weights
- `threshold` (float): Rebalancing threshold (default: 0.05)

**Methods:**

#### Should Rebalance

```python
should_rebalance = threshold_rebalancer.should_rebalance(current_weights)
```

**Parameters:**
- `current_weights` (dict): Current portfolio weights

**Returns:**
- `bool`: Whether to rebalance

##  Complete Example

```python
import meridianalgo as ma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def portfolio_management_example():
    """Complete portfolio management example."""
    
    # Get market data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    data = ma.get_market_data(symbols, start_date='2023-01-01')
    returns = data.pct_change().dropna()
    
    print(" Portfolio Management Example")
    print("=" * 50)
    
    # 1. Portfolio Optimization
    print("\n Portfolio Optimization...")
    optimizer = ma.PortfolioOptimizer(returns)
    
    # Calculate efficient frontier
    frontier = optimizer.calculate_efficient_frontier(num_portfolios=1000)
    
    # Optimize for different objectives
    optimal_sharpe = optimizer.optimize_portfolio(objective='sharpe')
    optimal_min_vol = optimizer.optimize_portfolio(objective='min_vol')
    optimal_max_return = optimizer.optimize_portfolio(objective='max_return')
    
    print(f"Sharpe-optimal weights: {optimal_sharpe['weights']}")
    print(f"Min-vol weights: {optimal_min_vol['weights']}")
    print(f"Max-return weights: {optimal_max_return['weights']}")
    
    # 2. Risk Analysis
    print("\n Risk Analysis...")
    portfolio_returns = returns.mean(axis=1)
    
    # Calculate VaR
    var_95 = ma.calculate_value_at_risk(portfolio_returns, confidence_level=0.95)
    var_99 = ma.calculate_value_at_risk(portfolio_returns, confidence_level=0.99)
    
    # Calculate Expected Shortfall
    es_95 = ma.calculate_expected_shortfall(portfolio_returns, confidence_level=0.95)
    
    # Calculate Maximum Drawdown
    max_dd = ma.calculate_max_drawdown(portfolio_returns)
    
    print(f"95% VaR: {var_95:.2%}")
    print(f"99% VaR: {var_99:.2%}")
    print(f"95% ES: {es_95:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    
    # 3. Performance Analysis
    print("\n Performance Analysis...")
    analyzer = ma.PerformanceAnalyzer(portfolio_returns)
    metrics = analyzer.calculate_metrics()
    
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # 4. Rebalancing
    print("\n Rebalancing...")
    target_weights = dict(zip(symbols, optimal_sharpe['weights']))
    rebalancer = ma.Rebalancer(target_weights)
    
    # Simulate current weights (with drift)
    current_weights = {symbol: weight * (1 + np.random.normal(0, 0.05)) 
                      for symbol, weight in target_weights.items()}
    current_weights = {k: v/sum(current_weights.values()) 
                      for k, v in current_weights.items()}
    
    # Check if rebalancing is needed
    threshold_rebalancer = ma.ThresholdRebalancer(target_weights, threshold=0.05)
    should_rebalance = threshold_rebalancer.should_rebalance(current_weights)
    
    print(f"Current weights: {current_weights}")
    print(f"Target weights: {target_weights}")
    print(f"Should rebalance: {should_rebalance}")
    
    if should_rebalance:
        new_weights = rebalancer.rebalance(current_weights)
        print(f"New weights: {new_weights}")
    
    # 5. Visualization
    print("\n Creating visualizations...")
    
    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(frontier['volatility'], frontier['returns'], c=frontier['sharpe'], cmap='viridis', alpha=0.6)
    plt.scatter(optimal_sharpe['volatility'], optimal_sharpe['return'], color='red', s=100, label='Sharpe Optimal')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.colorbar(label='Sharpe Ratio')
    plt.legend()
    
    # Plot portfolio weights
    plt.subplot(2, 2, 2)
    weights_data = [optimal_sharpe['weights'], optimal_min_vol['weights'], optimal_max_return['weights']]
    x = np.arange(len(symbols))
    width = 0.25
    
    plt.bar(x - width, weights_data[0], width, label='Sharpe Optimal')
    plt.bar(x, weights_data[1], width, label='Min Vol')
    plt.bar(x + width, weights_data[2], width, label='Max Return')
    
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.title('Portfolio Weights Comparison')
    plt.xticks(x, symbols)
    plt.legend()
    
    # Plot cumulative returns
    plt.subplot(2, 2, 3)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    plt.plot(cumulative_returns.index, cumulative_returns.values)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Portfolio Cumulative Returns')
    plt.xticks(rotation=45)
    
    # Plot drawdown
    plt.subplot(2, 2, 4)
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    plt.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.3, color='red')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Portfolio Drawdown')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\n Portfolio management example completed successfully!")

# Run example
if __name__ == "__main__":
    portfolio_management_example()
```

##  Additional Resources

- [Technical Indicators API](technical_indicators.md) - Technical analysis
- [Risk Analysis API](risk_analysis.md) - Risk metrics and analysis
- [Examples](../examples/) - Practical use cases
- [Performance Benchmarks](../benchmarks.md) - Performance metrics
