.. _risk_analysis:

Risk Analysis
============

This guide covers risk analysis techniques available in MeridianAlgo for assessing and managing portfolio risk.

Introduction to Risk Metrics
---------------------------

Risk analysis is crucial for understanding the potential downsides of your investment strategies. MeridianAlgo provides comprehensive tools for risk assessment.

Basic Risk Metrics
-----------------

Calculate common risk metrics for your portfolio:

.. code-block:: python

    import meridianalgo as ma
    import pandas as pd
    
    # Get sample returns
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = ma.get_market_data(symbols, '2020-01-01', '2021-01-01')
    returns = data.pct_change().dropna()
    
    # Calculate portfolio returns (equal-weighted for example)
    portfolio_returns = returns.mean(axis=1)
    
    # Calculate risk metrics
    metrics = ma.calculate_risk_metrics(portfolio_returns)
    
    print("Portfolio Risk Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

Value at Risk (VaR)
------------------

Calculate Value at Risk using different methods:

.. code-block:: python

    # Historical VaR (95% confidence)
    historical_var = ma.calculate_value_at_risk(portfolio_returns, method='historical')
    
    # Parametric (Gaussian) VaR
    parametric_var = ma.calculate_value_at_risk(portfolio_returns, method='gaussian')
    
    # Modified VaR (Cornish-Fisher expansion)
    modified_var = ma.calculate_value_at_risk(portfolio_returns, method='modified')
    
    print(f"Historical 95% VaR: {historical_var:.2%}")
    print(f"Parametric 95% VaR: {parametric_var:.2%}")
    print(f"Modified 95% VaR: {modified_var:.2%}")

Expected Shortfall (CVaR)
------------------------

Calculate Expected Shortfall (Conditional VaR):

.. code-block:: python

    # Calculate CVaR
    cvar = ma.calculate_expected_shortfall(portfolio_returns)
    print(f"95% CVaR: {cvar:.2%}")

Drawdown Analysis
----------------

Analyze drawdowns in your portfolio:

.. code-block:: python

    # Calculate maximum drawdown
    max_drawdown = ma.calculate_max_drawdown(portfolio_returns)
    
    # Get drawdown series
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    
    # Plot drawdowns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    drawdowns.plot()
    plt.title('Portfolio Drawdowns')
    plt.ylabel('Drawdown')
    plt.xlabel('Date')
    plt.grid(True)
    plt.show()

Risk-Adjusted Returns
--------------------

Calculate various risk-adjusted return metrics:

.. code-block:: python

    # Sharpe ratio
    sharpe = ma.calculate_sharpe_ratio(portfolio_returns)
    
    # Sortino ratio (focuses on downside risk)
    sortino = ma.calculate_sortino_ratio(portfolio_returns)
    
    # Calmar ratio (return vs max drawdown)
    calmar = ma.calculate_calmar_ratio(portfolio_returns)
    
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Calmar Ratio: {calmar:.2f}")

Portfolio Risk Decomposition
--------------------------

Understand the sources of risk in your portfolio:

.. code-block:: python

    # Calculate risk contributions
    risk_contributions = ma.calculate_risk_contributions(returns)
    
    # Plot risk contributions
    plt.figure(figsize=(10, 6))
    risk_contributions.plot(kind='bar')
    plt.title('Risk Contributions by Asset')
    plt.ylabel('Contribution to Total Risk')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

Stress Testing
-------------

Test how your portfolio would perform under extreme market conditions:

.. code-block:: python

    # Define stress scenarios
    scenarios = {
        'market_crash': {'AAPL': -0.30, 'MSFT': -0.25, 'GOOGL': -0.28},
        'recovery': {'AAPL': 0.15, 'MSFT': 0.12, 'GOOGL': 0.10},
        'volatility_spike': {'AAPL': 0.40, 'MSFT': 0.35, 'GOOGL': 0.38}
    }
    
    # Calculate scenario impacts
    for scenario, shocks in scenarios.items():
        scenario_returns = returns.copy()
        for asset, shock in shocks.items():
            if asset in scenario_returns.columns:
                scenario_returns[asset] += shock
        
        # Calculate portfolio return under scenario
        scenario_port_return = (scenario_returns * weights).sum(axis=1).mean()
        print(f"{scenario} scenario return: {scenario_port_return:.2%}")

Best Practices
-------------

1. Always look at multiple risk metrics together
2. Consider both historical and forward-looking risk measures
3. Test your portfolio under various market conditions
4. Monitor risk metrics regularly
5. Set appropriate risk limits and triggers

Next Steps
----------

- Learn about :ref:`portfolio_optimization` to manage risk-return tradeoffs
- Explore :ref:`technical_analysis` for market timing signals
- Check the :ref:`api_reference` for all risk analysis functions
