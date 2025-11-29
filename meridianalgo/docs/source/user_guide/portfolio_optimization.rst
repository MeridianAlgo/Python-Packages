.. _portfolio_optimization:

Portfolio Optimization
=====================

This guide explains how to use MeridianAlgo for portfolio optimization, including various optimization techniques and strategies.

Introduction
-----------

Portfolio optimization is the process of selecting the best portfolio (asset distribution) out of the set of all portfolios being considered, according to some objective. The objective typically maximizes factors like expected return while minimizing risk.

Basic Portfolio Optimization
---------------------------

Here's how to perform basic mean-variance optimization:

.. code-block:: python

    import meridianalgo as ma
    import numpy as np
    import pandas as pd
    
    # Get historical returns
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    data = ma.get_market_data(symbols, start_date='2020-01-01', end_date='2021-01-01')
    returns = data.pct_change().dropna()
    
    # Initialize the optimizer
    optimizer = ma.PortfolioOptimizer(returns)
    
    # Optimize for maximum Sharpe ratio (risk-adjusted return)
    weights = optimizer.optimize_portfolio(method='sharpe')
    
    # Display optimal weights
    print("Optimal Portfolio Weights:")
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.2%}")

Available Optimization Methods
----------------------------

MeridianAlgo supports several optimization methods:

1. **Maximum Sharpe Ratio** (``'sharpe'``)
   - Maximizes the risk-adjusted return
   - Default method if none specified

2. **Minimum Volatility** (``'min_vol'``)
   - Minimizes portfolio volatility
   - Good for risk-averse investors

3. **Efficient Risk** (``'efficient_risk'``)
   - Maximizes return for a given target risk
   - Requires ``target_volatility`` parameter

4. **Efficient Return** (``'efficient_return'``)
   - Minimizes risk for a given target return
   - Requires ``target_return`` parameter

5. **Risk Parity** (``'risk_parity'``)
   - Allocates risk equally among assets
   - Good for diversification

Example: Using Different Methods
------------------------------

.. code-block:: python

    # Minimum volatility portfolio
    min_vol_weights = optimizer.optimize_portfolio(method='min_vol')
    
    # Efficient risk portfolio (target 15% annualized volatility)
    eff_risk_weights = optimizer.optimize_portfolio(
        method='efficient_risk',
        target_volatility=0.15
    )
    
    # Risk parity portfolio
    rp_weights = optimizer.optimize_portfolio(method='risk_parity')

Constraints
----------

You can add constraints to your optimization:

.. code-block:: python

    # Optimization with constraints
    constraints = {
        'AAPL': (0.1, 0.3),  # 10% to 30% allocation
        'MSFT': (0.05, 0.25),  # 5% to 25% allocation
        'max_sector': {
            'Technology': 0.6,  # Max 60% in Tech
            'Finance': 0.4      # Max 40% in Finance
        }
    }
    
    constrained_weights = optimizer.optimize_portfolio(
        method='sharpe',
        constraints=constraints
    )

Transaction Costs
----------------

Account for transaction costs in your optimization:

.. code-block:: python

    # With transaction costs (as a percentage)
    weights_with_costs = optimizer.optimize_portfolio(
        method='sharpe',
        transaction_costs=0.001  # 0.1% transaction cost
    )

Backtesting Optimized Portfolios
------------------------------

Test how your optimized portfolio would have performed:

.. code-block:: python

    # Backtest the optimized portfolio
    backtest_returns = (returns * pd.Series(weights)).sum(axis=1)
    
    # Calculate performance metrics
    metrics = ma.calculate_risk_metrics(backtest_returns)
    
    print("\nPortfolio Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

Advanced Topics
--------------

### Black-Litterman Model

Incorporate your market views:

.. code-block:: python

    # Define your market views
    views = {
        'AAPL': 0.05,  # 5% expected return for AAPL
        'MSFT': 0.03   # 3% expected return for MSFT
    }
    
    # Confidence in your views (lower = more confidence)
    view_confidences = {
        'AAPL': 0.5,
        'MSFT': 0.3
    }
    
    bl_weights = optimizer.black_litterman_optimization(
        views=views,
        view_confidences=view_confidences,
        risk_aversion=1.0
    )

### Hierarchical Risk Parity

For more robust optimization:

.. code-block:: python

    hrp_weights = optimizer.hierarchical_risk_parity()

Best Practices
-------------

1. Use out-of-sample testing to validate your optimization
2. Consider rebalancing frequency
3. Account for transaction costs and taxes
4. Use appropriate risk models
5. Consider regime changes in market conditions

Next Steps
----------

- Learn about :ref:`risk_analysis` to better understand portfolio risk
- Explore :ref:`backtesting` to test your strategies
- Check the :ref:`api_reference` for all available optimization options
