.. _backtesting:

Backtesting
===========

This guide covers backtesting strategies and evaluating trading performance using MeridianAlgo.

Introduction
-----------

Backtesting is the process of testing a trading strategy on historical data to evaluate its performance. MeridianAlgo provides comprehensive backtesting tools with realistic market conditions.

Basic Backtesting
-----------------

Let's create a simple moving average crossover strategy:

.. code-block:: python

    import meridianalgo as ma
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Get historical data
    symbol = 'AAPL'
    data = ma.get_market_data([symbol], start_date='2019-01-01', end_date='2021-01-01')
    prices = data[symbol]
    
    # Calculate moving averages
    short_ma = ma.calculate_sma(prices, window=20)
    long_ma = ma.calculate_sma(prices, window=50)
    
    # Generate trading signals
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices
    signals['short_ma'] = short_ma
    signals['long_ma'] = long_ma
    
    # Buy signal: short MA crosses above long MA
    signals['signal'] = 0
    signals.loc[short_ma > long_ma, 'signal'] = 1
    
    # Calculate positions
    signals['position'] = signals['signal'].shift(1)  # Trade on next day
    
    # Calculate returns
    signals['returns'] = signals['price'].pct_change()
    signals['strategy_returns'] = signals['position'] * signals['returns']
    
    # Calculate cumulative returns
    signals['cumulative_returns'] = (1 + signals['returns']).cumprod()
    signals['cumulative_strategy'] = (1 + signals['strategy_returns']).cumprod()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    signals['cumulative_returns'].plot(label='Buy and Hold')
    signals['cumulative_strategy'].plot(label='MA Crossover Strategy')
    plt.title('Strategy Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

Performance Metrics
------------------

Calculate comprehensive performance metrics:

.. code-block:: python

    # Calculate performance metrics
    strategy_returns = signals['strategy_returns'].dropna()
    
    metrics = ma.calculate_risk_metrics(strategy_returns)
    
    # Additional metrics
    total_return = signals['cumulative_strategy'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility
    
    # Maximum drawdown
    cumulative = signals['cumulative_strategy']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate and profit factor
    winning_trades = strategy_returns[strategy_returns > 0]
    losing_trades = strategy_returns[strategy_returns < 0]
    
    win_rate = len(winning_trades) / len(strategy_returns)
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    print("Strategy Performance Metrics:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")

Advanced Backtesting
--------------------

### Including Transaction Costs

.. code-block:: python

    class Backtester:
        def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0001):
            self.initial_capital = initial_capital
            self.commission = commission
            self.slippage = slippage
            self.reset()
        
        def reset(self):
            self.capital = self.initial_capital
            self.position = 0
            self.trades = []
            self.portfolio_value = []
        
        def execute_trade(self, signal, price):
            """Execute trade with costs"""
            if signal == 1 and self.position == 0:  # Buy
                shares = self.capital / (price * (1 + self.slippage))
                cost = shares * price * self.commission
                self.position = shares
                self.capital -= shares * price * (1 + self.slippage) + cost
                self.trades.append(('buy', price, shares))
                
            elif signal == -1 and self.position > 0:  # Sell
                proceeds = self.position * price * (1 - self.slippage)
                cost = proceeds * self.commission
                self.capital += proceeds - cost
                self.trades.append(('sell', price, self.position))
                self.position = 0
        
        def run_backtest(self, signals, prices):
            """Run backtest on signal series"""
            self.reset()
            
            for i in range(1, len(signals)):
                if signals[i] != signals[i-1]:  # Signal change
                    if signals[i] == 1:
                        self.execute_trade(1, prices[i])
                    elif signals[i] == -1:
                        self.execute_trade(-1, prices[i])
                
                # Calculate portfolio value
                portfolio_val = self.capital + self.position * prices[i]
                self.portfolio_value.append(portfolio_val)
            
            return self.portfolio_value
    
    # Run backtest with costs
    bt = Backtester(initial_capital=100000, commission=0.001, slippage=0.0001)
    portfolio_values = bt.run_backtest(signals['signal'].values, prices.values)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Strategy with Costs')
    plt.axhline(y=100000, color='r', linestyle='--', label='Initial Capital')
    plt.title('Backtest with Transaction Costs')
    plt.legend()
    plt.grid(True)
    plt.show()

Walk-Forward Analysis
--------------------

Test strategy stability over different time periods:

.. code-block:: python

    def walk_forward_analysis(prices, window_size=252, step_size=63):
        """Perform walk-forward analysis"""
        results = []
        
        for i in range(window_size, len(prices) - step_size, step_size):
            # Training period
            train_start = i - window_size
            train_end = i
            
            # Test period
            test_start = i
            test_end = min(i + step_size, len(prices))
            
            # Get data
            train_prices = prices.iloc[train_start:train_end]
            test_prices = prices.iloc[test_start:test_end]
            
            # Optimize parameters on training data
            # Example: optimize MA windows
            best_sharpe = -float('inf')
            best_params = None
            
            for short_window in [10, 20, 30]:
                for long_window in [40, 50, 60]:
                    if short_window < long_window:
                        # Calculate strategy on training data
                        short_ma = ma.calculate_sma(train_prices, short_window)
                        long_ma = ma.calculate_sma(train_prices, long_window)
                        
                        # Generate signals
                        signals = (short_ma > long_ma).astype(int)
                        returns = train_prices.pct_change().dropna()
                        strategy_returns = signals.shift(1) * returns
                        
                        # Calculate Sharpe ratio
                        if len(strategy_returns) > 0:
                            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_params = (short_window, long_window)
            
            # Test on out-of-sample data
            if best_params:
                short_window, long_window = best_params
                short_ma = ma.calculate_sma(test_prices, short_window)
                long_ma = ma.calculate_sma(test_prices, long_window)
                signals = (short_ma > long_ma).astype(int)
                returns = test_prices.pct_change().dropna()
                strategy_returns = signals.shift(1) * returns
                
                if len(strategy_returns) > 0:
                    test_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                    results.append({
                        'period': f"{test_prices.index[0].date()} to {test_prices.index[-1].date()}",
                        'train_sharpe': best_sharpe,
                        'test_sharpe': test_sharpe,
                        'params': best_params
                    })
        
        return pd.DataFrame(results)
    
    # Run walk-forward analysis
    wf_results = walk_forward_analysis(prices)
    print(wf_results)

Monte Carlo Simulation
---------------------

Test strategy robustness with random variations:

.. code-block:: python

    def monte_carlo_backtest(strategy_func, prices, n_simulations=1000, noise_level=0.01):
        """Monte Carlo simulation of strategy performance"""
        results = []
        
        for i in range(n_simulations):
            # Add noise to prices
            noise = np.random.normal(0, noise_level, len(prices))
            noisy_prices = prices * (1 + noise)
            
            # Run strategy
            returns = strategy_func(noisy_prices)
            
            # Calculate metrics
            if len(returns) > 0:
                total_return = (1 + returns).prod() - 1
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                max_dd = ma.calculate_max_drawdown(returns)
                
                results.append({
                    'total_return': total_return,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd
                })
        
        return pd.DataFrame(results)
    
    # Example strategy function
    def ma_crossover_strategy(prices):
        short_ma = ma.calculate_sma(prices, 20)
        long_ma = ma.calculate_sma(prices, 50)
        signals = (short_ma > long_ma).astype(int)
        returns = prices.pct_change().dropna()
        return signals.shift(1) * returns
    
    # Run Monte Carlo simulation
    mc_results = monte_carlo_backtest(ma_crossover_strategy, prices, n_simulations=100)
    
    # Plot distribution of returns
    plt.figure(figsize=(12, 6))
    plt.hist(mc_results['total_return'], bins=30, alpha=0.7)
    plt.title('Distribution of Strategy Returns (Monte Carlo)')
    plt.xlabel('Total Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    print(f"Mean Return: {mc_results['total_return'].mean():.2%}")
    print(f"Std Dev: {mc_results['total_return'].std():.2%}")
    print(f"5th Percentile: {mc_results['total_return'].quantile(0.05):.2%}")
    print(f"95th Percentile: {mc_results['total_return'].quantile(0.95):.2%}")

Best Practices
-------------

1. Always include transaction costs and slippage
2. Use out-of-sample testing
3. Perform walk-forward analysis
4. Test strategy robustness with Monte Carlo
5. Be aware of look-ahead bias
6. Consider market regime changes
7. Use proper risk management

Next Steps
----------

- Learn about :ref:`portfolio_optimization` with backtested strategies
- Explore :ref:`risk_analysis` for better risk management
- Check the :ref:`api_reference` for all backtesting functions
