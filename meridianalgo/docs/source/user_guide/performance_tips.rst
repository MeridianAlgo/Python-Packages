.. _performance_tips:

Performance Tips
================

This guide provides tips and best practices to optimize the performance of your MeridianAlgo applications.

Introduction
-----------

When working with financial data and complex calculations, performance can become a bottleneck. This guide covers various techniques to speed up your code and use resources efficiently.

Vectorization
-------------

Use vectorized operations instead of loops:

.. code-block:: python

    import meridianalgo as ma
    import numpy as np
    import pandas as pd
    
    # Get data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    data = ma.get_market_data(symbols, start_date='2020-01-01', end_date='2021-01-01')
    returns = data.pct_change().dropna()
    
    # BAD: Using loops
    def calculate_correlation_slow(returns):
        n_assets = returns.shape[1]
        corr_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                corr_matrix[i, j] = returns.iloc[:, i].corr(returns.iloc[:, j])
        
        return corr_matrix
    
    # GOOD: Vectorized
    def calculate_correlation_fast(returns):
        return returns.corr().values
    
    # Time comparison
    import time
    
    start = time.time()
    corr_slow = calculate_correlation_slow(returns)
    slow_time = time.time() - start
    
    start = time.time()
    corr_fast = calculate_correlation_fast(returns)
    fast_time = time.time() - start
    
    print(f"Loop method: {slow_time:.3f} seconds")
    print(f"Vectorized method: {fast_time:.3f} seconds")
    print(f"Speedup: {slow_time/fast_time:.1f}x")

Efficient Data Structures
------------------------

Choose the right data structures for your use case:

.. code-block:: python

    # Use appropriate data types
    # BAD: Using float64 when float32 is sufficient
    data_float64 = data.astype(np.float64)
    
    # GOOD: Use float32 for price data
    data_float32 = data.astype(np.float32)
    
    # Memory usage comparison
    print(f"Float64 memory: {data_float64.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Float32 memory: {data_float32.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Use categorical for repeated strings
    # BAD: String columns
    symbols_list = symbols * 1000
    symbols_series = pd.Series(symbols_list)
    
    # GOOD: Categorical
    symbols_categorical = symbols_series.astype('category')
    
    print(f"String series memory: {symbols_series.memory_usage() / 1024:.2f} KB")
    print(f"Categorical memory: {symbols_categorical.memory_usage() / 1024:.2f} KB")

Caching Results
--------------

Cache expensive computations:

.. code-block:: python

    from functools import lru_cache
    import joblib
    import os
    
    # Example: Cache expensive calculations
    @lru_cache(maxsize=128)
    def calculate_returns_cached(symbol, start_date, end_date):
        data = ma.get_market_data([symbol], start_date, end_date)
        return data[symbol].pct_change().dropna()
    
    # Disk caching for larger datasets
    def get_or_calculate_features(symbol, start_date, end_date, cache_dir='cache'):
        cache_file = f"{cache_dir}/{symbol}_{start_date}_{end_date}.pkl"
        
        if os.path.exists(cache_file):
            return joblib.load(cache_file)
        
        # Calculate features
        data = ma.get_market_data([symbol], start_date, end_date)
        prices = data[symbol]
        
        features = pd.DataFrame()
        features['returns'] = prices.pct_change()
        features['sma_20'] = ma.calculate_sma(prices, 20)
        features['rsi'] = ma.calculate_rsi(prices)
        
        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(features, cache_file)
        
        return features

Parallel Processing
------------------

Use parallel processing for independent computations:

.. code-block:: python

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from multiprocessing import cpu_count
    
    # Parallel data loading
    def load_symbol_data(symbol):
        try:
            data = ma.get_market_data([symbol], '2020-01-01', '2021-01-01')
            return symbol, data
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return symbol, None
    
    # Load multiple symbols in parallel
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
    
    # Using ProcessPoolExecutor (CPU-bound tasks)
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(executor.map(load_symbol_data, symbols))
    
    # Combine results
    all_data = {}
    for symbol, data in results:
        if data is not None:
            all_data[symbol] = data
    
    print(f"Loaded data for {len(all_data)} symbols")
    
    # Parallel portfolio optimization
    def optimize_portfolio_single(returns, method='sharpe'):
        try:
            optimizer = ma.PortfolioOptimizer(returns)
            return optimizer.optimize_portfolio(method=method)
        except Exception as e:
            print(f"Optimization error: {e}")
            return None
    
    # Optimize multiple portfolios
    portfolios = {}
    for symbol in symbols:
        if symbol in all_data:
            returns = all_data[symbol].pct_change().dropna()
            if len(returns) > 100:  # Minimum data requirement
                portfolios[symbol] = returns
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        optimization_results = list(executor.map(
            optimize_portfolio_single, 
            portfolios.values()
        ))

Memory Management
-----------------

Manage memory efficiently for large datasets:

.. code-block:: python

    # Process data in chunks
    def process_large_dataset(symbols, chunk_size=10):
        results = {}
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            
            # Load chunk
            chunk_data = {}
            for symbol in chunk:
                data = ma.get_market_data([symbol], '2019-01-01', '2021-01-01')
                chunk_data[symbol] = data
            
            # Process chunk
            for symbol, data in chunk_data.items():
                # Calculate metrics
                returns = data[symbol].pct_change().dropna()
                metrics = ma.calculate_risk_metrics(returns)
                results[symbol] = metrics
            
            # Clear memory
            del chunk_data
            
        return results
    
    # Use generators for memory efficiency
    def data_generator(symbols):
        for symbol in symbols:
            data = ma.get_market_data([symbol], '2020-01-01', '2021-01-01')
            yield symbol, data
    
    # Process without storing all data
    for symbol, data in data_generator(symbols[:5]):
        returns = data[symbol].pct_change().dropna()
        print(f"{symbol}: {len(returns)} data points")

Optimizing Calculations
---------------------

Optimize mathematical operations:

.. code-block:: python

    # Use efficient libraries
    import numexpr as ne
    
    # BAD: Standard pandas operations
    def calculate_volatility_slow(returns):
        return returns.rolling(20).std()
    
    # GOOD: Use optimized calculations
    def calculate_volatility_fast(returns):
        return returns.rolling(20, min_periods=1).std()
    
    # Pre-allocate arrays
    def preallocate_example(n):
        # BAD: Growing list
        result_slow = []
        for i in range(n):
            result_slow.append(i * 2)
        
        # GOOD: Pre-allocated
        result_fast = np.empty(n)
        for i in range(n):
            result_fast[i] = i * 2
        
        return result_fast

Database Integration
--------------------

Use databases for efficient data storage and retrieval:

.. code-block:: python

    import sqlite3
    from sqlalchemy import create_engine
    
    # Create database connection
    engine = create_engine('sqlite:///market_data.db')
    
    # Save data to database
    def save_to_database(data, table_name):
        data.to_sql(table_name, engine, if_exists='replace', index=True)
    
    # Load data from database
    def load_from_database(table_name, start_date=None, end_date=None):
        query = f"SELECT * FROM {table_name}"
        
        if start_date and end_date:
            query += f" WHERE Date BETWEEN '{start_date}' AND '{end_date}'"
        
        return pd.read_sql(query, engine, index_col='Date', parse_dates=['Date'])
    
    # Example usage
    # save_to_database(data, 'daily_prices')
    # loaded_data = load_from_database('daily_prices', '2020-01-01', '2020-12-31')

Monitoring Performance
--------------------

Profile your code to identify bottlenecks:

.. code-block:: python

    import cProfile
    import pstats
    
    def profile_function(func, *args, **kwargs):
        """Profile a function and print results"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        return result
    
    # Example: Profile portfolio optimization
    def portfolio_optimization_test():
        data = ma.get_market_data(symbols, '2020-01-01', '2021-01-01')
        returns = data.pct_change().dropna()
        optimizer = ma.PortfolioOptimizer(returns)
        return optimizer.optimize_portfolio()
    
    # Run profiling
    profile_function(portfolio_optimization_test)

Best Practices Summary
---------------------

1. **Vectorization**: Use vectorized operations instead of loops
2. **Data Types**: Use appropriate data types (float32 vs float64)
3. **Caching**: Cache expensive computations
4. **Parallel Processing**: Use multiple cores for independent tasks
5. **Memory Management**: Process data in chunks for large datasets
6. **Efficient Libraries**: Use optimized libraries (NumPy, Pandas, Numba)
7. **Database**: Use databases for large datasets
8. **Profiling**: Profile code to identify bottlenecks

Performance Checklist
--------------------

- [ ] Are you using vectorized operations?
- [ ] Have you chosen appropriate data types?
- [ ] Are you caching expensive computations?
- [ ] Are you using parallel processing where possible?
- [ ] Are you managing memory efficiently?
- [ ] Have you profiled your code?
- [ ] Are you using efficient algorithms?
- [ ] Are you minimizing I/O operations?

Next Steps
----------

- Learn about :ref:`machine_learning` optimization techniques
- Explore :ref:`api_reference` for performance-optimized functions
- Check out the examples for efficient implementations
