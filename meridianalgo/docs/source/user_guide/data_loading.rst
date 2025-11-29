.. _data_loading:

Data Loading
===========

This guide covers how to load and manage financial data in MeridianAlgo.

Loading Market Data
------------------

MeridianAlgo provides several ways to load market data. The simplest way is to use the built-in data loader:

.. code-block:: python

    import meridianalgo as ma
    
    # Load daily data for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = ma.get_market_data(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2021-01-01',
        interval='1d'  # '1d' for daily, '1h' for hourly, '1m' for minute data
    )
    
    print(data.head())

Data Sources
-----------

MeridianAlgo supports multiple data sources. The default is Yahoo Finance, but you can specify others:

.. code-block:: python

    # Using a different data source
    data = ma.get_market_data(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2021-01-01',
        data_source='yfinance'  # Default, other options: 'alpha_vantage', 'quandl', etc.
    )

Handling Missing Data
--------------------

Financial data often contains missing values. Here's how to handle them:

.. code-block:: python

    # Forward fill missing values
    data_filled = data.ffill()
    
    # Or interpolate
    data_interpolated = data.interpolate()
    
    # Drop rows with any missing values
    data_clean = data.dropna()

Resampling Data
--------------

You can easily resample time series data:

.. code-block:: python

    # Resample to weekly data (W), taking the last value each week
    weekly_data = data.resample('W').last()
    
    # Resample to monthly data with OHLC
    monthly_ohlc = data.resample('M').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

Calculating Returns
------------------

Basic return calculations:

.. code-block:: python

    # Simple returns
    returns = data.pct_change().dropna()
    
    # Log returns
    import numpy as np
    log_returns = np.log(data / data.shift(1)).dropna()
    
    # Cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1

Working with Multiple Timeframes
------------------------------

You can analyze data across different timeframes:

.. code-block:: python

    # Get daily and weekly data
    daily_data = ma.get_market_data(symbols, '2020-01-01', '2021-01-01', '1d')
    weekly_data = ma.get_market_data(symbols, '2020-01-01', '2021-01-01', '1wk')
    
    # Align data to common index
    aligned_daily, aligned_weekly = daily_data.align(weekly_data, join='inner')

Saving and Loading Data
----------------------

Save your processed data for later use:

.. code-block:: python

    # Save to CSV
    data.to_csv('market_data.csv')
    
    # Save to HDF5 (efficient for large datasets)
    data.to_hdf('market_data.h5', key='data')
    
    # Load data
    loaded_data = pd.read_hdf('market_data.h5', 'data')

Best Practices
-------------

1. Always check for and handle missing data
2. Be aware of look-ahead bias when working with time series data
3. Consider transaction costs and slippage in your analysis
4. Use appropriate data types to save memory
5. Cache frequently used data to improve performance

Next Steps
----------

- Learn about :ref:`portfolio_optimization`
- Explore :ref:`risk_analysis` techniques
- Check out the :ref:`api_reference` for all available functions
