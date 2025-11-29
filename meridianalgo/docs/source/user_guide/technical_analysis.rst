.. _technical_analysis:

Technical Analysis
=================

This guide covers technical analysis tools and indicators available in MeridianAlgo for market analysis and trading signals.

Introduction
-----------

Technical analysis uses historical price and volume data to predict future price movements. MeridianAlgo provides a comprehensive set of technical indicators and analysis tools.

Moving Averages
---------------

Moving averages help smooth price data to identify trends:

.. code-block:: python

    import meridianalgo as ma
    import matplotlib.pyplot as plt
    
    # Get price data
    symbol = 'AAPL'
    data = ma.get_market_data([symbol], start_date='2020-01-01', end_date='2021-01-01')
    prices = data[symbol]
    
    # Calculate moving averages
    sma_20 = ma.calculate_sma(prices, window=20)
    sma_50 = ma.calculate_sma(prices, window=50)
    ema_20 = ma.calculate_ema(prices, window=20)
    
    # Plot
    plt.figure(figsize=(12, 6))
    prices.plot(label='Price', linewidth=2)
    sma_20.plot(label='SMA 20', alpha=0.7)
    sma_50.plot(label='SMA 50', alpha=0.7)
    ema_20.plot(label='EMA 20', alpha=0.7)
    plt.title(f'{symbol} Price with Moving Averages')
    plt.legend()
    plt.grid(True)
    plt.show()

Oscillators
-----------

### Relative Strength Index (RSI)

RSI measures the speed and change of price movements:

.. code-block:: python

    # Calculate RSI
    rsi = ma.calculate_rsi(prices, window=14)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price chart
    prices.plot(ax=ax1, label='Price')
    ax1.set_title(f'{symbol} Price')
    ax1.grid(True)
    
    # RSI chart
    rsi.plot(ax=ax2, label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.set_title('RSI (14)')
    ax2.set_ylabel('RSI')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

### MACD (Moving Average Convergence Divergence)

MACD is a trend-following momentum indicator:

.. code-block:: python

    # Calculate MACD
    macd_data = ma.calculate_macd(prices)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Price and moving averages
    plt.subplot(3, 1, 1)
    prices.plot(label='Price')
    plt.title(f'{symbol} Price')
    plt.grid(True)
    
    # MACD line
    plt.subplot(3, 1, 2)
    macd_data['macd'].plot(label='MACD')
    macd_data['signal'].plot(label='Signal', alpha=0.7)
    plt.title('MACD')
    plt.legend()
    plt.grid(True)
    
    # Histogram
    plt.subplot(3, 1, 3)
    macd_data['histogram'].plot(kind='bar', color='green', alpha=0.7)
    plt.title('MACD Histogram')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

Bollinger Bands
--------------

Bollinger Bands consist of a middle band (SMA) and two outer bands (standard deviations):

.. code-block:: python

    # Calculate Bollinger Bands
    bb = ma.calculate_bollinger_bands(prices, window=20, num_std=2)
    
    # Plot
    plt.figure(figsize=(12, 6))
    prices.plot(label='Price', linewidth=2)
    bb['middle'].plot(label='Middle Band (SMA 20)', alpha=0.7)
    bb['upper'].plot(label='Upper Band (+2 std)', alpha=0.7)
    bb['lower'].plot(label='Lower Band (-2 std)', alpha=0.7)
    plt.fill_between(bb.index, bb['upper'], bb['lower'], alpha=0.1)
    plt.title(f'{symbol} Bollinger Bands')
    plt.legend()
    plt.grid(True)
    plt.show()

Volume Indicators
-----------------

### On-Balance Volume (OBV)

OBV uses volume flow to predict price changes:

.. code-block:: python

    # Get volume data (if available)
    volume_data = ma.get_market_data([symbol], start_date='2020-01-01', end_date='2021-01-01', include_volume=True)
    
    if 'Volume' in volume_data.columns:
        volume = volume_data[f'{symbol}_Volume']
        obv = ma.calculate_obv(prices, volume)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        prices.plot(ax=ax1, label='Price')
        ax1.set_title(f'{symbol} Price')
        ax1.grid(True)
        
        obv.plot(ax=ax2, label='OBV', color='orange')
        ax2.set_title('On-Balance Volume')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

Volatility Indicators
--------------------

### Average True Range (ATR)

ATR measures market volatility:

.. code-block:: python

    # Calculate ATR
    high = data[f'{symbol}_High'] if f'{symbol}_High' in data.columns else prices + prices.rolling(10).std()
    low = data[f'{symbol}_Low'] if f'{symbol}_Low' in data.columns else prices - prices.rolling(10).std()
    
    atr = ma.calculate_atr(prices, high, low, window=14)
    
    # Plot
    plt.figure(figsize=(12, 6))
    prices.plot(label='Price')
    plt.twinx()
    atr.plot(label='ATR', color='red', alpha=0.7)
    plt.title(f'{symbol} Price and ATR')
    plt.ylabel('ATR')
    plt.legend()
    plt.grid(True)
    plt.show()

Pattern Recognition
-------------------

MeridianAlgo can help identify common chart patterns:

.. code-block:: python

    # Identify support and resistance levels
    support_resistance = ma.identify_support_resistance(prices, window=20)
    
    # Plot with support/resistance
    plt.figure(figsize=(12, 6))
    prices.plot(label='Price', linewidth=2)
    
    for level in support_resistance['support']:
        plt.axhline(y=level, color='green', linestyle='--', alpha=0.5)
    
    for level in support_resistance['resistance']:
        plt.axhline(y=level, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f'{symbol} Support and Resistance Levels')
    plt.grid(True)
    plt.show()

Combining Indicators
--------------------

Create a comprehensive technical analysis dashboard:

.. code-block:: python

    def create_technical_dashboard(symbol, start_date='2020-01-01', end_date='2021-01-01'):
        data = ma.get_market_data([symbol], start_date, end_date)
        prices = data[symbol]
        
        # Calculate indicators
        sma_20 = ma.calculate_sma(prices, 20)
        rsi = ma.calculate_rsi(prices, 14)
        macd = ma.calculate_macd(prices)
        bb = ma.calculate_bollinger_bands(prices)
        
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        
        # Price and Bollinger Bands
        axes[0].plot(prices, label='Price', linewidth=2)
        axes[0].plot(bb['middle'], label='SMA 20', alpha=0.7)
        axes[0].fill_between(bb.index, bb['upper'], bb['lower'], alpha=0.1)
        axes[0].set_title(f'{symbol} Price and Bollinger Bands')
        axes[0].legend()
        axes[0].grid(True)
        
        # RSI
        axes[1].plot(rsi, label='RSI', color='purple')
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        axes[1].set_title('RSI (14)')
        axes[1].set_ylabel('RSI')
        axes[1].grid(True)
        
        # MACD
        axes[2].plot(macd['macd'], label='MACD')
        axes[2].plot(macd['signal'], label='Signal', alpha=0.7)
        axes[2].set_title('MACD')
        axes[2].legend()
        axes[2].grid(True)
        
        # Volume (if available)
        if 'Volume' in data.columns:
            volume = data[f'{symbol}_Volume']
            axes[3].bar(volume.index, volume.values, alpha=0.7)
            axes[3].set_title('Volume')
            axes[3].set_ylabel('Volume')
        else:
            axes[3].text(0.5, 0.5, 'Volume data not available', 
                        transform=axes[3].transAxes, ha='center')
        
        plt.tight_layout()
        plt.show()
    
    # Create dashboard
    create_technical_dashboard('AAPL')

Best Practices
-------------

1. Use multiple indicators together for confirmation
2. Consider the time horizon of your analysis
3. Be aware of indicator lag and false signals
4. Combine technical analysis with fundamental analysis
5. Always consider market context and regime

Next Steps
----------

- Learn about :ref:`machine_learning` for predictive models
- Explore :ref:`backtesting` to test your strategies
- Check the :ref:`api_reference` for all technical indicators
