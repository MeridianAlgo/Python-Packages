# Technical Indicators API

Comprehensive technical analysis indicators for financial data.

##  Overview

The Technical Indicators module provides 50+ indicators across five categories:
- **Momentum Indicators**: RSI, Stochastic, Williams %R, ROC, Momentum
- **Trend Indicators**: Moving averages, MACD, ADX, Aroon, Parabolic SAR, Ichimoku
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels, Donchian Channels
- **Volume Indicators**: OBV, AD Line, Chaikin Oscillator, Money Flow Index
- **Overlay Indicators**: Pivot Points, Fibonacci Retracement, Support/Resistance

##  Quick Start

```python
import meridianalgo as ma
import pandas as pd

# Get sample data
data = ma.get_market_data(['AAPL'], start_date='2023-01-01')
close = data['AAPL']
high = data['AAPL'].high  # If you have OHLC data
low = data['AAPL'].low
volume = data['AAPL'].volume

# Calculate indicators
rsi = ma.RSI(close, period=14)
macd_line, signal_line, histogram = ma.MACD(close)
bb_upper, bb_middle, bb_lower = ma.BollingerBands(close)
```

##  Momentum Indicators

### RSI (Relative Strength Index)

```python
rsi = ma.RSI(prices, period=14)
```

**Parameters:**
- `prices` (pd.Series): Price series
- `period` (int): RSI period (default: 14)

**Returns:**
- `pd.Series`: RSI values (0-100)

**Example:**
```python
rsi = ma.RSI(close, period=14)
print(f"Current RSI: {rsi.iloc[-1]:.2f}")

# RSI interpretation
if rsi.iloc[-1] > 70:
    print("Overbought")
elif rsi.iloc[-1] < 30:
    print("Oversold")
```

### Stochastic Oscillator

```python
stoch_k, stoch_d = ma.Stochastic(high, low, close, k_period=14, d_period=3)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `k_period` (int): %K period (default: 14)
- `d_period` (int): %D period (default: 3)

**Returns:**
- `tuple`: (%K, %D) values

**Example:**
```python
stoch_k, stoch_d = ma.Stochastic(high, low, close)
print(f"Stochastic %K: {stoch_k.iloc[-1]:.2f}")
print(f"Stochastic %D: {stoch_d.iloc[-1]:.2f}")
```

### Williams %R

```python
wr = ma.WilliamsR(high, low, close, period=14)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `period` (int): Williams %R period (default: 14)

**Returns:**
- `pd.Series`: Williams %R values (-100 to 0)

### Rate of Change (ROC)

```python
roc = ma.ROC(prices, period=12)
```

**Parameters:**
- `prices` (pd.Series): Price series
- `period` (int): ROC period (default: 12)

**Returns:**
- `pd.Series`: ROC values (percentage)

### Momentum

```python
momentum = ma.Momentum(prices, period=10)
```

**Parameters:**
- `prices` (pd.Series): Price series
- `period` (int): Momentum period (default: 10)

**Returns:**
- `pd.Series`: Momentum values

##  Trend Indicators

### Simple Moving Average (SMA)

```python
sma = ma.SMA(prices, period)
```

**Parameters:**
- `prices` (pd.Series): Price series
- `period` (int): SMA period

**Returns:**
- `pd.Series`: SMA values

**Example:**
```python
sma_20 = ma.SMA(close, 20)
sma_50 = ma.SMA(close, 50)

# Golden cross signal
if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
    print("Golden cross detected!")
```

### Exponential Moving Average (EMA)

```python
ema = ma.EMA(prices, period)
```

**Parameters:**
- `prices` (pd.Series): Price series
- `period` (int): EMA period

**Returns:**
- `pd.Series`: EMA values

### MACD (Moving Average Convergence Divergence)

```python
macd_line, signal_line, histogram = ma.MACD(prices, fast=12, slow=26, signal=9)
```

**Parameters:**
- `prices` (pd.Series): Price series
- `fast` (int): Fast EMA period (default: 12)
- `slow` (int): Slow EMA period (default: 26)
- `signal` (int): Signal line period (default: 9)

**Returns:**
- `tuple`: (MACD line, Signal line, Histogram)

**Example:**
```python
macd_line, signal_line, histogram = ma.MACD(close)

# MACD signal
if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
    print("MACD bullish crossover!")
```

### ADX (Average Directional Index)

```python
adx = ma.ADX(high, low, close, period=14)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `period` (int): ADX period (default: 14)

**Returns:**
- `pd.Series`: ADX values (0-100)

### Aroon Indicator

```python
aroon_up, aroon_down = ma.Aroon(high, low, period=25)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `period` (int): Aroon period (default: 25)

**Returns:**
- `tuple`: (Aroon Up, Aroon Down)

### Parabolic SAR

```python
psar = ma.ParabolicSAR(high, low, close, acceleration=0.02, maximum=0.2)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `acceleration` (float): Acceleration factor (default: 0.02)
- `maximum` (float): Maximum acceleration (default: 0.2)

**Returns:**
- `pd.Series`: Parabolic SAR values

### Ichimoku Cloud

```python
ichimoku = ma.Ichimoku(high, low, close)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices

**Returns:**
- `dict`: Dictionary with Ichimoku components
  - `tenkan_sen`: Conversion Line
  - `kijun_sen`: Base Line
  - `senkou_span_a`: Leading Span A
  - `senkou_span_b`: Leading Span B
  - `chikou_span`: Lagging Span

##  Volatility Indicators

### Bollinger Bands

```python
bb_upper, bb_middle, bb_lower = ma.BollingerBands(prices, period=20, std_dev=2)
```

**Parameters:**
- `prices` (pd.Series): Price series
- `period` (int): Moving average period (default: 20)
- `std_dev` (float): Standard deviation multiplier (default: 2)

**Returns:**
- `tuple`: (Upper Band, Middle Band, Lower Band)

**Example:**
```python
bb_upper, bb_middle, bb_lower = ma.BollingerBands(close)

# Bollinger Band squeeze
bb_width = (bb_upper - bb_lower) / bb_middle
if bb_width.iloc[-1] < 0.1:  # 10% width threshold
    print("Bollinger Band squeeze detected!")
```

### Average True Range (ATR)

```python
atr = ma.ATR(high, low, close, period=14)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `period` (int): ATR period (default: 14)

**Returns:**
- `pd.Series`: ATR values

### Keltner Channels

```python
kc_upper, kc_middle, kc_lower = ma.KeltnerChannels(high, low, close, period=20, multiplier=2)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `period` (int): Channel period (default: 20)
- `multiplier` (float): ATR multiplier (default: 2)

**Returns:**
- `tuple`: (Upper Channel, Middle Channel, Lower Channel)

### Donchian Channels

```python
dc_upper, dc_middle, dc_lower = ma.DonchianChannels(high, low, period=20)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `period` (int): Channel period (default: 20)

**Returns:**
- `tuple`: (Upper Channel, Middle Channel, Lower Channel)

##  Volume Indicators

### On-Balance Volume (OBV)

```python
obv = ma.OBV(close, volume)
```

**Parameters:**
- `close` (pd.Series): Close prices
- `volume` (pd.Series): Volume data

**Returns:**
- `pd.Series`: OBV values

### Accumulation/Distribution Line

```python
ad_line = ma.ADLine(high, low, close, volume)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `volume` (pd.Series): Volume data

**Returns:**
- `pd.Series`: AD Line values

### Chaikin Oscillator

```python
chaikin = ma.ChaikinOscillator(high, low, close, volume, fast=3, slow=10)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `volume` (pd.Series): Volume data
- `fast` (int): Fast period (default: 3)
- `slow` (int): Slow period (default: 10)

**Returns:**
- `pd.Series`: Chaikin Oscillator values

### Money Flow Index (MFI)

```python
mfi = ma.MoneyFlowIndex(high, low, close, volume, period=14)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `volume` (pd.Series): Volume data
- `period` (int): MFI period (default: 14)

**Returns:**
- `pd.Series`: MFI values (0-100)

### Ease of Movement

```python
eom = ma.EaseOfMovement(high, low, volume, period=14)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `volume` (pd.Series): Volume data
- `period` (int): EOM period (default: 14)

**Returns:**
- `pd.Series`: Ease of Movement values

##  Overlay Indicators

### Pivot Points

```python
pivot_data = ma.PivotPoints(high, low, close)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices

**Returns:**
- `dict`: Dictionary with pivot point levels
  - `pivot`: Pivot point
  - `r1`, `r2`, `r3`: Resistance levels
  - `s1`, `s2`, `s3`: Support levels

### Fibonacci Retracement

```python
fib_data = ma.FibonacciRetracement(high, low, levels=None)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `levels` (list): Fibonacci levels (default: [0.236, 0.382, 0.5, 0.618, 0.786])

**Returns:**
- `dict`: Dictionary with Fibonacci levels

### Support and Resistance

```python
sr_data = ma.SupportResistance(prices, window=20, min_touches=2)
```

**Parameters:**
- `prices` (pd.Series): Price series
- `window` (int): Rolling window for local extrema (default: 20)
- `min_touches` (int): Minimum touches for level validation (default: 2)

**Returns:**
- `dict`: Dictionary with support and resistance data
  - `resistance`: Resistance levels
  - `support`: Support levels
  - `resistance_levels`: Key resistance levels
  - `support_levels`: Key support levels

##  Complete Example

```python
import meridianalgo as ma
import pandas as pd
import matplotlib.pyplot as plt

def technical_analysis_example():
    """Complete technical analysis example."""
    
    # Get data
    data = ma.get_market_data(['AAPL'], start_date='2023-01-01')
    close = data['AAPL']
    
    # Calculate indicators
    rsi = ma.RSI(close, 14)
    macd_line, signal_line, histogram = ma.MACD(close)
    bb_upper, bb_middle, bb_lower = ma.BollingerBands(close)
    sma_20 = ma.SMA(close, 20)
    sma_50 = ma.SMA(close, 50)
    
    # Create signals
    signals = pd.DataFrame(index=close.index)
    signals['price'] = close
    signals['rsi'] = rsi
    signals['macd'] = macd_line
    signals['bb_upper'] = bb_upper
    signals['bb_lower'] = bb_lower
    signals['sma_20'] = sma_20
    signals['sma_50'] = sma_50
    
    # Generate trading signals
    signals['rsi_signal'] = 0
    signals['rsi_signal'][rsi < 30] = 1  # Oversold
    signals['rsi_signal'][rsi > 70] = -1  # Overbought
    
    signals['macd_signal'] = 0
    signals['macd_signal'][macd_line > signal_line] = 1
    signals['macd_signal'][macd_line < signal_line] = -1
    
    signals['bb_signal'] = 0
    signals['bb_signal'][close < bb_lower] = 1  # Below lower band
    signals['bb_signal'][close > bb_upper] = -1  # Above upper band
    
    signals['trend_signal'] = 0
    signals['trend_signal'][sma_20 > sma_50] = 1  # Uptrend
    signals['trend_signal'][sma_20 < sma_50] = -1  # Downtrend
    
    # Combined signal
    signals['combined_signal'] = (
        signals['rsi_signal'] + 
        signals['macd_signal'] + 
        signals['bb_signal'] + 
        signals['trend_signal']
    )
    
    # Print current signals
    print("Current Technical Analysis Signals:")
    print(f"RSI: {rsi.iloc[-1]:.2f} ({'Oversold' if rsi.iloc[-1] < 30 else 'Overbought' if rsi.iloc[-1] > 70 else 'Neutral'})")
    print(f"MACD: {macd_line.iloc[-1]:.4f} ({'Bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'Bearish'})")
    print(f"Bollinger Bands: {close.iloc[-1]:.2f} ({'Below Lower' if close.iloc[-1] < bb_lower.iloc[-1] else 'Above Upper' if close.iloc[-1] > bb_upper.iloc[-1] else 'Within Bands'})")
    print(f"Trend: {'Uptrend' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'Downtrend'}")
    print(f"Combined Signal: {signals['combined_signal'].iloc[-1]}")
    
    return signals

# Run example
if __name__ == "__main__":
    signals = technical_analysis_example()
```

##  Additional Resources

- [Portfolio Management API](portfolio_management.md) - Portfolio optimization
- [Risk Analysis API](risk_analysis.md) - Risk metrics and analysis
- [Examples](../examples/) - Practical use cases
- [Performance Benchmarks](../benchmarks.md) - Performance metrics
