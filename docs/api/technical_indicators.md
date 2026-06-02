# Technical Indicators

MeridianAlgo ships 40+ technical indicators. They are available three ways:

```python
import meridianalgo as ma                              # ma.RSI, ma.MACD, ...
from meridianalgo.technical_indicators import RSI, SMA  # flat functional import
from meridianalgo.signals.indicators import RSI         # source module
```

All functions take a price `pd.Series` (and `high`/`low`/`volume` where relevant)
and return `pd.Series` (or a tuple of series).

## Quick Example

```python
import meridianalgo as ma

data = ma.get_market_data(["AAPL"], start_date="2023-01-01")
close = data["AAPL"]

rsi = ma.RSI(close, period=14)
macd_line, signal_line, histogram = ma.MACD(close, fast=12, slow=26, signal=9)
upper, middle, lower = ma.BollingerBands(close, period=20, std_dev=2)
sma_20 = ma.SMA(close, period=20)
ema_12 = ma.EMA(close, period=12)
```

## Available Indicators

**Trend / Moving Averages**
`SMA`, `EMA`, `WMA`, `MACD`, `ADX`, `Aroon`, `ParabolicSAR`, `Ichimoku`

**Momentum**
`RSI`, `Stochastic`, `WilliamsR`, `CCI`, `ROC`

**Volatility**
`BollingerBands`, `ATR`, `KeltnerChannels`, `DonchianChannels`

**Volume**
`OBV`, `MFI`, `AccumulationDistribution`, `ChaikinMoneyFlow`, `VWAP`

See [`meridianalgo/signals/indicators.py`](https://github.com/MeridianAlgo/Python-Packages/blob/main/meridianalgo/signals/indicators.py)
for the full list and signatures.

## Signatures (selected)

| Indicator | Call | Returns |
|-----------|------|---------|
| RSI | `ma.RSI(close, period=14)` | `Series` |
| MACD | `ma.MACD(close, fast=12, slow=26, signal=9)` | `(macd, signal, hist)` |
| Bollinger Bands | `ma.BollingerBands(close, period=20, std_dev=2)` | `(upper, middle, lower)` |
| ATR | `ma.ATR(high, low, close, period=14)` | `Series` |
| Stochastic | `ma.Stochastic(high, low, close, k_period=14, d_period=3)` | `(%K, %D)` |
| OBV | `ma.OBV(close, volume)` | `Series` |
| MFI | `ma.MFI(high, low, close, volume, period=14)` | `Series` |

## Object-Oriented API

For a class-based interface (and custom indicators), use
`meridianalgo.technical_analysis`:

```python
from meridianalgo.technical_analysis import RSI, MACD, BollingerBands, BaseIndicator

rsi = RSI(period=14).calculate(close)
macd_line, signal_line, hist = MACD(fast=12, slow=26, signal=9).calculate(close)

# Build a custom indicator
class RangeMA(BaseIndicator):
    def __init__(self, period=10):
        self.period = period

    def calculate(self, data):
        return data.rolling(self.period).mean()

result = RangeMA(20).calculate(close)
```

## Candlestick Patterns

```python
from meridianalgo.technical_analysis import CandlestickPatterns

patterns = CandlestickPatterns()
doji = patterns.detect_doji(ohlcv_df)       # boolean Series
hammer = patterns.detect_hammer(ohlcv_df)   # boolean Series
```

`ohlcv_df` must contain `Open`, `High`, `Low`, `Close` columns (case-insensitive).
