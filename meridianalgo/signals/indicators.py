"""
Technical Indicators Module

Comprehensive technical indicators for trading signal generation.
Includes trend, momentum, volatility, and volume indicators.
"""

from typing import Dict, Tuple


import numpy as np
import pandas as pd

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


# =============================================================================
# MOVING AVERAGES (TREND)
# =============================================================================


def SMA(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Simple Moving Average.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.SMA(data.values, timeperiod=period), index=data.index)
    return data.rolling(window=period).mean()


def EMA(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Exponential Moving Average.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.EMA(data.values, timeperiod=period), index=data.index)
    return data.ewm(span=period, adjust=False).mean()


def WMA(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Weighted Moving Average.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.WMA(data.values, timeperiod=period), index=data.index)
    weights = np.arange(1, period + 1)
    return data.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def DEMA(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Double Exponential Moving Average.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.DEMA(data.values, timeperiod=period), index=data.index)
    ema1 = EMA(data, period)
    ema2 = EMA(ema1, period)
    return 2 * ema1 - ema2


def TEMA(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Triple Exponential Moving Average.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.TEMA(data.values, timeperiod=period), index=data.index)
    ema1 = EMA(data, period)
    ema2 = EMA(ema1, period)
    ema3 = EMA(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def HMA(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Hull Moving Average (HMA).
    Eliminates lag while improving smoothness.
    """
    if TALIB_AVAILABLE:
        half_window = int(window / 2)
        sqrt_window = int(np.sqrt(window))
        wma_half = talib.WMA(series, timeperiod=half_window)
        wma_full = talib.WMA(series, timeperiod=window)
        diff = 2 * wma_half - wma_full
        return talib.WMA(diff, timeperiod=sqrt_window)
    
    def _wma_internal(s, w):
        weights = np.arange(1, w + 1)
        return s.rolling(w).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
    half_window = int(window / 2)
    sqrt_window = int(np.sqrt(window))
    wma_half = _wma_internal(series, half_window)
    wma_full = _wma_internal(series, window)
    diff = 2 * wma_half - wma_full
    return _wma_internal(diff, sqrt_window)


def ALMA(series: pd.Series, window: int = 9, sigma: float = 6.0, offset: float = 0.85) -> pd.Series:
    """
    Arnaud Legoux Moving Average (ALMA).
    Uses Gaussian distribution to reduce lag and provide smoothness.
    """
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)



def KAMA(data: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Kaufman Adaptive Moving Average.
    Adapts to market volatility using efficiency ratio.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.KAMA(data.values, timeperiod=period), index=data.index)

    change = abs(data - data.shift(period))
    volatility = data.diff().abs().rolling(period).sum()

    efficiency_ratio = change / volatility
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], 0).fillna(0)

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)

    sc = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = np.zeros(len(data))
    kama[:period] = data.iloc[:period]
    
    # Vectorized calculation is difficult for KAMA due to its recursive nature
    # but we can use values for faster access
    data_values = data.values
    sc_values = sc.values
    
    for i in range(period, len(data)):
        kama[i] = kama[i - 1] + sc_values[i] * (data_values[i] - kama[i - 1])

    return pd.Series(kama, index=data.index)



# =============================================================================
# MACD
# =============================================================================


def MACD(
    data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.
    """
    if TALIB_AVAILABLE:
        macd, macdsignal, macdhist = talib.MACD(
            data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return pd.Series(macd, index=data.index), pd.Series(macdsignal, index=data.index), pd.Series(macdhist, index=data.index)

    ema_fast = EMA(data, fast)
    ema_slow = EMA(data, slow)

    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================


def RSI(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)

    delta = data.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def Stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    """
    if TALIB_AVAILABLE:
        slowk, slowd = talib.STOCH(
            high.values, low.values, close.values, 
            fastk_period=k_period, slowk_period=d_period, slowk_matype=0, 
            slowd_period=d_period, slowd_matype=0
        )
        return pd.Series(slowk, index=close.index), pd.Series(slowd, index=close.index)

    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()

    return k, d



def WilliamsR(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Williams %R.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=period), index=close.index)

    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()


    return -100 * (highest_high - close) / (highest_high - lowest_low)




def CCI(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """
    Commodity Channel Index.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.CCI(high.values, low.values, close.values, timeperiod=period), index=close.index)

    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )

    return (typical_price - sma) / (0.015 * mad)


def ROC(data: pd.Series, period: int = 12) -> pd.Series:
    """
    Rate of Change (ROC).
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.ROC(data.values, timeperiod=period), index=data.index)

    return ((data - data.shift(period)) / data.shift(period)) * 100


def Momentum(data: pd.Series, period: int = 10) -> pd.Series:
    """
    Momentum indicator.

    Args:
        data: Price series
        period: Lookback period

    Returns:
        Momentum (price difference)
    """
    return data - data.shift(period)


def MFI(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Money Flow Index.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.MFI(high.values, low.values, close.values, volume.values.astype(float), timeperiod=period), index=close.index)

    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_sum = positive_flow.rolling(period).sum()
    negative_sum = negative_flow.rolling(period).sum()

    money_ratio = positive_sum / negative_sum

    return 100 - (100 / (1 + money_ratio))


def TSI(
    data: pd.Series, long: int = 25, short: int = 13, signal: int = 7
) -> Tuple[pd.Series, pd.Series]:
    """
    True Strength Index.

    Returns:
        Tuple of (TSI, Signal line)
    """
    momentum = data.diff()

    smoothed_momentum = EMA(EMA(momentum, long), short)
    smoothed_abs_momentum = EMA(EMA(momentum.abs(), long), short)

    tsi = 100 * smoothed_momentum / smoothed_abs_momentum
    signal_line = EMA(tsi, signal)

    return tsi, signal_line


def UltimateOscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pd.Series:
    """
    Ultimate Oscillator.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.ULTOSC(high.values, low.values, close.values, timeperiod1=period1, timeperiod2=period2, timeperiod3=period3), index=close.index)

    prev_close = close.shift(1)

    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat(
        [low, prev_close], axis=1
    ).min(axis=1)

    avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
    avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
    avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()

    return 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================


def BollingerBands(
    data: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    """
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
        return pd.Series(upper, index=data.index), pd.Series(middle, index=data.index), pd.Series(lower, index=data.index)

    middle = SMA(data, period)
    std = data.rolling(period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def ATR(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average True Range.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()


# Alias
AverageTrueRange = ATR


def KeltnerChannels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_mult: float = 2.0,
    atr_period: int = 10,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels.

    Returns:
        Tuple of (Upper, Middle, Lower)
    """
    middle = EMA(close, period)
    atr = ATR(high, low, close, atr_period)

    upper = middle + (atr * atr_mult)
    lower = middle - (atr * atr_mult)

    return upper, middle, lower


def DonchianChannels(
    high: pd.Series, low: pd.Series, period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channels.

    Returns:
        Tuple of (Upper, Middle, Lower)
    """
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    middle = (upper + lower) / 2

    return upper, middle, lower


def StandardDeviation(data: pd.Series, period: int = 20) -> pd.Series:
    """Rolling standard deviation."""
    return data.rolling(period).std()


# =============================================================================
# TREND INDICATORS
# =============================================================================



def ADX(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (Wilder's).
    
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    if TALIB_AVAILABLE:
        adx = pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
        plus_di = pd.Series(talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period), index=close.index)
        minus_di = pd.Series(talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period), index=close.index)
        return adx, plus_di, minus_di

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, abs(high - prev_close), abs(low - prev_close)], axis=1
    ).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Wilder's Smoothing
    def wilder_smoothing(s, n):
        return s.ewm(alpha=1/n, adjust=False).mean()

    atr = wilder_smoothing(tr, period)
    plus_di = 100 * wilder_smoothing(pd.Series(plus_dm, index=close.index), period) / atr
    minus_di = 100 * wilder_smoothing(pd.Series(minus_dm, index=close.index), period) / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smoothing(dx, period)

    return adx, plus_di, minus_di



def Aroon(
    high: pd.Series, low: pd.Series, period: int = 25
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Aroon Indicator.
    """
    if TALIB_AVAILABLE:
        aroonup, aroondown = talib.AROON(high.values, low.values, timeperiod=period)
        aroonosc = talib.AROONOSC(high.values, low.values, timeperiod=period)
        return pd.Series(aroonup, index=high.index), pd.Series(aroondown, index=high.index), pd.Series(aroonosc, index=high.index)

    aroon_up = (
        100
        * (
            period
            - high.rolling(period + 1).apply(lambda x: period - x.argmax(), raw=True)
        )
        / period
    )

    aroon_down = (
        100
        * (
            period
            - low.rolling(period + 1).apply(lambda x: period - x.argmin(), raw=True)
        )
        / period
    )

    aroon_osc = aroon_up - aroon_down

    return aroon_up, aroon_down, aroon_osc


def ParabolicSAR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """
    Parabolic SAR.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.SAR(high.values, low.values, acceleration=af_start, maximum=af_max), index=close.index)

    length = len(close)
    sar = pd.Series(index=close.index, dtype=float)

    # Initialize
    is_uptrend = True
    af = af_start
    ep = low.iloc[0]
    sar.iloc[0] = high.iloc[0]

    for i in range(1, length):
        if is_uptrend:
            sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])
            sar.iloc[i] = min(
                sar.iloc[i],
                low.iloc[i - 1],
                low.iloc[i - 2] if i > 1 else low.iloc[i - 1],
            )

            if low.iloc[i] < sar.iloc[i]:
                is_uptrend = False
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_increment, af_max)
        else:
            sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])
            sar.iloc[i] = max(
                sar.iloc[i],
                high.iloc[i - 1],
                high.iloc[i - 2] if i > 1 else high.iloc[i - 1],
            )

            if high.iloc[i] > sar.iloc[i]:
                is_uptrend = True
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_increment, af_max)

    return sar



def Supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.
    
    Returns:
        Tuple of (Supertrend line, Direction: 1=up, -1=down)
    """
    atr = ATR(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Final bands (tightening logic)
    final_upper = pd.Series(upper_band)
    final_lower = pd.Series(lower_band)
    
    for i in range(1, len(close)):
        # Upper band tightening: if current upper is lower than previous or previous close was above previous upper
        if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
            
        # Lower band tightening
        if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    supertrend = pd.Series(0.0, index=close.index)
    direction = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        if direction.iloc[i-1] == 1:
            if close.iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            if close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]


    return supertrend, direction




def Ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> Dict[str, pd.Series]:
    """
    Ichimoku Cloud.

    Returns:
        Dictionary with all Ichimoku components
    """
    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2

    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # Senkou Span B (Leading Span B)
    senkou_span_b = (
        (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
    ).shift(displacement)

    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-displacement)

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }


# =============================================================================
# VOLUME INDICATORS
# =============================================================================


def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.OBV(close.values, volume.values.astype(float)), index=close.index)

    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def VWAP(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """
    Volume Weighted Average Price.
    """
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def ChaikinMoneyFlow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Chaikin Money Flow.
    """
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)

    mf_volume = mf_multiplier * volume

    return mf_volume.rolling(period).sum() / volume.rolling(period).sum()


def AccumulationDistribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """
    Accumulation/Distribution Line.
    """
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)

    mf_volume = mf_multiplier * volume

    return mf_volume.cumsum()


def ForceIndex(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """
    Force Index.
    """
    force = close.diff() * volume
    return EMA(force, period)


def EaseOfMovement(
    high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """
    Ease of Movement.
    """
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    br = volume / (high - low)

    emv = dm / br
    return emv.rolling(period).mean()


# =============================================================================
# SUPPORT/RESISTANCE
# =============================================================================


def PivotPoints(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Calculate pivot points.

    Args:
        high: Previous period high
        low: Previous period low
        close: Previous period close

    Returns:
        Dictionary with pivot and support/resistance levels
    """
    pivot = (high + low + close) / 3

    return {
        "pivot": pivot,
        "r1": 2 * pivot - low,
        "r2": pivot + (high - low),
        "r3": high + 2 * (pivot - low),
        "s1": 2 * pivot - high,
        "s2": pivot - (high - low),
        "s3": low - 2 * (high - pivot),
    }


def FibonacciRetracement(
    high: float, low: float, trend: str = "up"
) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels.

    Args:
        high: Swing high
        low: Swing low
        trend: 'up' or 'down'

    Returns:
        Dictionary with Fibonacci levels
    """
    diff = high - low

    levels = {
        "0.0": high if trend == "up" else low,
        "23.6": high - 0.236 * diff if trend == "up" else low + 0.236 * diff,
        "38.2": high - 0.382 * diff if trend == "up" else low + 0.382 * diff,
        "50.0": high - 0.5 * diff if trend == "up" else low + 0.5 * diff,
        "61.8": high - 0.618 * diff if trend == "up" else low + 0.618 * diff,
        "78.6": high - 0.786 * diff if trend == "up" else low + 0.786 * diff,
        "100.0": low if trend == "up" else high,
    }

    return levels


def FibonacciExtension(high: float, low: float, trend: str = "up") -> Dict[str, float]:
    """
    Calculate Fibonacci extension levels.

    Args:
        high: Swing high
        low: Swing low
        trend: 'up' or 'down'

    Returns:
        Dictionary with Fibonacci extension levels
    """
    diff = high - low

    if trend == "up":
        levels = {
            "100.0": high,
            "127.2": high + 0.272 * diff,
            "161.8": high + 0.618 * diff,
            "200.0": high + diff,
            "261.8": high + 1.618 * diff,
        }
    else:
        levels = {
            "100.0": low,
            "127.2": low - 0.272 * diff,
            "161.8": low - 0.618 * diff,
            "200.0": low - diff,
            "261.8": low - 1.618 * diff,
        }

    return levels

# =============================================================================
# MISCELLANEOUS & ADVANCED INDICATORS
# =============================================================================

def CoppockCurve(data: pd.Series, roc1: int = 14, roc2: int = 11, wma_period: int = 10) -> pd.Series:
    """
    Coppock Curve.
    Used for identifying long-term buying opportunities.
    """
    roc_sum = ROC(data, roc1) + ROC(data, roc2)
    return WMA(roc_sum, wma_period)


def VortexIndicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Vortex Indicator (VI+ and VI-).
    Identifies the start of a trend.
    """
    tr = ATR(high, low, close, 1) # True range
    tr_sum = tr.rolling(period).sum()

    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    vi_plus = vm_plus.rolling(period).sum() / tr_sum
    vi_minus = vm_minus.rolling(period).sum() / tr_sum

    return vi_plus, vi_minus


def MassIndex(high: pd.Series, low: pd.Series, period: int = 25, ema_period: int = 9) -> pd.Series:
    """
    Mass Index.
    Identifies trend reversals by measuring range expansion.
    """
    range_series = high - low
    ema1 = EMA(range_series, ema_period)
    ema2 = EMA(ema1, ema_period)
    
    ratio = ema1 / ema2
    return ratio.rolling(period).sum()


def ChopinessIndex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Chopiness Index.
    Determines if the market is trending or choppy.
    Values > 61.8 (choppy), < 38.2 (trending).
    """
    tr = ATR(high, low, close, 1)
    sum_tr = tr.rolling(period).sum()
    max_high = high.rolling(period).max()
    min_low = low.rolling(period).min()
    
    return 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(period)


def FisherTransform(high: pd.Series, low: pd.Series, period: int = 10) -> pd.Series:
    """
    Fisher Transform.
    Converts price data into a Gaussian normal distribution.
    """
    med = (high + low) / 2
    low_val = med.rolling(period).min()
    high_val = med.rolling(period).max()
    
    # Scale to -1 to 1
    raw = 0.33 * 2 * ((med - low_val) / (high_val - low_val) - 0.5) + 0.67 * 0.0 # simplified
    # In practice should use recursive smoothing
    
    fish = 0.5 * np.log((1 + raw) / (1 - raw))
    return fish.rolling(period).mean() # smoothing


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SMA",
    "EMA",
    "WMA",
    "DEMA",
    "TEMA",
    "HMA",
    "ALMA",
    "KAMA",
    "MACD",
    "RSI",
    "Stochastic",
    "WilliamsR",
    "CCI",
    "ROC",
    "Momentum",
    "MFI",
    "TSI",
    "UltimateOscillator",
    "BollingerBands",
    "ATR",
    "KeltnerChannels",
    "DonchianChannels",
    "StandardDeviation",
    "ADX",
    "Aroon",
    "ParabolicSAR",
    "Supertrend",
    "Ichimoku",
    "OBV",
    "VWAP",
    "ChaikinMoneyFlow",
    "AccumulationDistribution",
    "ForceIndex",
    "EaseOfMovement",
    "PivotPoints",
    "FibonacciRetracement",
    "FibonacciExtension",
    "CoppockCurve",
    "VortexIndicator",
    "MassIndex",
    "ChopinessIndex",
    "FisherTransform",
]
