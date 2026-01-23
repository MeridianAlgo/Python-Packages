"""
Trend Technical Indicators

This module contains trend-based technical indicators including moving averages, MACD, etc.
"""

from typing import Tuple

import pandas as pd


def SMA(prices: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average (SMA)

    Args:
        prices: Price series
        period: SMA period

    Returns:
        SMA values
    """
    return prices.rolling(window=period).mean()


def EMA(prices: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA)

    Args:
        prices: Price series
        period: EMA period

    Returns:
        EMA values
    """
    return prices.ewm(span=period).mean()


def MACD(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence)

    Args:
        prices: Price series
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = EMA(prices, fast)
    ema_slow = EMA(prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def ADX(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average Directional Index (ADX)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)

    Returns:
        ADX values
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement
    dm_plus = high.diff()
    dm_minus = -low.diff()

    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

    # Calculate smoothed values
    atr = tr.rolling(window=period).mean()
    di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

    # Calculate ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()

    return adx


def Aroon(
    high: pd.Series, low: pd.Series, period: int = 25
) -> Tuple[pd.Series, pd.Series]:
    """
    Aroon Indicator

    Args:
        high: High prices
        low: Low prices
        period: Aroon period (default: 25)

    Returns:
        Tuple of (Aroon Up, Aroon Down)
    """
    aroon_up = high.rolling(window=period).apply(
        lambda x: (period - x.argmax()) / period * 100
    )
    aroon_down = low.rolling(window=period).apply(
        lambda x: (period - x.argmin()) / period * 100
    )

    return aroon_up, aroon_down


def ParabolicSAR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> pd.Series:
    """
    Parabolic Stop and Reverse (SAR)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        acceleration: Acceleration factor (default: 0.02)
        maximum: Maximum acceleration (default: 0.2)

    Returns:
        Parabolic SAR values
    """
    psar = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    af = pd.Series(index=close.index, dtype=float)
    ep = pd.Series(index=close.index, dtype=float)

    # Initialize
    psar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1
    af.iloc[0] = acceleration
    ep.iloc[0] = high.iloc[0]

    for i in range(1, len(close)):
        # Calculate SAR
        if trend.iloc[i - 1] == 1:
            psar.iloc[i] = psar.iloc[i - 1] + af.iloc[i - 1] * (
                ep.iloc[i - 1] - psar.iloc[i - 1]
            )
        else:
            psar.iloc[i] = psar.iloc[i - 1] + af.iloc[i - 1] * (
                ep.iloc[i - 1] - psar.iloc[i - 1]
            )

        # Check for trend reversal
        if trend.iloc[i - 1] == 1:
            if low.iloc[i] <= psar.iloc[i]:
                trend.iloc[i] = -1
                psar.iloc[i] = ep.iloc[i - 1]
                af.iloc[i] = acceleration
                ep.iloc[i] = low.iloc[i]
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep.iloc[i - 1]:
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = min(af.iloc[i - 1] + acceleration, maximum)
                else:
                    ep.iloc[i] = ep.iloc[i - 1]
                    af.iloc[i] = af.iloc[i - 1]
        else:
            if high.iloc[i] >= psar.iloc[i]:
                trend.iloc[i] = 1
                psar.iloc[i] = ep.iloc[i - 1]
                af.iloc[i] = acceleration
                ep.iloc[i] = high.iloc[i]
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep.iloc[i - 1]:
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = min(af.iloc[i - 1] + acceleration, maximum)
                else:
                    ep.iloc[i] = ep.iloc[i - 1]
                    af.iloc[i] = af.iloc[i - 1]

    return psar


def Ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """
    Ichimoku Cloud

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Dictionary with Ichimoku components
    """
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=9).max()
    tenkan_low = low.rolling(window=9).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=26).max()
    kijun_low = low.rolling(window=26).min()
    kijun_sen = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    senkou_high = high.rolling(window=52).max()
    senkou_low = low.rolling(window=52).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)

    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-26)

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }
