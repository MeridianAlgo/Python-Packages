"""
Volume Technical Indicators

This module contains volume-based technical indicators including OBV, AD Line, etc.
"""

import pandas as pd


def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV)

    Args:
        close: Close prices
        volume: Volume data

    Returns:
        OBV values
    """
    price_change = close.diff()
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(close)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    return obv


def ADLine(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """
    Accumulation/Distribution Line (AD Line)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data

    Returns:
        AD Line values
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)  # Handle division by zero

    ad_line = (clv * volume).cumsum()
    return ad_line


def ChaikinOscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """
    Chaikin Oscillator

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        fast: Fast period (default: 3)
        slow: Slow period (default: 10)

    Returns:
        Chaikin Oscillator values
    """
    ad_line = ADLine(high, low, close, volume)
    chaikin_osc = ad_line.ewm(span=fast).mean() - ad_line.ewm(span=slow).mean()
    return chaikin_osc


def MoneyFlowIndex(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Money Flow Index (MFI)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        period: MFI period (default: 14)

    Returns:
        MFI values
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi


def EaseOfMovement(
    high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """
    Ease of Movement

    Args:
        high: High prices
        low: Low prices
        volume: Volume data
        period: EOM period (default: 14)

    Returns:
        Ease of Movement values
    """
    distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_height = volume / (high - low)

    eom = distance_moved / box_height
    eom_smoothed = eom.rolling(window=period).mean()

    return eom_smoothed
