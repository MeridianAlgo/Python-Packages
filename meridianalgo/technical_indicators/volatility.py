"""
Volatility Technical Indicators

This module contains volatility-based technical indicators including Bollinger Bands, ATR, etc.
"""

from typing import Tuple

import pandas as pd


def BollingerBands(
    prices: pd.Series, period: int = 20, std_dev: float = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands

    Args:
        prices: Price series
        period: Moving average period (default: 20)
        std_dev: Standard deviation multiplier (default: 2)

    Returns:
        Tuple of (Upper Band, Middle Band, Lower Band)
    """
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return upper_band, middle_band, lower_band


def ATR(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average True Range (ATR)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        ATR values
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    return atr


def KeltnerChannels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    multiplier: float = 2,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Channel period (default: 20)
        multiplier: ATR multiplier (default: 2)

    Returns:
        Tuple of (Upper Channel, Middle Channel, Lower Channel)
    """
    middle_channel = close.rolling(window=period).mean()
    atr = ATR(high, low, close, period)

    upper_channel = middle_channel + (atr * multiplier)
    lower_channel = middle_channel - (atr * multiplier)

    return upper_channel, middle_channel, lower_channel


def DonchianChannels(
    high: pd.Series, low: pd.Series, period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channels

    Args:
        high: High prices
        low: Low prices
        period: Channel period (default: 20)

    Returns:
        Tuple of (Upper Channel, Middle Channel, Lower Channel)
    """
    upper_channel = high.rolling(window=period).max()
    lower_channel = low.rolling(window=period).min()
    middle_channel = (upper_channel + lower_channel) / 2

    return upper_channel, middle_channel, lower_channel
