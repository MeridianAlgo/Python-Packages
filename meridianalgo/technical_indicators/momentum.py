"""
Momentum Technical Indicators

This module contains momentum-based technical indicators including RSI, Stochastic, etc.
"""

from typing import Tuple

import pandas as pd


def RSI(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)

    Args:
        prices: Price series
        period: RSI period (default: 14)

    Returns:
        RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def Stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)

    Returns:
        Tuple of (%K, %D) values
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent


def WilliamsR(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Williams %R

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Williams %R period (default: 14)

    Returns:
        Williams %R values
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr


def ROC(prices: pd.Series, period: int = 12) -> pd.Series:
    """
    Rate of Change (ROC)

    Args:
        prices: Price series
        period: ROC period (default: 12)

    Returns:
        ROC values
    """
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def Momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Momentum indicator

    Args:
        prices: Price series
        period: Momentum period (default: 10)

    Returns:
        Momentum values
    """
    return prices - prices.shift(period)
