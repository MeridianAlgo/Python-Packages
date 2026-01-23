"""
Overlay Technical Indicators

This module contains overlay indicators including Pivot Points, Fibonacci levels, etc.
"""

from typing import Dict

import pandas as pd


def PivotPoints(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Dict[str, pd.Series]:
    """
    Pivot Points

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Dictionary with pivot point levels
    """
    # Calculate pivot point
    pivot = (high + low + close) / 3

    # Calculate resistance and support levels
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


def FibonacciRetracement(
    high: pd.Series, low: pd.Series, levels: list = None
) -> Dict[str, pd.Series]:
    """
    Fibonacci Retracement Levels

    Args:
        high: High prices
        low: Low prices
        levels: Fibonacci levels (default: [0.236, 0.382, 0.5, 0.618, 0.786])

    Returns:
        Dictionary with Fibonacci levels
    """
    if levels is None:
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    fib_levels = {}
    price_range = high - low

    for level in levels:
        fib_levels[f"fib_{level}"] = high - (price_range * level)

    return fib_levels


def SupportResistance(
    prices: pd.Series, window: int = 20, min_touches: int = 2
) -> Dict[str, pd.Series]:
    """
    Support and Resistance Levels

    Args:
        prices: Price series
        window: Rolling window for local extrema (default: 20)
        min_touches: Minimum touches for level validation (default: 2)

    Returns:
        Dictionary with support and resistance levels
    """
    # Find local maxima and minima
    local_max = prices.rolling(window=window, center=True).max() == prices
    local_min = prices.rolling(window=window, center=True).min() == prices

    # Extract resistance levels (local maxima)
    resistance_levels = prices[local_max].dropna()

    # Extract support levels (local minima)
    support_levels = prices[local_min].dropna()

    # Create series with NaN values except at support/resistance points
    resistance = pd.Series(index=prices.index, dtype=float)
    support = pd.Series(index=prices.index, dtype=float)

    resistance[local_max] = prices[local_max]
    support[local_min] = prices[local_min]

    return {
        "resistance": resistance,
        "support": support,
        "resistance_levels": resistance_levels,
        "support_levels": support_levels,
    }
