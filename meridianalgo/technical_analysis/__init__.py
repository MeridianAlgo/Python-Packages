"""
Object-oriented technical analysis API.

Provides class-based indicators built on a common :class:`BaseIndicator`
framework, plus candlestick pattern detection. For the functional API, use
:mod:`meridianalgo.technical_indicators` or :mod:`meridianalgo.signals.indicators`.
"""

from .framework import BaseIndicator
from .indicators import MACD, RSI, BollingerBands
from .patterns import CandlestickPatterns

__all__ = [
    "BaseIndicator",
    "RSI",
    "MACD",
    "BollingerBands",
    "CandlestickPatterns",
]
