"""Object-oriented wrappers around the functional indicator library."""

from typing import Tuple

import pandas as pd

from ..signals import indicators as _ind
from .framework import BaseIndicator


class RSI(BaseIndicator):
    """Relative Strength Index."""

    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.Series) -> pd.Series:
        return _ind.RSI(data, self.period)


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        return _ind.MACD(data, self.fast, self.slow, self.signal)


class BollingerBands(BaseIndicator):
    """Bollinger Bands (upper, middle, lower)."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        return _ind.BollingerBands(data, self.period, self.std_dev)
