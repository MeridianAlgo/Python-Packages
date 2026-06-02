"""Base framework for building custom technical indicators."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseIndicator(ABC):
    """Abstract base class for technical indicators.

    Subclasses implement :meth:`calculate`, which accepts a price series (or
    OHLCV frame) and returns the indicator output.
    """

    @abstractmethod
    def calculate(self, data: pd.Series) -> pd.Series:
        """Compute the indicator from input data."""
        raise NotImplementedError

    def __call__(self, data: pd.Series) -> pd.Series:
        return self.calculate(data)
