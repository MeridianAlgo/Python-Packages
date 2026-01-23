"""
Signal Evaluation Module

Provides tools to evaluate the performance of trading signals.
"""

from typing import Dict

import numpy as np
import pandas as pd

from ..quant.advanced_signals import information_coefficient


class SignalEvaluator:
    """
    Evaluate trading signals using standard quantitative metrics.
    """

    def __init__(self, signals: pd.Series, returns: pd.Series):
        """
        Initialize the evaluator.

        Args:
            signals: The generated signals (e.g. scores or binary)
            returns: Forward-looking returns corresponding to the signals
        """
        self.data = pd.concat([signals, returns], axis=1).dropna()
        self.data.columns = ["signal", "return"]

    def calculate_ic(self) -> float:
        """Calculate the Information Coefficient."""
        return information_coefficient(self.data["signal"], self.data["return"])

    def calculate_rank_ic(self) -> float:
        """Calculate the Rank Information Coefficient (Spearman)."""
        return self.data["signal"].corr(self.data["return"], method="spearman")

    def calculate_hit_rate(self) -> float:
        """Calculate the hit rate (percentage of signals with correct direction)."""
        hits = np.sign(self.data["signal"]) == np.sign(self.data["return"])
        return hits.mean()

    def calculate_turnover(self) -> float:
        """Calculate the signal turnover (average change in signal)."""
        return self.data["signal"].diff().abs().mean()

    def summary(self) -> Dict[str, float]:
        """Generate a summary of metrics."""
        return {
            "ic": self.calculate_ic(),
            "rank_ic": self.calculate_rank_ic(),
            "hit_rate": self.calculate_hit_rate(),
            "turnover": self.calculate_turnover(),
            "count": len(self.data),
        }
