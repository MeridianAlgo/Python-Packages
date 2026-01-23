"""
Regime Analysis Module

This module provides market regime detection and analysis.
"""

from typing import Dict

import pandas as pd


class RegimeDetector:
    """Market Regime Detector."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def detect_regimes(self, method: str = "volatility") -> pd.Series:
        """Detect market regimes."""
        if method == "volatility":
            return self.detect_volatility_regimes()
        elif method == "trend":
            return self.detect_trend_regimes()
        else:
            raise ValueError("Method must be 'volatility' or 'trend'")

    def detect_volatility_regimes(
        self, window: int = 21, threshold: float = 1.5
    ) -> pd.Series:
        """Detect volatility regimes."""
        rolling_vol = self.returns.rolling(window=window).std()
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()

        regimes = pd.Series(index=self.returns.index, dtype=str)
        regimes[rolling_vol > vol_mean + threshold * vol_std] = "High Volatility"
        regimes[rolling_vol < vol_mean - threshold * vol_std] = "Low Volatility"
        regimes[regimes.isna()] = "Normal Volatility"

        return regimes

    def detect_trend_regimes(self, window: int = 21) -> pd.Series:
        """Detect trend regimes."""
        rolling_mean = self.returns.rolling(window=window).mean()

        regimes = pd.Series(index=self.returns.index, dtype=str)
        regimes[rolling_mean > 0] = "Bull Market"
        regimes[rolling_mean < 0] = "Bear Market"
        regimes[rolling_mean == 0] = "Sideways"

        return regimes


class VolatilityRegime:
    """Volatility Regime Analysis."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def analyze_volatility_regimes(self) -> Dict[str, Dict[str, float]]:
        """Analyze volatility regimes."""
        regimes = RegimeDetector(self.returns).detect_volatility_regimes()

        analysis = {}
        for regime in regimes.unique():
            regime_returns = self.returns[regimes == regime]
            analysis[regime] = {
                "count": len(regime_returns),
                "mean_return": regime_returns.mean(),
                "volatility": regime_returns.std(),
                "sharpe_ratio": (
                    regime_returns.mean() / regime_returns.std()
                    if regime_returns.std() > 0
                    else 0
                ),
            }

        return analysis


class MarketRegime:
    """Market Regime Analysis."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def analyze_market_regimes(self) -> Dict[str, Dict[str, float]]:
        """Analyze market regimes."""
        regimes = RegimeDetector(self.returns).detect_trend_regimes()

        analysis = {}
        for regime in regimes.unique():
            regime_returns = self.returns[regimes == regime]
            analysis[regime] = {
                "count": len(regime_returns),
                "mean_return": regime_returns.mean(),
                "volatility": regime_returns.std(),
                "max_drawdown": self.calculate_max_drawdown(regime_returns),
            }

        return analysis

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min()
