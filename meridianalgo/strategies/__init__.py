"""
Trading strategies module.

This module provides various trading strategies for algorithmic trading.
"""

from .momentum import (
    BaseStrategy,
    BollingerBandsStrategy,
    MACDCrossover,
    MomentumStrategy,
    PairsTrading,
    RSIMeanReversion,
    create_strategy,
)

__all__ = [
    "BaseStrategy",
    "MomentumStrategy",
    "RSIMeanReversion",
    "MACDCrossover",
    "PairsTrading",
    "BollingerBandsStrategy",
    "create_strategy",
]
