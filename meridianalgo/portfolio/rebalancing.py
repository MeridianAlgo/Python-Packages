"""
Portfolio Rebalancing Module

This module provides portfolio rebalancing strategies for the MeridianAlgo platform.
"""

from datetime import datetime
from typing import Dict


class Rebalancer:
    """Base Portfolio Rebalancer."""

    def __init__(self, target_weights: Dict[str, float]):
        self.target_weights = target_weights

    def rebalance(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Rebalance portfolio to target weights."""
        return self.target_weights


class CalendarRebalancer(Rebalancer):
    """Calendar-based rebalancing strategy."""

    def __init__(self, target_weights: Dict[str, float], frequency: str = "monthly"):
        super().__init__(target_weights)
        self.frequency = frequency

    def should_rebalance(self, last_rebalance: datetime) -> bool:
        """Check if rebalancing is needed based on calendar."""
        if self.frequency == "monthly":
            return (datetime.now() - last_rebalance).days >= 30
        elif self.frequency == "quarterly":
            return (datetime.now() - last_rebalance).days >= 90
        elif self.frequency == "annually":
            return (datetime.now() - last_rebalance).days >= 365
        return False


class ThresholdRebalancer(Rebalancer):
    """Threshold-based rebalancing strategy."""

    def __init__(self, target_weights: Dict[str, float], threshold: float = 0.05):
        super().__init__(target_weights)
        self.threshold = threshold

    def should_rebalance(self, current_weights: Dict[str, float]) -> bool:
        """Check if rebalancing is needed based on weight drift."""
        for asset, target_weight in self.target_weights.items():
            if asset in current_weights:
                drift = abs(current_weights[asset] - target_weight)
                if drift > self.threshold:
                    return True
        return False


class OptimalRebalancer(Rebalancer):
    """Optimal rebalancing with transaction cost consideration."""

    def __init__(
        self, target_weights: Dict[str, float], transaction_cost: float = 0.001
    ):
        super().__init__(target_weights)
        self.transaction_cost = transaction_cost

    def calculate_rebalancing_trades(
        self, current_weights: Dict[str, float], portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate optimal rebalancing trades considering transaction costs."""
        trades = {}

        for asset in self.target_weights:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = self.target_weights[asset]

            weight_diff = target_weight - current_weight
            trade_value = weight_diff * portfolio_value

            # Only trade if benefit exceeds transaction cost
            if (
                abs(trade_value) * self.transaction_cost
                < abs(weight_diff) * portfolio_value * 0.01
            ):
                trades[asset] = trade_value
            else:
                trades[asset] = 0.0

        return trades
