"""
Portfolio Rebalancing Module

This module provides portfolio rebalancing strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

class Rebalancer:
    """Base Portfolio Rebalancer."""
    
    def __init__(self, target_weights: Dict[str, float]):
        self.target_weights = target_weights
    
    def rebalance(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Rebalance portfolio to target weights."""
        return self.target_weights

class CalendarRebalancer(Rebalancer):
    """Calendar-based rebalancing strategy."""
    
    def __init__(self, target_weights: Dict[str, float], frequency: str = 'monthly'):
        super().__init__(target_weights)
        self.frequency = frequency
    
    def should_rebalance(self, last_rebalance: datetime) -> bool:
        """Check if rebalancing is needed based on calendar."""
        if self.frequency == 'monthly':
            return (datetime.now() - last_rebalance).days >= 30
        elif self.frequency == 'quarterly':
            return (datetime.now() - last_rebalance).days >= 90
        elif self.frequency == 'annually':
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
