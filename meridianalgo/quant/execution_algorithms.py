"""
Execution Algorithms Module

Institutional-grade execution algorithms including VWAP, TWAP, POV,
Implementation Shortfall, and advanced optimal execution strategies.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ExecutionState:
    """State of an execution algorithm."""

    time: float
    remaining_quantity: float
    executed_quantity: float
    average_price: float
    market_price: float
    total_cost: float


class VWAP:
    """
    Volume-Weighted Average Price (VWAP) execution algorithm.

    Aims to execute trades in proportion to historical volume patterns
    to achieve a price close to the VWAP.
    """

    def __init__(self, total_quantity: float, start_time: str, end_time: str):
        """
        Initialize VWAP algorithm.

        Parameters:
        -----------
        total_quantity : float
            Total quantity to execute
        start_time : str
            Start time for execution
        end_time : str
            End time for execution
        """
        self.total_quantity = total_quantity
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self.executed_quantity = 0
        self.execution_schedule = None

    def calculate_schedule(self, historical_volume: pd.Series) -> pd.Series:
        """
        Calculate execution schedule based on historical volume profile.

        Parameters:
        -----------
        historical_volume : pd.Series
            Historical intraday volume pattern

        Returns:
        --------
        pd.Series
            Execution schedule (quantity per interval)
        """
        # Normalize volume to create weights
        total_volume = historical_volume.sum()
        volume_weights = historical_volume / total_volume

        # Allocate quantity based on volume distribution
        execution_schedule = volume_weights * self.total_quantity

        self.execution_schedule = execution_schedule
        return execution_schedule

    def execute_slice(
        self,
        current_time: pd.Timestamp,
        market_volume: float,
        market_price: float,
        max_participation: float = 0.10,
    ) -> Dict:
        """
        Execute a slice of the order.

        Parameters:
        -----------
        current_time : pd.Timestamp
            Current timestamp
        market_volume : float
            Current market volume
        market_price : float
            Current market price
        max_participation : float
            Maximum participation rate (e.g., 0.10 = 10% of volume)

        Returns:
        --------
        Dict
            Execution details
        """
        if self.execution_schedule is None:
            raise ValueError("Call calculate_schedule() first")

        # Get target quantity for this slice
        idx = min(
            len(self.execution_schedule) - 1,
            int((current_time - self.start_time).total_seconds() / 60),
        )
        target_quantity = self.execution_schedule.iloc[idx]

        # Limit by participation rate
        max_quantity = market_volume * max_participation
        actual_quantity = min(
            target_quantity, max_quantity, self.total_quantity - self.executed_quantity
        )

        self.executed_quantity += actual_quantity

        return {
            "time": current_time,
            "quantity": actual_quantity,
            "price": market_price,
            "remaining": self.total_quantity - self.executed_quantity,
            "executed_pct": self.executed_quantity / self.total_quantity,
        }

    def calculate_vwap_benchmark(self, prices: pd.Series, volumes: pd.Series) -> float:
        """
        Calculate VWAP benchmark.

        VWAP = (Price * Volume) / (Volume)
        """
        vwap = (prices * volumes).sum() / volumes.sum()
        return vwap


class TWAP:
    """
    Time-Weighted Average Price (TWAP) execution algorithm.

    Executes equal quantities at regular time intervals.
    """

    def __init__(
        self,
        total_quantity: float,
        duration_minutes: int,
        slice_interval_minutes: int = 5,
    ):
        """
        Initialize TWAP algorithm.

        Parameters:
        -----------
        total_quantity : float
            Total quantity to execute
        duration_minutes : int
            Total duration in minutes
        slice_interval_minutes : int
            Time interval between slices
        """
        self.total_quantity = total_quantity
        self.duration_minutes = duration_minutes
        self.slice_interval = slice_interval_minutes
        self.n_slices = duration_minutes // slice_interval_minutes
        self.quantity_per_slice = total_quantity / self.n_slices
        self.executed_quantity = 0
        self.slice_count = 0

    def execute_slice(self, market_price: float, available_liquidity: float) -> Dict:
        """
        Execute one TWAP slice.

        Parameters:
        -----------
        market_price : float
            Current market price
        available_liquidity : float
            Available liquidity in the market

        Returns:
        --------
        Dict
            Execution details
        """
        if self.slice_count >= self.n_slices:
            return {"quantity": 0, "price": market_price, "status": "completed"}

        # Execute target quantity (or less if liquidity constrained)
        target_quantity = min(
            self.quantity_per_slice,
            available_liquidity,
            self.total_quantity - self.executed_quantity,
        )

        self.executed_quantity += target_quantity
        self.slice_count += 1

        return {
            "slice_number": self.slice_count,
            "quantity": target_quantity,
            "price": market_price,
            "remaining": self.total_quantity - self.executed_quantity,
            "progress": self.executed_quantity / self.total_quantity,
            "status": "active" if self.slice_count < self.n_slices else "completed",
        }

    def get_schedule(self) -> np.ndarray:
        """Get the complete execution schedule."""
        return np.full(self.n_slices, self.quantity_per_slice)


class POV:
    """
    Percentage of Volume (POV) execution algorithm.

    Executes a target percentage of market volume.
    """

    def __init__(
        self,
        total_quantity: float,
        target_pov: float = 0.10,
        min_pov: float = 0.05,
        max_pov: float = 0.20,
    ):
        """
        Initialize POV algorithm.

        Parameters:
        -----------
        total_quantity : float
            Total quantity to execute
        target_pov : float
            Target percentage of volume (e.g., 0.10 = 10%)
        min_pov : float
            Minimum POV rate
        max_pov : float
            Maximum POV rate
        """
        self.total_quantity = total_quantity
        self.target_pov = target_pov
        self.min_pov = min_pov
        self.max_pov = max_pov
        self.executed_quantity = 0
        self.cumulative_market_volume = 0

    def execute(
        self, market_volume: float, market_price: float, time_remaining_pct: float
    ) -> Dict:
        """
        Execute based on current market volume.

        Parameters:
        -----------
        market_volume : float
            Current interval's market volume
        market_price : float
            Current market price
        time_remaining_pct : float
            Percentage of time remaining (0 to 1)

        Returns:
        --------
        Dict
            Execution details
        """
        remaining_quantity = self.total_quantity - self.executed_quantity

        # Adjust POV based on progress
        if time_remaining_pct > 0:
            executed_pct = self.executed_quantity / self.total_quantity
            if executed_pct < (1 - time_remaining_pct):
                # Behind schedule, increase POV
                adjusted_pov = min(self.target_pov * 1.2, self.max_pov)
            elif executed_pct > (1 - time_remaining_pct):
                # Ahead of schedule, decrease POV
                adjusted_pov = max(self.target_pov * 0.8, self.min_pov)
            else:
                adjusted_pov = self.target_pov
        else:
            adjusted_pov = self.max_pov  # Urgent execution

        # Calculate target quantity
        target_quantity = market_volume * adjusted_pov
        actual_quantity = min(target_quantity, remaining_quantity)

        self.executed_quantity += actual_quantity
        self.cumulative_market_volume += market_volume

        # Calculate realized POV
        realized_pov = (
            self.executed_quantity / self.cumulative_market_volume
            if self.cumulative_market_volume > 0
            else 0
        )

        return {
            "quantity": actual_quantity,
            "price": market_price,
            "target_pov": adjusted_pov,
            "realized_pov": realized_pov,
            "remaining": remaining_quantity - actual_quantity,
            "progress": self.executed_quantity / self.total_quantity,
        }


class ImplementationShortfall:
    """
    Implementation Shortfall (Almgren-Chriss) optimal execution algorithm.

    Minimizes the trade-off between market impact and timing risk.
    """

    def __init__(
        self,
        total_quantity: float,
        total_time: float,
        volatility: float,
        risk_aversion: float = 1e-6,
        permanent_impact: float = 0.1,
        temporary_impact: float = 0.01,
    ):
        """
        Initialize Implementation Shortfall algorithm.

        Parameters:
        -----------
        total_quantity : float
            Total shares to execute
        total_time : float
            Total time horizon (in same units as volatility)
        volatility : float
            Price volatility
        risk_aversion : float
            Risk aversion parameter (lambda)
        permanent_impact : float
            Permanent market impact coefficient
        temporary_impact : float
            Temporary market impact coefficient
        """
        self.Q = total_quantity
        self.T = total_time
        self.sigma = volatility
        self.lambda_risk = risk_aversion
        self.gamma = permanent_impact
        self.eta = temporary_impact

        self.trajectory = None
        self.trade_list = None

    def calculate_optimal_trajectory(self, n_intervals: int = 10) -> pd.DataFrame:
        """
        Calculate optimal execution trajectory.

        Solves the Almgren-Chriss optimization problem.

        Parameters:
        -----------
        n_intervals : int
            Number of execution intervals

        Returns:
        --------
        pd.DataFrame
            Optimal trajectory with holdings and trade sizes
        """
        self.T / n_intervals
        kappa = np.sqrt(self.lambda_risk * self.sigma**2 / self.eta)

        # Calculate trajectory using closed-form solution
        times = np.linspace(0, self.T, n_intervals + 1)

        # Remaining holdings at each time
        sinh_terms = np.sinh(kappa * (self.T - times))
        holdings = self.Q * sinh_terms / np.sinh(kappa * self.T)

        # Trade sizes (negative of change in holdings)
        trades = -np.diff(holdings, prepend=0)

        # Create trajectory DataFrame
        trajectory = pd.DataFrame(
            {
                "time": times,
                "holdings": holdings,
                "trades": np.concatenate([[0], trades[1:]]),  # No trade at t=0
                "cumulative_executed": self.Q - holdings,
            }
        )

        self.trajectory = trajectory
        self.trade_list = trades[1:]  # Exclude t=0

        return trajectory

    def calculate_expected_cost(self) -> Dict[str, float]:
        """
        Calculate expected implementation shortfall cost components.

        Returns:
        --------
        Dict
            Cost breakdown (timing risk, permanent impact, temporary impact)
        """
        if self.trajectory is None:
            self.calculate_optimal_trajectory()

        n = len(self.trade_list)
        tau = self.T / n

        # Timing risk (price volatility cost)
        timing_risk = (
            0.5
            * self.lambda_risk
            * self.sigma**2
            * np.sum(self.trajectory["holdings"].values[:-1] ** 2)
            * tau
        )

        # Permanent impact cost
        permanent_cost = self.gamma * np.sum(np.abs(self.trade_list)) * self.Q / 2

        # Temporary impact cost
        temporary_cost = self.eta * np.sum(self.trade_list**2)

        total_cost = timing_risk + permanent_cost + temporary_cost

        return {
            "total_cost": total_cost,
            "timing_risk": timing_risk,
            "permanent_impact": permanent_cost,
            "temporary_impact": temporary_cost,
            "cost_per_share": total_cost / self.Q,
        }

    def execute_interval(
        self,
        interval: int,
        current_price: float,
        market_conditions: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute trades for a specific interval.

        Parameters:
        -----------
        interval : int
            Current interval number (0 to n_intervals-1)
        current_price : float
            Current market price
        market_conditions : Dict, optional
            Real-time market condition adjustments

        Returns:
        --------
        Dict
            Execution details
        """
        if self.trajectory is None:
            self.calculate_optimal_trajectory()

        if interval >= len(self.trade_list):
            return {"quantity": 0, "status": "completed"}

        target_quantity = self.trade_list[interval]

        # Adjust for market conditions if provided
        if market_conditions is not None:
            liquidity_factor = market_conditions.get("liquidity_factor", 1.0)
            urgency_factor = market_conditions.get("urgency_factor", 1.0)
            target_quantity *= liquidity_factor * urgency_factor

        # Estimate impact
        permanent_impact_price = self.gamma * target_quantity
        temporary_impact_price = self.eta * target_quantity

        expected_price = current_price + permanent_impact_price + temporary_impact_price

        return {
            "interval": interval,
            "quantity": target_quantity,
            "current_price": current_price,
            "expected_execution_price": expected_price,
            "permanent_impact": permanent_impact_price,
            "temporary_impact": temporary_impact_price,
            "remaining_holdings": self.trajectory["holdings"].iloc[interval + 1],
            "status": "active",
        }


class AdaptiveExecution:
    """
    Adaptive execution algorithm that adjusts to real-time market conditions.
    """

    def __init__(self, total_quantity: float, base_algorithm: str = "vwap"):
        """
        Initialize adaptive execution.

        Parameters:
        -----------
        total_quantity : float
            Total quantity to execute
        base_algorithm : str
            Base algorithm ('vwap', 'twap', 'pov', 'is')
        """
        self.total_quantity = total_quantity
        self.base_algorithm = base_algorithm
        self.executed_quantity = 0
        self.market_state = "normal"
        self.execution_history = []

    def assess_market_conditions(
        self,
        recent_volatility: float,
        average_volatility: float,
        recent_volume: float,
        average_volume: float,
        spread: float,
    ) -> str:
        """
        Assess current market conditions.

        Returns:
        --------
        str
            Market state: 'favorable', 'normal', 'adverse', 'highly_adverse'
        """
        vol_ratio = (
            recent_volatility / average_volatility if average_volatility > 0 else 1.0
        )
        volume_ratio = recent_volume / average_volume if average_volume > 0 else 1.0

        # Score market conditions
        score = 0

        # Volatility assessment
        if vol_ratio > 2.0:
            score -= 2
        elif vol_ratio > 1.5:
            score -= 1
        elif vol_ratio < 0.7:
            score += 1

        # Volume assessment
        if volume_ratio > 1.5:
            score += 1
        elif volume_ratio < 0.5:
            score -= 1

        # Spread assessment
        if spread > 0.005:  # 50 bps
            score -= 1
        elif spread < 0.001:  # 10 bps
            score += 1

        # Determine state
        if score >= 2:
            return "favorable"
        elif score <= -2:
            return "highly_adverse"
        elif score < 0:
            return "adverse"
        else:
            return "normal"

    def adapt_execution_rate(self, market_state: str, base_rate: float) -> float:
        """
        Adapt execution rate based on market conditions.

        Parameters:
        -----------
        market_state : str
            Current market state
        base_rate : float
            Base execution rate

        Returns:
        --------
        float
            Adjusted execution rate
        """
        adjustments = {
            "favorable": 1.3,  # Execute faster in good conditions
            "normal": 1.0,  # Normal pace
            "adverse": 0.7,  # Slow down
            "highly_adverse": 0.4,  # Significant slowdown
        }

        return base_rate * adjustments.get(market_state, 1.0)

    def execute(self, market_data: Dict) -> Dict:
        """
        Execute with adaptive strategy.

        Parameters:
        -----------
        market_data : Dict
            Current market data

        Returns:
        --------
        Dict
            Execution decision
        """
        # Assess market
        self.market_state = self.assess_market_conditions(
            market_data["recent_volatility"],
            market_data["average_volatility"],
            market_data["recent_volume"],
            market_data["average_volume"],
            market_data["spread"],
        )

        # Calculate base execution rate
        remaining = self.total_quantity - self.executed_quantity
        time_remaining = market_data.get("time_remaining_pct", 0.5)

        base_rate = (
            remaining * (1 - time_remaining) if time_remaining < 1 else remaining * 0.1
        )

        # Adapt rate
        adjusted_rate = self.adapt_execution_rate(self.market_state, base_rate)

        # Calculate actual quantity
        max_quantity = market_data["available_volume"] * 0.15  # Max 15% participation
        actual_quantity = min(adjusted_rate, max_quantity, remaining)

        execution = {
            "quantity": actual_quantity,
            "market_state": self.market_state,
            "base_rate": base_rate,
            "adjusted_rate": adjusted_rate,
            "price": market_data["price"],
            "remaining": remaining - actual_quantity,
        }

        self.executed_quantity += actual_quantity
        self.execution_history.append(execution)

        return execution


__all__ = [
    "VWAP",
    "TWAP",
    "POV",
    "ImplementationShortfall",
    "AdaptiveExecution",
    "ExecutionState",
]
