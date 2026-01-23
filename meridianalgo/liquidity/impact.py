"""
Market Impact Analysis Module

Market impact models including Almgren-Chriss, square root law,
and empirical impact estimation.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ImpactEstimate:
    """Container for market impact estimates."""

    total_impact_bps: float
    permanent_impact_bps: float
    temporary_impact_bps: float
    expected_cost: float
    cost_std: float


class MarketImpact:
    """
    Market impact estimation and analysis.

    Provides multiple impact models including:
    - Square root law
    - Linear impact
    - Almgren-Chriss model
    - Empirical estimation
    """

    def __init__(self, daily_volume: float, volatility: float, spread_bps: float = 5.0):
        """
        Initialize MarketImpact.

        Args:
            daily_volume: Average daily volume (shares)
            volatility: Daily volatility (decimal)
            spread_bps: Average bid-ask spread in basis points
        """
        self.daily_volume = daily_volume
        self.volatility = volatility
        self.spread_bps = spread_bps

    def square_root_law(self, order_size: float, eta: float = 0.1) -> float:
        """
        Square root market impact model.

        Impact = eta * sigma * sqrt(Q / V)

        Args:
            order_size: Order size (shares)
            eta: Impact coefficient

        Returns:
            Expected impact (fraction of price)
        """
        if self.daily_volume == 0:
            return 0

        participation_rate = order_size / self.daily_volume
        impact = eta * self.volatility * np.sqrt(participation_rate)

        return impact

    def square_root_law_bps(self, order_size: float, eta: float = 0.1) -> float:
        """Get square root impact in basis points."""
        return self.square_root_law(order_size, eta) * 10000

    def linear_impact(self, order_size: float, lambda_coef: float = 0.01) -> float:
        """
        Linear market impact model.

        Impact = lambda * (Q / V)

        Args:
            order_size: Order size
            lambda_coef: Impact coefficient

        Returns:
            Expected impact
        """
        if self.daily_volume == 0:
            return 0

        return lambda_coef * (order_size / self.daily_volume)

    def power_law(
        self, order_size: float, eta: float = 0.1, delta: float = 0.5
    ) -> float:
        """
        Power law market impact model.

        Impact = eta * sigma * (Q / V)^delta

        Args:
            order_size: Order size
            eta: Impact coefficient
            delta: Power coefficient (0.5 = square root)

        Returns:
            Expected impact
        """
        if self.daily_volume == 0:
            return 0

        participation = order_size / self.daily_volume
        return eta * self.volatility * (participation**delta)

    def estimate_total_cost(
        self, order_size: float, price: float, model: str = "square_root"
    ) -> Dict[str, float]:
        """
        Estimate total execution cost.

        Args:
            order_size: Order size (shares)
            price: Current price
            model: Impact model to use

        Returns:
            Dictionary with cost breakdown
        """
        # Market impact
        if model == "square_root":
            impact = self.square_root_law(order_size)
        elif model == "linear":
            impact = self.linear_impact(order_size)
        else:
            impact = self.power_law(order_size)

        # Spread cost (half spread)
        spread_cost = (self.spread_bps / 10000) / 2

        # Total expected cost
        total_cost_pct = impact + spread_cost
        total_cost_dollars = total_cost_pct * price * order_size

        return {
            "impact_cost_pct": impact,
            "impact_cost_bps": impact * 10000,
            "spread_cost_pct": spread_cost,
            "spread_cost_bps": spread_cost * 10000,
            "total_cost_pct": total_cost_pct,
            "total_cost_bps": total_cost_pct * 10000,
            "total_cost_dollars": total_cost_dollars,
        }


class ImpactModel:
    """
    Empirical market impact model estimation.

    Calibrates impact model parameters from historical trade data.
    """

    def __init__(
        self,
        trades: Optional[pd.DataFrame] = None,
        quotes: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize ImpactModel.

        Args:
            trades: Historical trade data
            quotes: Historical quote data
        """
        self.trades = trades
        self.quotes = quotes
        self.is_fitted = False
        self.params = {}

    def fit(self, model_type: str = "power_law") -> "ImpactModel":
        """
        Fit impact model to historical data.

        Args:
            model_type: 'linear', 'square_root', or 'power_law'

        Returns:
            Self for chaining
        """
        if self.trades is None or self.quotes is None:
            raise ValueError("Need trades and quotes data to fit model")

        # Merge trades with quotes
        merged = pd.merge_asof(
            self.trades.sort_values("timestamp"),
            self.quotes.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        # Calculate returns following trades
        merged["mid"] = (merged["bid"] + merged["ask"]) / 2
        merged["mid_return"] = merged["mid"].pct_change()

        # Calculate participation rate
        merged["daily_volume"] = merged.groupby(merged["timestamp"].dt.date)[
            "size"
        ].transform("sum")
        merged["participation"] = merged["size"] / merged["daily_volume"]

        # Fit model based on type
        if model_type == "linear":
            self._fit_linear(merged)
        elif model_type == "square_root":
            self._fit_square_root(merged)
        else:
            self._fit_power_law(merged)

        self.is_fitted = True
        return self

    def _fit_linear(self, data: pd.DataFrame):
        """Fit linear impact model."""
        X = data["participation"].values.reshape(-1, 1)
        y = data["mid_return"].values

        # Simple OLS
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            self.params = {"alpha": betas[0], "lambda": betas[1]}
        except Exception:
            self.params = {"alpha": 0, "lambda": 0.01}

    def _fit_square_root(self, data: pd.DataFrame):
        """Fit square root impact model."""
        X = np.sqrt(data["participation"].values).reshape(-1, 1)
        y = data["mid_return"].values

        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            self.params = {"alpha": betas[0], "eta": betas[1]}
        except Exception:
            self.params = {"alpha": 0, "eta": 0.1}

    def _fit_power_law(self, data: pd.DataFrame):
        """Fit power law impact model."""
        # Log-transform for power law estimation
        valid = data[data["participation"] > 0].copy()

        if len(valid) < 10:
            self.params = {"eta": 0.1, "delta": 0.5}
            return

        X = np.log(valid["participation"].values)
        y = np.log(np.abs(valid["mid_return"].values) + 1e-10)

        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            self.params = {
                "log_eta": betas[0],
                "eta": np.exp(betas[0]),
                "delta": betas[1],
            }
        except Exception:
            self.params = {"eta": 0.1, "delta": 0.5}

    def predict(
        self, order_size: float, daily_volume: float, volatility: float = 0.02
    ) -> float:
        """
        Predict market impact.

        Args:
            order_size: Order size
            daily_volume: Daily trading volume
            volatility: Daily volatility

        Returns:
            Predicted impact
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        participation = order_size / daily_volume if daily_volume > 0 else 0

        if "lambda" in self.params:
            return self.params.get("alpha", 0) + self.params["lambda"] * participation
        elif "delta" in self.params:
            return (
                self.params["eta"]
                * volatility
                * (participation ** self.params["delta"])
            )
        else:
            return self.params["eta"] * volatility * np.sqrt(participation)


class AlmgrenChrissImpact:
    """
    Almgren-Chriss optimal execution impact model.

    Based on "Optimal Execution of Portfolio Transactions" (2000).
    Separates impact into permanent and temporary components.
    """

    def __init__(
        self,
        total_shares: float,
        total_time: float,
        volatility: float,
        daily_volume: float,
        permanent_impact: float = 0.1,
        temporary_impact: float = 0.01,
        risk_aversion: float = 1e-6,
    ):
        """
        Initialize Almgren-Chriss model.

        Args:
            total_shares: Total shares to execute
            total_time: Total time for execution (in days)
            volatility: Daily volatility
            daily_volume: Average daily volume
            permanent_impact: Permanent impact coefficient (gamma)
            temporary_impact: Temporary impact coefficient (eta)
            risk_aversion: Risk aversion parameter (lambda)
        """
        self.X = total_shares
        self.T = total_time
        self.sigma = volatility
        self.V = daily_volume
        self.gamma = permanent_impact
        self.eta = temporary_impact
        self.lambda_ = risk_aversion

    def optimal_trajectory(self, n_intervals: int = 10) -> pd.DataFrame:
        """
        Calculate optimal execution trajectory.

        Args:
            n_intervals: Number of trading intervals

        Returns:
            DataFrame with optimal holdings and trades
        """
        self.T / n_intervals  # Time per interval

        # Urgency parameter
        kappa = np.sqrt(self.lambda_ * self.sigma**2 / self.eta)

        # Calculate trajectory
        times = np.linspace(0, self.T, n_intervals + 1)

        if kappa * self.T < 0.001:  # Linear trajectory for small kappa
            holdings = self.X * (1 - times / self.T)
        else:
            holdings = (
                self.X * np.sinh(kappa * (self.T - times)) / np.sinh(kappa * self.T)
            )

        trades = -np.diff(holdings)

        return pd.DataFrame(
            {
                "time": times[:-1],
                "holdings": holdings[:-1],
                "trades": trades,
                "cumulative_traded": self.X - holdings[:-1],
            }
        )

    def expected_cost(self) -> Dict[str, float]:
        """
        Calculate expected execution cost.

        Returns:
            Dictionary with cost components
        """
        # Permanent impact cost
        permanent_cost = 0.5 * self.gamma * self.sigma * self.X**2 / self.V

        # Temporary impact cost
        kappa = np.sqrt(self.lambda_ * self.sigma**2 / self.eta)

        if kappa * self.T < 0.001:
            temporary_cost = self.eta * self.sigma * self.X**2 / self.T
        else:
            temporary_cost = (
                self.eta
                * self.sigma
                * self.X**2
                * np.cosh(kappa * self.T)
                / (self.T * np.sinh(kappa * self.T))
            )

        # Timing risk (variance of cost)
        timing_risk = 0.5 * self.lambda_ * self.sigma**2 * self.T * self.X**2

        total_cost = permanent_cost + temporary_cost

        return {
            "permanent_impact": permanent_cost,
            "temporary_impact": temporary_cost,
            "total_cost": total_cost,
            "timing_risk": timing_risk,
            "cost_per_share": total_cost / self.X if self.X > 0 else 0,
            "cost_bps": (total_cost / self.X) * 10000 if self.X > 0 else 0,
        }

    def efficient_frontier(self, n_points: int = 20) -> pd.DataFrame:
        """
        Calculate efficient frontier of expected cost vs variance.

        Args:
            n_points: Number of points on frontier

        Returns:
            DataFrame with frontier points
        """
        risk_aversions = np.logspace(-8, -3, n_points)

        results = []
        for ra in risk_aversions:
            self.lambda_ = ra
            costs = self.expected_cost()

            results.append(
                {
                    "risk_aversion": ra,
                    "expected_cost": costs["total_cost"],
                    "variance": costs["timing_risk"],
                    "std_dev": np.sqrt(costs["timing_risk"]),
                }
            )

        return pd.DataFrame(results)

    def sensitivity_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze sensitivity to model parameters.

        Returns:
            Dictionary of DataFrames with sensitivity results
        """
        base_cost = self.expected_cost()["total_cost"]

        # Sensitivity to each parameter
        results = {}

        # Volatility sensitivity
        vol_range = np.linspace(self.sigma * 0.5, self.sigma * 1.5, 10)
        vol_costs = []
        for vol in vol_range:
            orig = self.sigma
            self.sigma = vol
            vol_costs.append(
                {
                    "volatility": vol,
                    "cost": self.expected_cost()["total_cost"],
                    "cost_change": (self.expected_cost()["total_cost"] - base_cost)
                    / base_cost,
                }
            )
            self.sigma = orig
        results["volatility"] = pd.DataFrame(vol_costs)

        # Time sensitivity
        time_range = np.linspace(self.T * 0.25, self.T * 2, 10)
        time_costs = []
        for t in time_range:
            orig = self.T
            self.T = t
            time_costs.append({"time": t, "cost": self.expected_cost()["total_cost"]})
            self.T = orig
        results["time"] = pd.DataFrame(time_costs)

        return results
