"""
Kelly Criterion position sizing for portfolio management.

Provides full Kelly, fractional Kelly, and multi-asset Kelly optimization
for optimal bet sizing given expected returns and risk estimates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class KellyCriterion:
    """
    Kelly Criterion position sizing.

    Computes the theoretically optimal fraction of capital to allocate to each
    asset or strategy to maximize long-run geometric growth rate.

    For a single bet:
        f* = (p * b - q) / b  where p=win prob, b=win/loss ratio, q=1-p

    For multiple assets (continuous Kelly):
        f* = Sigma^{-1} * mu  (capped at 1.0 per asset)

    Args:
        fraction: Kelly fraction multiplier. 1.0 = full Kelly, 0.5 = half Kelly.
            Values < 1 reduce volatility at the cost of lower growth rate.
        max_position: Maximum allocation per asset (default 1.0).
        min_position: Minimum allocation per asset (default 0.0, no short).
    """

    def __init__(
        self,
        fraction: float = 0.5,
        max_position: float = 1.0,
        min_position: float = 0.0,
    ) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        if max_position <= 0:
            raise ValueError("max_position must be positive")

        self.fraction = fraction
        self.max_position = max_position
        self.min_position = min_position
        self._last_weights: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Single-asset Kelly
    # ------------------------------------------------------------------

    def single_asset(
        self,
        win_prob: float,
        win_loss_ratio: float,
    ) -> float:
        """
        Compute Kelly fraction for a single binary bet.

        Args:
            win_prob: Probability of winning (0 < p < 1).
            win_loss_ratio: Ratio of gain on win to loss on loss (b > 0).

        Returns:
            Optimal fraction of capital to bet (0 to max_position).

        Example:
            >>> kc = KellyCriterion(fraction=0.5)
            >>> kc.single_asset(win_prob=0.55, win_loss_ratio=1.0)
            0.05
        """
        if not 0 < win_prob < 1:
            raise ValueError("win_prob must be in (0, 1)")
        if win_loss_ratio <= 0:
            raise ValueError("win_loss_ratio must be positive")

        lose_prob = 1.0 - win_prob
        full_kelly = (win_prob * win_loss_ratio - lose_prob) / win_loss_ratio
        fractional = full_kelly * self.fraction
        return float(np.clip(fractional, self.min_position, self.max_position))

    # ------------------------------------------------------------------
    # Multi-asset continuous Kelly
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ) -> pd.Series:
        """
        Compute Kelly weights for a multi-asset portfolio.

        Uses the continuous-time Kelly criterion:
            f* = Sigma^{-1} * (mu - r)

        where mu is the vector of expected excess returns and Sigma is the
        covariance matrix of returns.

        Args:
            returns: DataFrame of asset returns (rows=time, cols=assets).
            risk_free_rate: Annualized risk-free rate. Defaults to 0.

        Returns:
            pd.Series of position sizes (fractional Kelly applied, clipped).
        """
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
        if returns.shape[0] < 2:
            raise ValueError("Need at least 2 observations")

        mu = returns.mean().values
        excess_mu = mu - risk_free_rate / 252  # daily risk-free

        cov = returns.cov().values
        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Covariance matrix not invertible: {e}") from e

        full_kelly = cov_inv @ excess_mu
        fractional_kelly = full_kelly * self.fraction
        clipped = np.clip(fractional_kelly, self.min_position, self.max_position)

        weights = pd.Series(clipped, index=returns.columns)

        # Normalize to sum to 1 if any weight is positive
        total = weights.sum()
        if total > 0:
            weights = weights / total * min(total, 1.0)

        self._last_weights = weights
        return weights

    # ------------------------------------------------------------------
    # Return-based Kelly (given expected return + volatility)
    # ------------------------------------------------------------------

    def from_moments(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Compute Kelly fraction from return moments (single asset, continuous).

        f* = (mu - r) / sigma^2

        Args:
            expected_return: Annualized expected return.
            volatility: Annualized volatility.
            risk_free_rate: Annualized risk-free rate.

        Returns:
            Optimal fractional position size.
        """
        if volatility <= 0:
            raise ValueError("volatility must be positive")

        excess_return = expected_return - risk_free_rate
        full_kelly = excess_return / (volatility**2)
        fractional = full_kelly * self.fraction
        return float(np.clip(fractional, self.min_position, self.max_position))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def growth_rate(
        self,
        expected_return: float,
        volatility: float,
        fraction: Optional[float] = None,
    ) -> float:
        """
        Expected geometric growth rate for a given Kelly fraction.

        g = mu * f - 0.5 * sigma^2 * f^2

        Args:
            expected_return: Annualized expected excess return.
            volatility: Annualized volatility.
            fraction: Kelly fraction. Uses self.fraction if None.

        Returns:
            Expected annualized geometric growth rate.
        """
        f = fraction if fraction is not None else self.fraction
        return float(expected_return * f - 0.5 * volatility**2 * f**2)

    @property
    def weights(self) -> Optional[pd.Series]:
        """Last computed weights from optimize()."""
        return self._last_weights

    def __repr__(self) -> str:
        return (
            f"KellyCriterion(fraction={self.fraction}, "
            f"max_position={self.max_position}, "
            f"min_position={self.min_position})"
        )
