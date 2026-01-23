"""
Attribution Analytics Module

Performance attribution including Brinson attribution, factor attribution,
and contribution analysis.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class AttributionResult:
    """Container for attribution results."""

    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active: float


class PerformanceAttribution:
    """
    General performance attribution framework.

    Decomposes portfolio returns into various sources including:
    - Asset allocation decisions
    - Security selection
    - Factor exposures
    - Timing effects
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: Optional[pd.DataFrame] = None,
        benchmark_weights: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize PerformanceAttribution.

        Args:
            portfolio_returns: Portfolio returns series
            benchmark_returns: Benchmark returns series
            portfolio_weights: Portfolio weights over time
            benchmark_weights: Benchmark weights over time
        """
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.portfolio_weights = portfolio_weights
        self.benchmark_weights = benchmark_weights

    def active_return(self) -> float:
        """Calculate active return (portfolio - benchmark)."""
        return self.portfolio_returns.mean() - self.benchmark_returns.mean()

    def tracking_error(self) -> float:
        """Calculate tracking error."""
        active = self.portfolio_returns - self.benchmark_returns
        return active.std() * np.sqrt(252)

    def information_ratio(self) -> float:
        """Calculate information ratio."""
        te = self.tracking_error()
        return self.active_return() * 252 / te if te > 0 else 0


class BrinsonAttribution:
    """
    Brinson-Hood-Beebower attribution model.

    Decomposes active return into:
    - Allocation effect: Over/underweighting sectors that outperform
    - Selection effect: Picking securities within sectors
    - Interaction effect: Combined allocation and selection

    Example:
        >>> attribution = BrinsonAttribution(
        ...     portfolio_returns=port_returns,
        ...     benchmark_returns=bench_returns,
        ...     portfolio_weights=port_weights,
        ...     benchmark_weights=bench_weights
        ... )
        >>> result = attribution.calculate()
    """

    def __init__(
        self,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
    ):
        """
        Initialize BrinsonAttribution.

        Args:
            portfolio_returns: Returns by sector/asset (time x sector)
            benchmark_returns: Benchmark returns by sector
            portfolio_weights: Portfolio weights by sector
            benchmark_weights: Benchmark weights by sector
        """
        self.port_returns = portfolio_returns
        self.bench_returns = benchmark_returns
        self.port_weights = portfolio_weights
        self.bench_weights = benchmark_weights

    def allocation_effect(self) -> pd.Series:
        """
        Calculate allocation effect.

        Measures the impact of over/underweighting sectors
        relative to the benchmark.
        """
        weight_diff = self.port_weights - self.bench_weights
        return (weight_diff * (self.bench_returns - self.bench_returns.mean())).sum(
            axis=1
        )

    def selection_effect(self) -> pd.Series:
        """
        Calculate selection effect.

        Measures the impact of security selection within sectors.
        """
        return_diff = self.port_returns - self.bench_returns
        return (self.bench_weights * return_diff).sum(axis=1)

    def interaction_effect(self) -> pd.Series:
        """
        Calculate interaction effect.

        Combined effect of allocation and selection decisions.
        """
        weight_diff = self.port_weights - self.bench_weights
        return_diff = self.port_returns - self.bench_returns
        return (weight_diff * return_diff).sum(axis=1)

    def calculate(self) -> AttributionResult:
        """
        Calculate full Brinson attribution.

        Returns:
            AttributionResult with all effects
        """
        allocation = self.allocation_effect().mean()
        selection = self.selection_effect().mean()
        interaction = self.interaction_effect().mean()
        total = allocation + selection + interaction

        return AttributionResult(
            allocation_effect=allocation,
            selection_effect=selection,
            interaction_effect=interaction,
            total_active=total,
        )

    def summary(self) -> pd.DataFrame:
        """Generate attribution summary."""
        result = self.calculate()

        return pd.DataFrame(
            {
                "Effect": [
                    result.allocation_effect,
                    result.selection_effect,
                    result.interaction_effect,
                    result.total_active,
                ],
                "Percentage": [
                    (
                        result.allocation_effect / result.total_active
                        if result.total_active != 0
                        else 0
                    ),
                    (
                        result.selection_effect / result.total_active
                        if result.total_active != 0
                        else 0
                    ),
                    (
                        result.interaction_effect / result.total_active
                        if result.total_active != 0
                        else 0
                    ),
                    1.0,
                ],
            },
            index=["Allocation", "Selection", "Interaction", "Total"],
        )


class FactorAttribution:
    """
    Factor-based performance attribution.

    Decomposes returns into factor and residual (alpha) components
    using a multi-factor model.

    Example:
        >>> attribution = FactorAttribution(returns, factor_returns)
        >>> result = attribution.calculate()
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize FactorAttribution.

        Args:
            portfolio_returns: Portfolio returns series
            factor_returns: Factor returns DataFrame (time x factors)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = portfolio_returns
        self.factors = factor_returns
        self.risk_free_rate = risk_free_rate
        self._fitted = False
        self._alpha = None
        self._betas = None
        self._r_squared = None

    def fit(self) -> "FactorAttribution":
        """Fit the factor model."""
        # Align data
        aligned = pd.concat([self.returns, self.factors], axis=1).dropna()
        Y = aligned.iloc[:, 0].values
        X = aligned.iloc[:, 1:].values

        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            betas, residuals, rank, s = np.linalg.lstsq(X_with_const, Y, rcond=None)
        except Exception:
            betas = np.zeros(X_with_const.shape[1])

        self._alpha = betas[0]
        self._betas = dict(zip(self.factors.columns, betas[1:]))

        # Calculate R-squared
        y_pred = X_with_const @ betas
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        self._r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        self._fitted = True
        return self

    @property
    def alpha(self) -> float:
        """Get alpha (excess return not explained by factors)."""
        if not self._fitted:
            self.fit()
        return self._alpha

    @property
    def betas(self) -> Dict[str, float]:
        """Get factor exposures (betas)."""
        if not self._fitted:
            self.fit()
        return self._betas

    @property
    def r_squared(self) -> float:
        """Get R-squared of factor model."""
        if not self._fitted:
            self.fit()
        return self._r_squared

    def factor_contribution(self) -> Dict[str, float]:
        """
        Calculate return contribution from each factor.

        Returns:
            Dictionary of factor name to return contribution
        """
        if not self._fitted:
            self.fit()

        contributions = {}
        for factor, beta in self._betas.items():
            contributions[factor] = beta * self.factors[factor].mean() * 252

        return contributions

    def residual_return(self) -> pd.Series:
        """Get residual (unexplained) returns."""
        if not self._fitted:
            self.fit()

        # Calculate factor contribution
        aligned = pd.concat([self.returns, self.factors], axis=1).dropna()
        y = aligned.iloc[:, 0]

        factor_contribution = sum(
            self._betas[col] * aligned[col] for col in self.factors.columns
        )

        return y - self._alpha - factor_contribution

    def calculate(self) -> Dict[str, Any]:
        """
        Calculate full factor attribution.

        Returns:
            Dictionary with all attribution results
        """
        if not self._fitted:
            self.fit()

        contributions = self.factor_contribution()

        return {
            "alpha": self.alpha * 252,  # Annualized
            "alpha_pct": (
                self.alpha * 252 / self.returns.mean() / 252
                if self.returns.mean() != 0
                else 0
            ),
            "betas": self.betas,
            "factor_contributions": contributions,
            "r_squared": self.r_squared,
            "residual_vol": self.residual_return().std() * np.sqrt(252),
        }

    def summary(self) -> pd.DataFrame:
        """Generate attribution summary."""
        result = self.calculate()

        rows = [
            ("Alpha (annualized)", result["alpha"]),
            ("R-squared", result["r_squared"]),
            ("Residual Volatility", result["residual_vol"]),
        ]

        for factor, beta in result["betas"].items():
            rows.append((f"{factor} Beta", beta))
            rows.append(
                (f"{factor} Contribution", result["factor_contributions"][factor])
            )

        return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")
