"""
Performance attribution system for factor decomposition and benchmark analysis.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Result of performance attribution analysis."""

    total_return: float
    benchmark_return: float
    active_return: float
    attribution_breakdown: Dict[str, float]
    attribution_method: str
    period_start: datetime
    period_end: datetime
    metadata: Dict[str, Any] = None


@dataclass
class BrinsonAttributionResult:
    """Result of Brinson attribution analysis."""

    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    sector_breakdown: pd.DataFrame
    attribution_summary: Dict[str, float]


class BaseAttributionModel(ABC):
    """Abstract base class for performance attribution models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_attribution(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, **kwargs
    ) -> AttributionResult:
        """Calculate performance attribution."""
        pass


class BrinsonAttributionModel(BaseAttributionModel):
    """Brinson attribution model for sector/style analysis."""

    def __init__(self):
        super().__init__("BrinsonAttribution")

    def calculate_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        sector_returns: pd.DataFrame,
        **kwargs,
    ) -> BrinsonAttributionResult:
        """
        Calculate Brinson attribution analysis.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            portfolio_weights: Portfolio weights by sector/asset
            benchmark_weights: Benchmark weights by sector/asset
            sector_returns: Returns by sector/asset

        Returns:
            BrinsonAttributionResult with detailed attribution breakdown
        """

        # Align data
        common_index = portfolio_weights.index.intersection(benchmark_weights.index)
        common_columns = portfolio_weights.columns.intersection(
            benchmark_weights.columns
        )

        if len(common_index) == 0 or len(common_columns) == 0:
            raise ValueError(
                "No common periods or assets found between portfolio and benchmark"
            )

        pw = portfolio_weights.loc[common_index, common_columns]
        bw = benchmark_weights.loc[common_index, common_columns]
        sr = sector_returns.loc[common_index, common_columns]

        # Calculate attribution effects for each period
        allocation_effects = []
        selection_effects = []
        interaction_effects = []
        sector_attributions = []

        for date in common_index:
            pw_date = pw.loc[date]
            bw_date = bw.loc[date]
            sr_date = sr.loc[date]

            # Benchmark return for this period
            benchmark_return = np.sum(bw_date * sr_date)

            # Attribution effects by sector
            sector_data = []

            for sector in common_columns:
                # Allocation effect: (wp - wb) * rb
                allocation = (pw_date[sector] - bw_date[sector]) * sr_date[sector]

                # Selection effect: wb * (rp - rb)
                # For simplicity, assume sector return represents portfolio return in that sector
                selection = bw_date[sector] * (sr_date[sector] - benchmark_return)

                # Interaction effect: (wp - wb) * (rp - rb)
                interaction = (pw_date[sector] - bw_date[sector]) * (
                    sr_date[sector] - benchmark_return
                )

                sector_data.append(
                    {
                        "date": date,
                        "sector": sector,
                        "portfolio_weight": pw_date[sector],
                        "benchmark_weight": bw_date[sector],
                        "sector_return": sr_date[sector],
                        "allocation_effect": allocation,
                        "selection_effect": selection,
                        "interaction_effect": interaction,
                        "total_effect": allocation + selection + interaction,
                    }
                )

            sector_df = pd.DataFrame(sector_data)
            sector_attributions.append(sector_df)

            # Aggregate effects for this period
            allocation_effects.append(sector_df["allocation_effect"].sum())
            selection_effects.append(sector_df["selection_effect"].sum())
            interaction_effects.append(sector_df["interaction_effect"].sum())

        # Combine all sector attributions
        all_sector_data = pd.concat(sector_attributions, ignore_index=True)

        # Calculate summary statistics
        total_allocation = np.sum(allocation_effects)
        total_selection = np.sum(selection_effects)
        total_interaction = np.sum(interaction_effects)
        total_active_return = total_allocation + total_selection + total_interaction

        # Create sector breakdown summary
        sector_summary = (
            all_sector_data.groupby("sector")
            .agg(
                {
                    "allocation_effect": "sum",
                    "selection_effect": "sum",
                    "interaction_effect": "sum",
                    "total_effect": "sum",
                    "portfolio_weight": "mean",
                    "benchmark_weight": "mean",
                }
            )
            .round(4)
        )

        attribution_summary = {
            "allocation_effect": total_allocation,
            "selection_effect": total_selection,
            "interaction_effect": total_interaction,
            "total_active_return": total_active_return,
            "portfolio_return": portfolio_returns.sum(),
            "benchmark_return": benchmark_returns.sum(),
        }

        return BrinsonAttributionResult(
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            total_active_return=total_active_return,
            sector_breakdown=sector_summary,
            attribution_summary=attribution_summary,
        )


class FactorAttributionModel(BaseAttributionModel):
    """Factor-based performance attribution model."""

    def __init__(self, factor_model: str = "fama_french"):
        super().__init__("FactorAttribution")
        self.factor_model = factor_model

    def calculate_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Calculate factor-based attribution.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            factor_returns: Factor returns (e.g., Fama-French factors)

        Returns:
            AttributionResult with factor attribution breakdown
        """

        # Calculate active returns
        active_returns = portfolio_returns - benchmark_returns

        if factor_returns is None:
            # Create simple factor returns (market, size, value)
            factor_returns = self._create_default_factors(
                portfolio_returns, benchmark_returns
            )

        # Align data
        common_index = active_returns.index.intersection(factor_returns.index)
        if len(common_index) < 10:
            raise ValueError("Insufficient overlapping data for factor attribution")

        active_aligned = active_returns.loc[common_index]
        factors_aligned = factor_returns.loc[common_index]

        # Run factor regression
        X = factors_aligned.values
        y = active_aligned.values

        # Add constant term
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # Fit regression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_with_const, y)

        # Extract results
        alpha = reg.coef_[0]  # Intercept (alpha)
        factor_loadings = reg.coef_[1:]  # Factor loadings (betas)

        # Calculate factor contributions
        factor_contributions = {}
        ["Alpha"] + list(factor_returns.columns)

        # Alpha contribution
        factor_contributions["Alpha"] = alpha * len(active_aligned)

        # Factor contributions
        for i, factor_name in enumerate(factor_returns.columns):
            factor_return = factors_aligned[factor_name].sum()
            factor_loading = factor_loadings[i]
            contribution = factor_loading * factor_return
            factor_contributions[factor_name] = contribution

        # Calculate R-squared
        y_pred = reg.predict(X_with_const)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate tracking error
        tracking_error = active_aligned.std() * np.sqrt(252)  # Annualized

        # Calculate information ratio
        information_ratio = (
            (active_aligned.mean() * 252) / tracking_error if tracking_error > 0 else 0
        )

        return AttributionResult(
            total_return=portfolio_returns.sum(),
            benchmark_return=benchmark_returns.sum(),
            active_return=active_returns.sum(),
            attribution_breakdown=factor_contributions,
            attribution_method=self.name,
            period_start=common_index[0],
            period_end=common_index[-1],
            metadata={
                "factor_loadings": dict(zip(factor_returns.columns, factor_loadings)),
                "alpha": alpha,
                "r_squared": r_squared,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "factor_model": self.factor_model,
            },
        )

    def _create_default_factors(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> pd.DataFrame:
        """Create default factor returns if not provided."""

        # Simple factor construction
        market_factor = benchmark_returns.copy()

        # Size factor (simulated)
        size_factor = pd.Series(
            np.random.normal(0, 0.02, len(portfolio_returns)),
            index=portfolio_returns.index,
        )

        # Value factor (simulated)
        value_factor = pd.Series(
            np.random.normal(0, 0.015, len(portfolio_returns)),
            index=portfolio_returns.index,
        )

        return pd.DataFrame(
            {"Market": market_factor, "Size": size_factor, "Value": value_factor}
        )


class StyleAttributionModel(BaseAttributionModel):
    """Style-based performance attribution model."""

    def __init__(self):
        super().__init__("StyleAttribution")

    def calculate_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        style_exposures: pd.DataFrame = None,
        style_returns: pd.DataFrame = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Calculate style-based attribution.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            style_exposures: Style exposures (Growth, Value, Quality, etc.)
            style_returns: Style factor returns

        Returns:
            AttributionResult with style attribution breakdown
        """

        if style_exposures is None or style_returns is None:
            # Create default style factors
            style_exposures, style_returns = self._create_default_style_factors(
                portfolio_returns, benchmark_returns
            )

        # Calculate active returns
        active_returns = portfolio_returns - benchmark_returns

        # Align data
        common_index = active_returns.index.intersection(
            style_exposures.index
        ).intersection(style_returns.index)

        if len(common_index) < 10:
            raise ValueError("Insufficient data for style attribution")

        active_aligned = active_returns.loc[common_index]
        exposures_aligned = style_exposures.loc[common_index]
        returns_aligned = style_returns.loc[common_index]

        # Calculate style contributions
        style_contributions = {}

        for style in exposures_aligned.columns:
            if style in returns_aligned.columns:
                # Style contribution = exposure * style return
                exposure = exposures_aligned[style].mean()  # Average exposure
                style_return = returns_aligned[style].sum()  # Cumulative style return
                contribution = exposure * style_return
                style_contributions[style] = contribution

        # Calculate residual (unexplained) return
        total_style_contribution = sum(style_contributions.values())
        residual_return = active_aligned.sum() - total_style_contribution
        style_contributions["Residual"] = residual_return

        return AttributionResult(
            total_return=portfolio_returns.sum(),
            benchmark_return=benchmark_returns.sum(),
            active_return=active_returns.sum(),
            attribution_breakdown=style_contributions,
            attribution_method=self.name,
            period_start=common_index[0],
            period_end=common_index[-1],
            metadata={
                "average_exposures": exposures_aligned.mean().to_dict(),
                "style_returns": returns_aligned.sum().to_dict(),
                "total_style_contribution": total_style_contribution,
            },
        )

    def _create_default_style_factors(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create default style factors if not provided."""

        n_periods = len(portfolio_returns)

        # Style exposures (simulated)
        style_exposures = pd.DataFrame(
            {
                "Growth": np.random.uniform(0.3, 0.7, n_periods),
                "Value": np.random.uniform(0.2, 0.6, n_periods),
                "Quality": np.random.uniform(0.4, 0.8, n_periods),
                "Momentum": np.random.uniform(0.1, 0.5, n_periods),
                "Low_Volatility": np.random.uniform(0.3, 0.7, n_periods),
            },
            index=portfolio_returns.index,
        )

        # Style returns (simulated)
        style_returns = pd.DataFrame(
            {
                "Growth": np.random.normal(0, 0.02, n_periods),
                "Value": np.random.normal(0, 0.018, n_periods),
                "Quality": np.random.normal(0, 0.015, n_periods),
                "Momentum": np.random.normal(0, 0.025, n_periods),
                "Low_Volatility": np.random.normal(0, 0.012, n_periods),
            },
            index=portfolio_returns.index,
        )

        return style_exposures, style_returns


class CustomAttributionModel(BaseAttributionModel):
    """Custom attribution model for alternative strategies."""

    def __init__(self, attribution_factors: List[str]):
        super().__init__("CustomAttribution")
        self.attribution_factors = attribution_factors

    def calculate_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        custom_factors: pd.DataFrame,
        **kwargs,
    ) -> AttributionResult:
        """
        Calculate custom attribution based on user-defined factors.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            custom_factors: Custom factor returns/exposures

        Returns:
            AttributionResult with custom attribution breakdown
        """

        # Calculate active returns
        active_returns = portfolio_returns - benchmark_returns

        # Align data
        common_index = active_returns.index.intersection(custom_factors.index)
        if len(common_index) < 5:
            raise ValueError("Insufficient data for custom attribution")

        active_aligned = active_returns.loc[common_index]
        factors_aligned = custom_factors.loc[common_index]

        # Calculate correlations and contributions
        attribution_breakdown = {}

        for factor in self.attribution_factors:
            if factor in factors_aligned.columns:
                # Calculate correlation-based contribution
                correlation = active_aligned.corr(factors_aligned[factor])
                factor_volatility = factors_aligned[factor].std()
                active_volatility = active_aligned.std()

                # Contribution based on correlation and volatilities
                if not np.isnan(correlation) and active_volatility > 0:
                    contribution = (
                        correlation
                        * factor_volatility
                        / active_volatility
                        * active_aligned.sum()
                    )
                    attribution_breakdown[factor] = contribution
                else:
                    attribution_breakdown[factor] = 0.0

        # Calculate residual
        total_factor_contribution = sum(attribution_breakdown.values())
        attribution_breakdown["Residual"] = (
            active_aligned.sum() - total_factor_contribution
        )

        return AttributionResult(
            total_return=portfolio_returns.sum(),
            benchmark_return=benchmark_returns.sum(),
            active_return=active_returns.sum(),
            attribution_breakdown=attribution_breakdown,
            attribution_method=self.name,
            period_start=common_index[0],
            period_end=common_index[-1],
            metadata={
                "custom_factors": self.attribution_factors,
                "factor_correlations": {
                    factor: active_aligned.corr(factors_aligned[factor])
                    for factor in self.attribution_factors
                    if factor in factors_aligned.columns
                },
            },
        )


class PerformanceAttributionEngine:
    """Main engine for performance attribution analysis."""

    def __init__(self):
        self.attribution_models = {
            "brinson": BrinsonAttributionModel(),
            "factor": FactorAttributionModel(),
            "style": StyleAttributionModel(),
        }

    def add_custom_model(self, name: str, model: BaseAttributionModel):
        """Add custom attribution model."""
        self.attribution_models[name] = model

    def run_attribution_analysis(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        method: str = "factor",
        **kwargs,
    ) -> AttributionResult:
        """Run performance attribution analysis."""

        if method not in self.attribution_models:
            raise ValueError(f"Unknown attribution method: {method}")

        model = self.attribution_models[method]
        return model.calculate_attribution(
            portfolio_returns, benchmark_returns, **kwargs
        )

    def compare_attribution_methods(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        methods: List[str] = None,
        **kwargs,
    ) -> Dict[str, AttributionResult]:
        """Compare results from multiple attribution methods."""

        if methods is None:
            methods = ["factor", "style"]

        results = {}

        for method in methods:
            if method in self.attribution_models:
                try:
                    result = self.run_attribution_analysis(
                        portfolio_returns, benchmark_returns, method, **kwargs
                    )
                    results[method] = result
                except Exception as e:
                    logger.warning(f"Attribution method {method} failed: {e}")
                    continue

        return results

    def calculate_tracking_error_decomposition(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame = None,
    ) -> Dict[str, float]:
        """Decompose tracking error into factor components."""

        active_returns = portfolio_returns - benchmark_returns

        if factor_returns is None:
            # Use simple market factor
            factor_returns = pd.DataFrame({"Market": benchmark_returns})

        # Align data
        common_index = active_returns.index.intersection(factor_returns.index)
        active_aligned = active_returns.loc[common_index]
        factors_aligned = factor_returns.loc[common_index]

        # Calculate factor loadings
        X = factors_aligned.values
        y = active_aligned.values

        reg = LinearRegression()
        reg.fit(X, y)

        factor_loadings = reg.coef_
        residuals = y - reg.predict(X)

        # Decompose tracking error
        total_tracking_error = active_aligned.std() * np.sqrt(252)

        # Factor contributions to tracking error
        factor_contributions = {}

        for i, factor_name in enumerate(factor_returns.columns):
            factor_vol = factors_aligned[factor_name].std() * np.sqrt(252)
            factor_contribution = abs(factor_loadings[i]) * factor_vol
            factor_contributions[factor_name] = factor_contribution

        # Residual tracking error
        residual_tracking_error = np.std(residuals) * np.sqrt(252)
        factor_contributions["Residual"] = residual_tracking_error

        # Normalize to sum to total tracking error (approximately)
        total_factor_te = sum(factor_contributions.values())
        if total_factor_te > 0:
            scaling_factor = total_tracking_error / total_factor_te
            factor_contributions = {
                k: v * scaling_factor for k, v in factor_contributions.items()
            }

        factor_contributions["Total"] = total_tracking_error

        return factor_contributions

    def generate_attribution_report(
        self, attribution_result: AttributionResult, include_charts: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive attribution report."""

        report = {
            "summary": {
                "total_return": f"{attribution_result.total_return:.2%}",
                "benchmark_return": f"{attribution_result.benchmark_return:.2%}",
                "active_return": f"{attribution_result.active_return:.2%}",
                "attribution_method": attribution_result.attribution_method,
                "analysis_period": f"{attribution_result.period_start.date()} to {attribution_result.period_end.date()}",
            },
            "attribution_breakdown": {
                factor: f"{contribution:.2%}"
                for factor, contribution in attribution_result.attribution_breakdown.items()
            },
            "key_insights": [],
        }

        # Add key insights
        breakdown = attribution_result.attribution_breakdown

        # Find largest positive and negative contributors
        positive_contributors = {k: v for k, v in breakdown.items() if v > 0}
        negative_contributors = {k: v for k, v in breakdown.items() if v < 0}

        if positive_contributors:
            best_contributor = max(positive_contributors, key=positive_contributors.get)
            report["key_insights"].append(
                f"Largest positive contributor: {best_contributor} "
                f"({positive_contributors[best_contributor]:.2%})"
            )

        if negative_contributors:
            worst_contributor = min(
                negative_contributors, key=negative_contributors.get
            )
            report["key_insights"].append(
                f"Largest negative contributor: {worst_contributor} "
                f"({negative_contributors[worst_contributor]:.2%})"
            )

        # Add metadata if available
        if attribution_result.metadata:
            report["additional_metrics"] = attribution_result.metadata

        return report
