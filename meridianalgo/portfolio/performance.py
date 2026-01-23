"""
Performance attribution and analysis system.
Implements Brinson attribution, sector/style analysis, and benchmark comparison.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Result of performance attribution analysis."""

    total_return: float
    benchmark_return: float
    active_return: float
    attribution_breakdown: pd.DataFrame
    summary_stats: Dict[str, float]
    method: str
    success: bool
    message: str
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    tracking_error: float
    information_ratio: float
    beta: float
    alpha: float
    up_capture: float
    down_capture: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class BaseAttributionAnalyzer(ABC):
    """Abstract base class for attribution analyzers."""

    @abstractmethod
    def analyze(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, **kwargs
    ) -> AttributionResult:
        """Perform attribution analysis."""
        pass


class BrinsonAttributionAnalyzer(BaseAttributionAnalyzer):
    """Brinson attribution analysis for factor decomposition."""

    def __init__(self):
        self.name = "BrinsonAttribution"

    def analyze(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        sector_returns: pd.DataFrame,
        **kwargs,
    ) -> AttributionResult:
        """
        Perform Brinson attribution analysis.

        Args:
            portfolio_returns: Portfolio returns time series
            benchmark_returns: Benchmark returns time series
            portfolio_weights: Portfolio weights by sector/factor over time
            benchmark_weights: Benchmark weights by sector/factor over time
            sector_returns: Sector/factor returns over time

        Returns:
            AttributionResult with Brinson attribution breakdown
        """
        # Align all data to common dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        common_dates = common_dates.intersection(portfolio_weights.index)
        common_dates = common_dates.intersection(benchmark_weights.index)
        common_dates = common_dates.intersection(sector_returns.index)

        if len(common_dates) == 0:
            return AttributionResult(
                total_return=0.0,
                benchmark_return=0.0,
                active_return=0.0,
                attribution_breakdown=pd.DataFrame(),
                summary_stats={},
                method=self.name,
                success=False,
                message="No common dates found between inputs",
            )

        # Filter data to common dates
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]
        port_weights = portfolio_weights.loc[common_dates]
        bench_weights = benchmark_weights.loc[common_dates]
        sector_ret = sector_returns.loc[common_dates]

        # Calculate attribution components
        attribution_data = []

        for date in common_dates:
            # Get weights and returns for this date
            pw = port_weights.loc[date]
            bw = bench_weights.loc[date]
            sr = sector_ret.loc[date]

            # Ensure all sectors are present
            sectors = pw.index.union(bw.index).union(sr.index)
            pw = pw.reindex(sectors, fill_value=0.0)
            bw = bw.reindex(sectors, fill_value=0.0)
            sr = sr.reindex(sectors, fill_value=0.0)

            # Brinson attribution components
            # Asset Allocation Effect: (wp - wb) * rb
            allocation_effect = (pw - bw) * sr

            # Security Selection Effect: wb * (rp - rb)
            # For simplicity, assume sector returns represent security selection
            selection_effect = bw * (
                sr - sr
            )  # This would need actual security-level data

            # Interaction Effect: (wp - wb) * (rp - rb)
            interaction_effect = (pw - bw) * (
                sr - sr
            )  # This would need actual security-level data

            date_attribution = pd.DataFrame(
                {
                    "date": date,
                    "sector": sectors,
                    "portfolio_weight": pw,
                    "benchmark_weight": bw,
                    "sector_return": sr,
                    "allocation_effect": allocation_effect,
                    "selection_effect": selection_effect,
                    "interaction_effect": interaction_effect,
                    "total_effect": allocation_effect
                    + selection_effect
                    + interaction_effect,
                }
            )

            attribution_data.append(date_attribution)

        # Combine all attribution data
        attribution_df = pd.concat(attribution_data, ignore_index=True)

        # Calculate summary statistics
        total_allocation = attribution_df.groupby("sector")["allocation_effect"].sum()
        total_selection = attribution_df.groupby("sector")["selection_effect"].sum()
        total_interaction = attribution_df.groupby("sector")["interaction_effect"].sum()

        summary_attribution = pd.DataFrame(
            {
                "allocation_effect": total_allocation,
                "selection_effect": total_selection,
                "interaction_effect": total_interaction,
                "total_effect": total_allocation + total_selection + total_interaction,
            }
        ).fillna(0.0)

        # Calculate performance metrics
        total_return = (1 + port_ret).prod() - 1
        benchmark_return = (1 + bench_ret).prod() - 1
        active_return = total_return - benchmark_return

        summary_stats = {
            "total_allocation_effect": total_allocation.sum(),
            "total_selection_effect": total_selection.sum(),
            "total_interaction_effect": total_interaction.sum(),
            "total_active_return": active_return,
            "explained_active_return": summary_attribution["total_effect"].sum(),
            "unexplained_return": active_return
            - summary_attribution["total_effect"].sum(),
        }

        return AttributionResult(
            total_return=total_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            attribution_breakdown=summary_attribution,
            summary_stats=summary_stats,
            method=self.name,
            success=True,
            message="Brinson attribution analysis completed successfully",
            metadata={
                "num_periods": len(common_dates),
                "num_sectors": len(sectors),
                "detailed_attribution": attribution_df,
            },
        )


class SectorStyleAttributionAnalyzer(BaseAttributionAnalyzer):
    """Sector and style attribution analysis."""

    def __init__(self):
        self.name = "SectorStyleAttribution"

    def analyze(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_exposures: pd.DataFrame,
        benchmark_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame,
        **kwargs,
    ) -> AttributionResult:
        """
        Perform sector and style attribution analysis.

        Args:
            portfolio_returns: Portfolio returns time series
            benchmark_returns: Benchmark returns time series
            portfolio_exposures: Portfolio exposures to factors (sectors, styles)
            benchmark_exposures: Benchmark exposures to factors
            factor_returns: Factor returns over time

        Returns:
            AttributionResult with sector/style attribution
        """
        # Align data
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        common_dates = common_dates.intersection(portfolio_exposures.index)
        common_dates = common_dates.intersection(benchmark_exposures.index)
        common_dates = common_dates.intersection(factor_returns.index)

        if len(common_dates) == 0:
            return AttributionResult(
                total_return=0.0,
                benchmark_return=0.0,
                active_return=0.0,
                attribution_breakdown=pd.DataFrame(),
                summary_stats={},
                method=self.name,
                success=False,
                message="No common dates found between inputs",
            )

        # Filter data
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]
        port_exp = portfolio_exposures.loc[common_dates]
        bench_exp = benchmark_exposures.loc[common_dates]
        factor_ret = factor_returns.loc[common_dates]

        # Calculate factor contributions
        attribution_results = []

        for factor in factor_ret.columns:
            if factor in port_exp.columns and factor in bench_exp.columns:
                # Active exposure to this factor
                active_exposure = port_exp[factor] - bench_exp[factor]

                # Factor return contribution
                factor_contribution = active_exposure * factor_ret[factor]

                attribution_results.append(
                    {
                        "factor": factor,
                        "avg_active_exposure": active_exposure.mean(),
                        "factor_return": factor_ret[factor].mean(),
                        "total_contribution": factor_contribution.sum(),
                        "contribution_volatility": factor_contribution.std(),
                    }
                )

        attribution_df = pd.DataFrame(attribution_results)
        if len(attribution_df) > 0:
            attribution_df = attribution_df.set_index("factor")

        # Calculate summary statistics
        total_return = (1 + port_ret).prod() - 1
        benchmark_return = (1 + bench_ret).prod() - 1
        active_return = total_return - benchmark_return

        explained_return = (
            attribution_df["total_contribution"].sum()
            if len(attribution_df) > 0
            else 0.0
        )

        summary_stats = {
            "total_active_return": active_return,
            "explained_active_return": explained_return,
            "unexplained_return": active_return - explained_return,
            "num_factors": len(attribution_df),
            "largest_contributor": (
                attribution_df["total_contribution"].idxmax()
                if len(attribution_df) > 0
                else None
            ),
            "largest_detractor": (
                attribution_df["total_contribution"].idxmin()
                if len(attribution_df) > 0
                else None
            ),
        }

        return AttributionResult(
            total_return=total_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            attribution_breakdown=attribution_df,
            summary_stats=summary_stats,
            method=self.name,
            success=True,
            message="Sector/style attribution analysis completed successfully",
        )


class BenchmarkComparisonAnalyzer:
    """Benchmark comparison and tracking error analysis."""

    def __init__(self):
        self.name = "BenchmarkComparison"

    def analyze(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive benchmark comparison analysis.

        Args:
            portfolio_returns: Portfolio returns time series
            benchmark_returns: Benchmark returns time series
            risk_free_rate: Risk-free rate for calculations

        Returns:
            Dictionary with comprehensive comparison metrics
        """
        # Align returns
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]

        if len(common_dates) == 0:
            return {"success": False, "message": "No common dates found"}

        # Calculate basic metrics
        port_total_return = (1 + port_ret).prod() - 1
        bench_total_return = (1 + bench_ret).prod() - 1
        active_return = port_total_return - bench_total_return

        # Annualized metrics
        periods_per_year = self._infer_frequency(port_ret.index)
        port_ann_return = (1 + port_total_return) ** (
            periods_per_year / len(port_ret)
        ) - 1
        bench_ann_return = (1 + bench_total_return) ** (
            periods_per_year / len(bench_ret)
        ) - 1

        # Volatility metrics
        port_vol = port_ret.std() * np.sqrt(periods_per_year)
        bench_vol = bench_ret.std() * np.sqrt(periods_per_year)
        tracking_error = (port_ret - bench_ret).std() * np.sqrt(periods_per_year)

        # Risk-adjusted metrics
        port_sharpe = (
            (port_ann_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        )
        bench_sharpe = (
            (bench_ann_return - risk_free_rate) / bench_vol if bench_vol > 0 else 0
        )
        information_ratio = (
            (port_ann_return - bench_ann_return) / tracking_error
            if tracking_error > 0
            else 0
        )

        # Beta and alpha
        if len(port_ret) > 1 and bench_ret.var() > 0:
            beta = np.cov(port_ret, bench_ret)[0, 1] / bench_ret.var()
            alpha = port_ann_return - (
                risk_free_rate + beta * (bench_ann_return - risk_free_rate)
            )
        else:
            beta = 0.0
            alpha = 0.0

        # Drawdown analysis
        port_cumret = (1 + port_ret).cumprod()
        port_drawdown = port_cumret / port_cumret.cummax() - 1
        max_drawdown = port_drawdown.min()

        bench_cumret = (1 + bench_ret).cumprod()
        bench_drawdown = bench_cumret / bench_cumret.cummax() - 1
        bench_max_drawdown = bench_drawdown.min()

        # Up/Down capture ratios
        up_periods = bench_ret > 0
        down_periods = bench_ret < 0

        if up_periods.sum() > 0 and bench_ret[up_periods].mean() != 0:
            up_capture = port_ret[up_periods].mean() / bench_ret[up_periods].mean()
        else:
            up_capture = 0.0

        if down_periods.sum() > 0 and bench_ret[down_periods].mean() != 0:
            down_capture = (
                port_ret[down_periods].mean() / bench_ret[down_periods].mean()
            )
        else:
            down_capture = 0.0

        # Win/Loss statistics
        active_returns = port_ret - bench_ret
        wins = active_returns > 0
        losses = active_returns < 0

        win_rate = wins.sum() / len(active_returns) if len(active_returns) > 0 else 0
        avg_win = active_returns[wins].mean() if wins.sum() > 0 else 0
        avg_loss = active_returns[losses].mean() if losses.sum() > 0 else 0

        # VaR and CVaR
        var_95 = np.percentile(port_ret, 5)
        cvar_95 = (
            port_ret[port_ret <= var_95].mean()
            if (port_ret <= var_95).sum() > 0
            else var_95
        )

        # Sortino ratio (downside deviation)
        downside_returns = port_ret[port_ret < risk_free_rate / periods_per_year]
        downside_deviation = (
            downside_returns.std() * np.sqrt(periods_per_year)
            if len(downside_returns) > 0
            else 0
        )
        sortino_ratio = (
            (port_ann_return - risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else 0
        )

        # Calmar ratio
        calmar_ratio = port_ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "success": True,
            "portfolio_metrics": {
                "total_return": port_total_return,
                "annualized_return": port_ann_return,
                "volatility": port_vol,
                "sharpe_ratio": port_sharpe,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "cvar_95": cvar_95,
            },
            "benchmark_metrics": {
                "total_return": bench_total_return,
                "annualized_return": bench_ann_return,
                "volatility": bench_vol,
                "sharpe_ratio": bench_sharpe,
                "max_drawdown": bench_max_drawdown,
            },
            "relative_metrics": {
                "active_return": active_return,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "beta": beta,
                "alpha": alpha,
                "up_capture": up_capture,
                "down_capture": down_capture,
            },
            "win_loss_stats": {
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            },
            "periods_analyzed": len(common_dates),
            "frequency": periods_per_year,
        }

    def _infer_frequency(self, index: pd.DatetimeIndex) -> int:
        """Infer the frequency of the time series and return periods per year."""
        if len(index) < 2:
            return 252  # Default to daily

        # Calculate median time difference
        time_diffs = index[1:] - index[:-1]
        median_diff = time_diffs.median()

        if median_diff <= timedelta(days=1):
            return 252  # Daily
        elif median_diff <= timedelta(days=7):
            return 52  # Weekly
        elif median_diff <= timedelta(days=31):
            return 12  # Monthly
        elif median_diff <= timedelta(days=92):
            return 4  # Quarterly
        else:
            return 1  # Annual


class PerformanceAnalyzer:
    """Main performance analysis class combining all attribution methods."""

    def __init__(self):
        self.brinson_analyzer = BrinsonAttributionAnalyzer()
        self.sector_style_analyzer = SectorStyleAttributionAnalyzer()
        self.benchmark_analyzer = BenchmarkComparisonAnalyzer()

    def comprehensive_analysis(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: pd.DataFrame = None,
        benchmark_weights: pd.DataFrame = None,
        sector_returns: pd.DataFrame = None,
        factor_exposures: Dict[str, pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis.

        Args:
            portfolio_returns: Portfolio returns time series
            benchmark_returns: Benchmark returns time series
            portfolio_weights: Portfolio sector/factor weights over time
            benchmark_weights: Benchmark sector/factor weights over time
            sector_returns: Sector returns over time
            factor_exposures: Dictionary of factor exposures
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with all analysis results
        """
        results = {}

        # 1. Benchmark comparison analysis (always available)
        results["benchmark_comparison"] = self.benchmark_analyzer.analyze(
            portfolio_returns, benchmark_returns, risk_free_rate
        )

        # 2. Brinson attribution (if weights and sector returns available)
        if (
            portfolio_weights is not None
            and benchmark_weights is not None
            and sector_returns is not None
        ):
            results["brinson_attribution"] = self.brinson_analyzer.analyze(
                portfolio_returns,
                benchmark_returns,
                portfolio_weights,
                benchmark_weights,
                sector_returns,
            )

        # 3. Sector/Style attribution (if factor exposures available)
        if factor_exposures is not None:
            for factor_type, exposures in factor_exposures.items():
                if (
                    "portfolio" in exposures
                    and "benchmark" in exposures
                    and "returns" in exposures
                ):
                    results[f"{factor_type}_attribution"] = (
                        self.sector_style_analyzer.analyze(
                            portfolio_returns,
                            benchmark_returns,
                            exposures["portfolio"],
                            exposures["benchmark"],
                            exposures["returns"],
                        )
                    )

        # 4. Rolling analysis
        results["rolling_analysis"] = self._rolling_analysis(
            portfolio_returns,
            benchmark_returns,
            window=252,  # 1 year rolling
        )

        return results

    def _rolling_analysis(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 252,
    ) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics."""
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]

        if len(common_dates) < window:
            return {
                "message": f"Insufficient data for rolling analysis (need {window} periods)"
            }

        rolling_metrics = {}

        # Rolling returns
        rolling_metrics["portfolio_return"] = port_ret.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )
        rolling_metrics["benchmark_return"] = bench_ret.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )
        rolling_metrics["active_return"] = (
            rolling_metrics["portfolio_return"] - rolling_metrics["benchmark_return"]
        )

        # Rolling volatility
        rolling_metrics["portfolio_volatility"] = port_ret.rolling(
            window
        ).std() * np.sqrt(252)
        rolling_metrics["benchmark_volatility"] = bench_ret.rolling(
            window
        ).std() * np.sqrt(252)
        rolling_metrics["tracking_error"] = (port_ret - bench_ret).rolling(
            window
        ).std() * np.sqrt(252)

        # Rolling Sharpe ratio
        rolling_metrics["portfolio_sharpe"] = (
            rolling_metrics["portfolio_return"]
            / rolling_metrics["portfolio_volatility"]
        )
        rolling_metrics["benchmark_sharpe"] = (
            rolling_metrics["benchmark_return"]
            / rolling_metrics["benchmark_volatility"]
        )

        # Rolling information ratio
        rolling_metrics["information_ratio"] = (
            rolling_metrics["active_return"] / rolling_metrics["tracking_error"]
        )

        return rolling_metrics

    def create_performance_report(
        self,
        analysis_results: Dict[str, Any],
        portfolio_name: str = "Portfolio",
        benchmark_name: str = "Benchmark",
    ) -> str:
        """Create a formatted performance report."""
        report = []
        report.append(
            f"Performance Analysis Report: {portfolio_name} vs {benchmark_name}"
        )
        report.append("=" * 80)

        # Benchmark comparison summary
        if "benchmark_comparison" in analysis_results:
            bc = analysis_results["benchmark_comparison"]
            if bc["success"]:
                pm = bc["portfolio_metrics"]
                bm = bc["benchmark_metrics"]
                rm = bc["relative_metrics"]

                report.append("\n1. PERFORMANCE SUMMARY")
                report.append("-" * 40)
                report.append(f"Portfolio Total Return:    {pm['total_return']:8.2%}")
                report.append(f"Benchmark Total Return:    {bm['total_return']:8.2%}")
                report.append(f"Active Return:             {rm['active_return']:8.2%}")
                report.append("")
                report.append(f"Portfolio Sharpe Ratio:    {pm['sharpe_ratio']:8.2f}")
                report.append(f"Benchmark Sharpe Ratio:    {bm['sharpe_ratio']:8.2f}")
                report.append(
                    f"Information Ratio:         {rm['information_ratio']:8.2f}"
                )
                report.append("")
                report.append(f"Tracking Error:            {rm['tracking_error']:8.2%}")
                report.append(f"Beta:                      {rm['beta']:8.2f}")
                report.append(f"Alpha:                     {rm['alpha']:8.2%}")

        # Attribution analysis
        if "brinson_attribution" in analysis_results:
            ba = analysis_results["brinson_attribution"]
            if ba.success:
                report.append("\n2. BRINSON ATTRIBUTION ANALYSIS")
                report.append("-" * 40)
                report.append(
                    f"Total Active Return:       {ba.summary_stats['total_active_return']:8.2%}"
                )
                report.append(
                    f"Allocation Effect:         {ba.summary_stats['total_allocation_effect']:8.2%}"
                )
                report.append(
                    f"Selection Effect:          {ba.summary_stats['total_selection_effect']:8.2%}"
                )
                report.append(
                    f"Interaction Effect:        {ba.summary_stats['total_interaction_effect']:8.2%}"
                )

        return "\n".join(report)


# Example usage and utility functions
def create_sample_attribution_data():
    """Create sample data for attribution analysis testing."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")

    # Sample portfolio and benchmark returns
    np.random.seed(42)
    portfolio_returns = pd.Series(
        np.random.normal(0.0008, 0.015, len(dates)), index=dates, name="portfolio"
    )
    benchmark_returns = pd.Series(
        np.random.normal(0.0006, 0.012, len(dates)), index=dates, name="benchmark"
    )

    # Sample sector weights and returns
    sectors = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]

    # Portfolio weights (varying over time)
    portfolio_weights = pd.DataFrame(
        np.random.dirichlet([1, 1, 1, 1, 1], len(dates)), index=dates, columns=sectors
    )

    # Benchmark weights (more stable)
    benchmark_base_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    benchmark_weights = pd.DataFrame(
        np.tile(benchmark_base_weights, (len(dates), 1))
        + np.random.normal(0, 0.02, (len(dates), len(sectors))),
        index=dates,
        columns=sectors,
    )
    # Normalize to sum to 1
    benchmark_weights = benchmark_weights.div(benchmark_weights.sum(axis=1), axis=0)

    # Sector returns
    sector_returns = pd.DataFrame(
        np.random.normal(0.0006, 0.02, (len(dates), len(sectors))),
        index=dates,
        columns=sectors,
    )

    return {
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "portfolio_weights": portfolio_weights,
        "benchmark_weights": benchmark_weights,
        "sector_returns": sector_returns,
    }


if __name__ == "__main__":
    # Example usage
    print("Performance Attribution System Example")
    print("=" * 50)

    # Create sample data
    data = create_sample_attribution_data()

    # Initialize analyzer
    analyzer = PerformanceAnalyzer()

    # Perform comprehensive analysis
    results = analyzer.comprehensive_analysis(
        portfolio_returns=data["portfolio_returns"],
        benchmark_returns=data["benchmark_returns"],
        portfolio_weights=data["portfolio_weights"],
        benchmark_weights=data["benchmark_weights"],
        sector_returns=data["sector_returns"],
    )

    # Generate report
    report = analyzer.create_performance_report(results)
    print(report)

    # Print attribution breakdown if available
    if "brinson_attribution" in results and results["brinson_attribution"].success:
        print("\n3. SECTOR ATTRIBUTION BREAKDOWN")
        print("-" * 40)
        attribution_df = results["brinson_attribution"].attribution_breakdown
        print(attribution_df.round(4))
