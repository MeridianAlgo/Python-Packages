"""
Benchmark-Relative Analytics

Active share, tracking error, information ratio, up/down capture ratios,
batting average, hit rate, and Brinson-Hood-Beebower (BHB) attribution.

References:
    Brinson, Hood & Beebower (1986) - Determinants of Portfolio Performance
    Cremers & Petajisto (2009) - Active Share
    Grinold & Kahn (1999) - Active Portfolio Management
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ActiveMetrics:
    """Container for benchmark-relative performance metrics."""

    active_return: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float
    capture_ratio: float
    batting_average: float
    max_active_drawdown: float
    beta: float
    alpha_annualized: float
    r_squared: float
    treynor_ratio: float


@dataclass
class BHBAttribution:
    """Brinson-Hood-Beebower attribution results."""

    allocation_effect: pd.Series
    selection_effect: pd.Series
    interaction_effect: pd.Series
    total_active_return: float
    total_allocation: float
    total_selection: float
    total_interaction: float


class BenchmarkAnalytics:
    """
    Benchmark-relative performance analytics for portfolio managers.

    Computes active return, tracking error, information ratio, capture ratios,
    and other metrics used to evaluate manager skill relative to a benchmark.

    Example:
        >>> analytics = BenchmarkAnalytics(portfolio_returns, benchmark_returns)
        >>> metrics = analytics.active_metrics()
        >>> print(f"Information Ratio: {metrics.information_ratio:.3f}")
        >>> print(f"Up Capture:        {metrics.up_capture:.2%}")
        >>> print(f"Down Capture:      {metrics.down_capture:.2%}")
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """
        Parameters
        ----------
        portfolio_returns : pd.Series
            Portfolio return series (decimal)
        benchmark_returns : pd.Series
            Benchmark return series (decimal)
        risk_free_rate : float
            Annual risk-free rate for alpha/Treynor calculation
        periods_per_year : int
            252 for daily, 12 for monthly, 4 for quarterly
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 10:
            raise ValueError("Insufficient overlapping observations (minimum 10)")

        self.portfolio = portfolio_returns.loc[common].dropna()
        self.benchmark = benchmark_returns.loc[common].dropna()
        self.rf = risk_free_rate
        self.periods = periods_per_year
        self._daily_rf = risk_free_rate / periods_per_year

    def active_returns(self) -> pd.Series:
        """Compute period-by-period active returns (portfolio - benchmark)."""
        return (self.portfolio - self.benchmark).rename("active_return")

    def tracking_error(self, annualize: bool = True) -> float:
        """
        Annualized tracking error (std dev of active returns).

        Parameters
        ----------
        annualize : bool
            If True, annualize tracking error
        """
        te = self.active_returns().std()
        if annualize:
            te = te * np.sqrt(self.periods)
        return float(te)

    def information_ratio(self) -> float:
        """
        Information ratio = annualized active return / tracking error.

        Measures the risk-adjusted outperformance per unit of active risk.
        """
        ar = self.active_returns()
        ann_active = ar.mean() * self.periods
        te = self.tracking_error(annualize=True)
        return float(ann_active / te) if te > 0 else 0.0

    def up_capture_ratio(self) -> float:
        """
        Up-market capture ratio.

        Measures portfolio return relative to benchmark during up-market periods.
        >100% = outperformance in up markets.
        """
        up_mask = self.benchmark > 0
        if up_mask.sum() == 0:
            return 0.0

        port_up = (1 + self.portfolio[up_mask]).prod() ** (
            self.periods / up_mask.sum()
        ) - 1
        bench_up = (1 + self.benchmark[up_mask]).prod() ** (
            self.periods / up_mask.sum()
        ) - 1

        return float(port_up / bench_up) if bench_up != 0 else 0.0

    def down_capture_ratio(self) -> float:
        """
        Down-market capture ratio.

        Measures portfolio return relative to benchmark during down-market periods.
        <100% = outperformance (smaller loss) in down markets.
        """
        down_mask = self.benchmark < 0
        if down_mask.sum() == 0:
            return 0.0

        port_down = (1 + self.portfolio[down_mask]).prod() ** (
            self.periods / down_mask.sum()
        ) - 1
        bench_down = (1 + self.benchmark[down_mask]).prod() ** (
            self.periods / down_mask.sum()
        ) - 1

        return float(port_down / bench_down) if bench_down != 0 else 0.0

    def batting_average(self) -> float:
        """
        Fraction of periods where portfolio outperformed the benchmark.
        """
        return float((self.portfolio > self.benchmark).mean())

    def beta_alpha(self) -> Tuple[float, float, float]:
        """
        OLS regression of portfolio excess returns on benchmark excess returns.

        Returns
        -------
        tuple
            (beta, annualized_alpha, r_squared)
        """
        port_excess = self.portfolio - self._daily_rf
        bench_excess = self.benchmark - self._daily_rf

        slope, intercept, r_value, p_value, stderr = stats.linregress(
            bench_excess.values, port_excess.values
        )

        return (
            float(slope),
            float(intercept * self.periods),
            float(r_value**2),
        )

    def treynor_ratio(self) -> float:
        """
        Treynor ratio = excess return / beta.

        Uses systematic (market) risk rather than total risk.
        """
        ann_return = (1 + self.portfolio).prod() ** (
            self.periods / len(self.portfolio)
        ) - 1
        beta, _, _ = self.beta_alpha()
        return float((ann_return - self.rf) / beta) if beta != 0 else 0.0

    def max_active_drawdown(self) -> float:
        """Maximum drawdown of the active (portfolio - benchmark) return stream."""
        ar = self.active_returns()
        cum_active = (1 + ar).cumprod()
        running_max = cum_active.cummax()
        drawdown = (cum_active - running_max) / running_max
        return float(drawdown.min())

    def active_metrics(self) -> ActiveMetrics:
        """
        Compute all benchmark-relative metrics.

        Returns
        -------
        ActiveMetrics
        """
        active = self.active_returns()
        ann_active = active.mean() * self.periods
        te = self.tracking_error()
        ir = self.information_ratio()
        up = self.up_capture_ratio()
        down = self.down_capture_ratio()
        capture = up / down if down != 0 else 0.0
        ba = self.batting_average()
        max_add = self.max_active_drawdown()
        beta, alpha, r2 = self.beta_alpha()
        treynor = self.treynor_ratio()

        return ActiveMetrics(
            active_return=ann_active,
            tracking_error=te,
            information_ratio=ir,
            up_capture=up,
            down_capture=down,
            capture_ratio=capture,
            batting_average=ba,
            max_active_drawdown=max_add,
            beta=beta,
            alpha_annualized=alpha,
            r_squared=r2,
            treynor_ratio=treynor,
        )

    def rolling_information_ratio(self, window: int = 252) -> pd.Series:
        """Rolling annualized information ratio over a lookback window."""
        active = self.active_returns()
        ann_active = active.rolling(window).mean() * self.periods
        te = active.rolling(window).std() * np.sqrt(self.periods)
        ir = ann_active / te
        return ir.rename("rolling_ir")

    def rolling_beta(self, window: int = 252) -> pd.Series:
        """Rolling beta using OLS over a lookback window."""
        port_ex = self.portfolio - self._daily_rf
        bench_ex = self.benchmark - self._daily_rf

        betas = pd.Series(np.nan, index=self.portfolio.index, name="rolling_beta")
        for i in range(window, len(self.portfolio) + 1):
            p = port_ex.iloc[i - window: i].values
            b = bench_ex.iloc[i - window: i].values
            cov = np.cov(p, b)
            betas.iloc[i - 1] = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan

        return betas


class ActiveShare:
    """
    Compute Active Share from portfolio and benchmark holdings.

    Active Share = 0.5 * sum(|w_portfolio - w_benchmark|)

    Values:
    - >90%: concentrated active
    - 60-90%: moderately active
    - <60%: closet indexer

    Reference:
        Cremers & Petajisto (2009)

    Example:
        >>> portfolio_weights = pd.Series({'AAPL': 0.10, 'MSFT': 0.15, ...})
        >>> benchmark_weights = pd.Series({'AAPL': 0.07, 'MSFT': 0.12, ...})
        >>> active_share = ActiveShare.compute(portfolio_weights, benchmark_weights)
        >>> print(f"Active Share: {active_share:.2%}")
    """

    @staticmethod
    def compute(
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
    ) -> float:
        """
        Compute active share between portfolio and benchmark.

        Parameters
        ----------
        portfolio_weights : pd.Series
            Portfolio holdings as fraction of portfolio value
        benchmark_weights : pd.Series
            Benchmark index weights

        Returns
        -------
        float
            Active share in [0, 1]
        """
        all_assets = portfolio_weights.index.union(benchmark_weights.index)
        p = portfolio_weights.reindex(all_assets).fillna(0)
        b = benchmark_weights.reindex(all_assets).fillna(0)
        return float(0.5 * (p - b).abs().sum())

    @staticmethod
    def categorize(active_share: float) -> str:
        """Classify manager style by active share level."""
        if active_share >= 0.90:
            return "concentrated_active"
        elif active_share >= 0.60:
            return "moderately_active"
        elif active_share >= 0.20:
            return "closet_indexer"
        return "index_fund"


class BrinsonAttribution:
    """
    Brinson-Hood-Beebower performance attribution.

    Decomposes active return into:
    - Allocation effect: Overweight/underweight sectors vs benchmark
    - Selection effect: Stock selection within sectors
    - Interaction effect: Combined allocation and selection

    Example:
        >>> attribution = BrinsonAttribution(
        ...     portfolio_weights=port_weights,
        ...     benchmark_weights=bench_weights,
        ...     portfolio_returns=port_sector_returns,
        ...     benchmark_returns=bench_sector_returns,
        ... )
        >>> result = attribution.compute()
        >>> print(result.allocation_effect)
    """

    def __init__(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ):
        """
        Parameters
        ----------
        portfolio_weights : pd.Series
            Portfolio sector/asset weights
        benchmark_weights : pd.Series
            Benchmark sector/asset weights
        portfolio_returns : pd.Series
            Portfolio return per sector/asset
        benchmark_returns : pd.Series
            Benchmark return per sector/asset
        """
        all_assets = (
            portfolio_weights.index
            .union(benchmark_weights.index)
            .union(portfolio_returns.index)
            .union(benchmark_returns.index)
        )

        self.pw = portfolio_weights.reindex(all_assets).fillna(0)
        self.bw = benchmark_weights.reindex(all_assets).fillna(0)
        self.pr = portfolio_returns.reindex(all_assets).fillna(0)
        self.br = benchmark_returns.reindex(all_assets).fillna(0)

    def compute(self) -> BHBAttribution:
        """
        Compute BHB attribution effects.

        Returns
        -------
        BHBAttribution
        """
        bench_total_return = (self.bw * self.br).sum()

        allocation = (self.pw - self.bw) * (self.br - bench_total_return)
        selection = self.bw * (self.pr - self.br)
        interaction = (self.pw - self.bw) * (self.pr - self.br)

        total_active = (self.pw * self.pr).sum() - bench_total_return

        return BHBAttribution(
            allocation_effect=allocation.rename("allocation"),
            selection_effect=selection.rename("selection"),
            interaction_effect=interaction.rename("interaction"),
            total_active_return=float(total_active),
            total_allocation=float(allocation.sum()),
            total_selection=float(selection.sum()),
            total_interaction=float(interaction.sum()),
        )
