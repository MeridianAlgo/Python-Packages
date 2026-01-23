"""
Tear Sheets Module

Comprehensive performance tear sheets inspired by pyfolio.
Generates beautiful, publication-ready performance reports.
"""

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class TearSheet:
    """
    Generate comprehensive performance tear sheets.

    Produces pyfolio-style performance reports with:
    - Cumulative returns
    - Rolling metrics
    - Drawdown analysis
    - Monthly returns heatmap
    - Distribution analysis
    - Position/trade analysis

    Example:
        >>> ts = TearSheet(returns, benchmark=spy_returns)
        >>> ts.create_full_tear_sheet()
        >>> ts.save('performance_report.pdf')
    """

    COLORS = {
        "strategy": "#2196F3",
        "benchmark": "#9E9E9E",
        "positive": "#4CAF50",
        "negative": "#F44336",
        "neutral": "#607D8B",
        "drawdown": "#E91E63",
    }

    def __init__(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        positions: Optional[pd.DataFrame] = None,
        transactions: Optional[pd.DataFrame] = None,
        gross_lev: Optional[pd.Series] = None,
        live_start_date: Optional[datetime] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """
        Initialize TearSheet.

        Args:
            returns: Daily returns series (decimal format)
            benchmark: Benchmark returns for comparison
            positions: Optional positions DataFrame (date x asset)
            transactions: Optional transactions DataFrame
            gross_lev: Optional gross leverage series
            live_start_date: Start of live trading (for in/out of sample)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading days per year
        """
        self.returns = returns.dropna()
        self.benchmark = benchmark.dropna() if benchmark is not None else None
        self.positions = positions
        self.transactions = transactions
        self.gross_lev = gross_lev
        self.live_start_date = live_start_date
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Lazy-load analytics
        self._performance_analyzer = None
        self._risk_analyzer = None

    @property
    def performance_analyzer(self):
        """Get or create performance analyzer."""
        if self._performance_analyzer is None:
            from .performance import PerformanceAnalyzer

            self._performance_analyzer = PerformanceAnalyzer(
                self.returns, self.benchmark, self.risk_free_rate, self.periods_per_year
            )
        return self._performance_analyzer

    @property
    def risk_analyzer(self):
        """Get or create risk analyzer."""
        if self._risk_analyzer is None:
            from .risk_analytics import RiskAnalyzer

            self._risk_analyzer = RiskAnalyzer(self.returns, self.periods_per_year)
        return self._risk_analyzer

    # =========================================================================
    # PLOTTING UTILITIES
    # =========================================================================

    def _setup_plot_style(self):
        """Set up matplotlib style."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        plt.style.use("seaborn-v0_8-whitegrid" if HAS_SEABORN else "ggplot")
        plt.rcParams["figure.figsize"] = [14, 8]
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 10

    def _create_text_box(self, ax, text: str, position: tuple = (0.05, 0.95)):
        """Add text box to plot."""
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            position[0],
            position[1],
            text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )

    # =========================================================================
    # INDIVIDUAL PLOTS
    # =========================================================================

    def plot_cumulative_returns(
        self, ax: Optional[Any] = None, logy: bool = False
    ) -> Any:
        """Plot cumulative returns."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        cumulative = (1 + self.returns).cumprod()
        ax.plot(
            cumulative.index,
            cumulative.values,
            color=self.COLORS["strategy"],
            linewidth=1.5,
            label="Strategy",
        )

        if self.benchmark is not None:
            cumulative.reindex(self.returns.index)
            bench_cum = (1 + self.benchmark).cumprod()
            ax.plot(
                bench_cum.index,
                bench_cum.values,
                color=self.COLORS["benchmark"],
                linewidth=1,
                linestyle="--",
                label="Benchmark",
                alpha=0.7,
            )

        if logy:
            ax.set_yscale("log")

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title("Cumulative Returns")
        ax.legend(loc="upper left")
        ax.axhline(1.0, color="black", linestyle="-", linewidth=0.5)

        # Add performance annotation
        total_ret = self.performance_analyzer.total_return()
        sharpe = self.performance_analyzer.sharpe_ratio()
        max_dd = self.performance_analyzer.max_drawdown()

        text = (
            f"Total Return: {total_ret:.1%}\nSharpe: {sharpe:.2f}\nMax DD: {max_dd:.1%}"
        )
        self._create_text_box(ax, text)

        return ax

    def plot_drawdown(self, ax: Optional[Any] = None) -> Any:
        """Plot underwater chart (drawdown)."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        drawdown = self.performance_analyzer.drawdown_series()

        ax.fill_between(
            drawdown.index, drawdown.values, 0, color=self.COLORS["drawdown"], alpha=0.4
        )
        ax.plot(
            drawdown.index,
            drawdown.values,
            color=self.COLORS["drawdown"],
            linewidth=0.5,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.set_title("Underwater Chart")
        ax.set_ylim([drawdown.min() * 1.1, 0])

        # Mark max drawdown
        max_dd_date = drawdown.idxmin()
        ax.axvline(max_dd_date, color="black", linestyle="--", linewidth=0.5, alpha=0.5)

        return ax

    def plot_rolling_sharpe(self, window: int = 63, ax: Optional[Any] = None) -> Any:
        """Plot rolling Sharpe ratio."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        rolling_sharpe = self.performance_analyzer.rolling_sharpe(window)

        ax.plot(
            rolling_sharpe.index,
            rolling_sharpe.values,
            color=self.COLORS["strategy"],
            linewidth=1,
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axhline(
            rolling_sharpe.mean(),
            color=self.COLORS["neutral"],
            linestyle="--",
            linewidth=1,
            label=f"Mean: {rolling_sharpe.mean():.2f}",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(f"Rolling Sharpe Ratio ({window}-day)")
        ax.legend()

        return ax

    def plot_rolling_volatility(
        self, window: int = 21, ax: Optional[Any] = None
    ) -> Any:
        """Plot rolling volatility."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        rolling_vol = self.performance_analyzer.rolling_volatility(window)

        ax.plot(
            rolling_vol.index,
            rolling_vol.values,
            color=self.COLORS["strategy"],
            linewidth=1,
        )
        ax.axhline(
            rolling_vol.mean(),
            color=self.COLORS["neutral"],
            linestyle="--",
            linewidth=1,
            label=f"Mean: {rolling_vol.mean():.1%}",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized Volatility")
        ax.set_title(f"Rolling Volatility ({window}-day)")
        ax.legend()

        return ax

    def plot_monthly_returns_heatmap(self, ax: Optional[Any] = None) -> Any:
        """Plot monthly returns heatmap."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        monthly_table = self.performance_analyzer.monthly_returns_table()

        if len(monthly_table) == 0:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, max(4, len(monthly_table) * 0.4)))

        # Create heatmap
        if HAS_SEABORN:
            sns.heatmap(
                monthly_table,
                annot=True,
                fmt=".1%",
                center=0,
                cmap="RdYlGn",
                ax=ax,
                vmin=-0.15,
                vmax=0.15,
                annot_kws={"size": 8},
            )
        else:
            im = ax.imshow(
                monthly_table.values,
                cmap="RdYlGn",
                aspect="auto",
                vmin=-0.15,
                vmax=0.15,
            )
            plt.colorbar(im, ax=ax)

        ax.set_title("Monthly Returns Heatmap")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")

        # Set month labels
        ax.set_xticks(range(12))
        ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )

        return ax

    def plot_returns_distribution(self, ax: Optional[Any] = None) -> Any:
        """Plot returns distribution."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if HAS_SEABORN:
            sns.histplot(
                self.returns, kde=True, ax=ax, color=self.COLORS["strategy"], alpha=0.6
            )
        else:
            ax.hist(
                self.returns,
                bins=50,
                alpha=0.6,
                color=self.COLORS["strategy"],
                edgecolor="white",
            )

        # Add normal distribution overlay
        mean = self.returns.mean()
        std = self.returns.std()
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

        ax2 = ax.twinx()
        ax2.plot(x, normal_dist, "r--", linewidth=1, label="Normal")
        ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
        ax2.set_visible(False)

        # Add VaR lines
        var_95 = self.returns.quantile(0.05)
        var_99 = self.returns.quantile(0.01)
        ax.axvline(
            var_95,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"VaR 95%: {var_95:.2%}",
        )
        ax.axvline(
            var_99,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"VaR 99%: {var_99:.2%}",
        )

        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.set_title("Returns Distribution")
        ax.legend()

        # Add stats
        skew = self.returns.skew()
        kurt = self.returns.kurtosis()
        text = (
            f"Mean: {mean:.4%}\nStd: {std:.4%}\nSkew: {skew:.2f}\nKurtosis: {kurt:.2f}"
        )
        self._create_text_box(ax, text, (0.75, 0.95))

        return ax

    def plot_yearly_returns(self, ax: Optional[Any] = None) -> Any:
        """Plot yearly returns bar chart."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        yearly = self.performance_analyzer.yearly_returns()

        if len(yearly) == 0:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        colors = [
            self.COLORS["positive"] if r > 0 else self.COLORS["negative"]
            for r in yearly.values
        ]

        bars = ax.bar(range(len(yearly)), yearly.values, color=colors, alpha=0.8)

        ax.set_xticks(range(len(yearly)))
        ax.set_xticklabels([d.year for d in yearly.index], rotation=45)
        ax.set_xlabel("Year")
        ax.set_ylabel("Return")
        ax.set_title("Annual Returns")
        ax.axhline(0, color="black", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, yearly.values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height > 0 else -10),
                textcoords="offset points",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=8,
            )

        return ax

    # =========================================================================
    # FULL TEAR SHEETS
    # =========================================================================

    def create_returns_tear_sheet(
        self, filename: Optional[str] = None, show: bool = True
    ) -> Optional[Any]:
        """
        Create returns-focused tear sheet.

        Args:
            filename: Optional file path to save figure
            show: Whether to display the figure

        Returns:
            Figure object if matplotlib available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for tear sheets")

        self._setup_plot_style()

        fig = plt.figure(figsize=(14, 16))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.2)

        # Cumulative returns (full width)
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_cumulative_returns(ax1)

        # Drawdown (full width)
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_drawdown(ax2)

        # Rolling Sharpe
        ax3 = fig.add_subplot(gs[2, 0])
        self.plot_rolling_sharpe(ax=ax3)

        # Rolling volatility
        ax4 = fig.add_subplot(gs[2, 1])
        self.plot_rolling_volatility(ax=ax4)

        # Returns distribution
        ax5 = fig.add_subplot(gs[3, 0])
        self.plot_returns_distribution(ax5)

        # Yearly returns
        ax6 = fig.add_subplot(gs[3, 1])
        self.plot_yearly_returns(ax6)

        plt.suptitle("Returns Tear Sheet", fontsize=14, fontweight="bold", y=1.02)

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def create_full_tear_sheet(
        self, filename: Optional[str] = None, show: bool = True
    ) -> Optional[Any]:
        """
        Create comprehensive tear sheet.

        Args:
            filename: Optional file path to save figure
            show: Whether to display the figure

        Returns:
            Figure object if matplotlib available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for tear sheets")

        self._setup_plot_style()

        fig = plt.figure(figsize=(16, 20))
        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.2)

        # Cumulative returns (full width)
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_cumulative_returns(ax1)

        # Drawdown (full width)
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_drawdown(ax2)

        # Monthly heatmap
        ax3 = fig.add_subplot(gs[2, :])
        self.plot_monthly_returns_heatmap(ax3)

        # Rolling Sharpe
        ax4 = fig.add_subplot(gs[3, 0])
        self.plot_rolling_sharpe(ax=ax4)

        # Rolling volatility
        ax5 = fig.add_subplot(gs[3, 1])
        self.plot_rolling_volatility(ax=ax5)

        # Returns distribution
        ax6 = fig.add_subplot(gs[4, 0])
        self.plot_returns_distribution(ax6)

        # Yearly returns
        ax7 = fig.add_subplot(gs[4, 1])
        self.plot_yearly_returns(ax7)

        plt.suptitle(
            "Full Performance Tear Sheet", fontsize=14, fontweight="bold", y=1.01
        )

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary metrics as DataFrame."""
        metrics = self.performance_analyzer.summary()
        risk_metrics = self.risk_analyzer.summary()

        all_metrics = {**metrics, **risk_metrics}

        return pd.DataFrame({"Value": all_metrics.values()}, index=all_metrics.keys())

    def print_summary(self):
        """Print performance summary to console."""
        metrics = self.performance_analyzer.summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"\n{'RETURNS':^30}")
        print("-" * 30)
        print(f"Total Return:          {metrics['total_return']:>12.2%}")
        print(f"Annualized Return:     {metrics['annualized_return']:>12.2%}")
        print(f"Annualized Volatility: {metrics['annualized_volatility']:>12.2%}")

        print(f"\n{'RISK-ADJUSTED':^30}")
        print("-" * 30)
        print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>12.2f}")
        print(f"Sortino Ratio:         {metrics['sortino_ratio']:>12.2f}")
        print(f"Calmar Ratio:          {metrics['calmar_ratio']:>12.2f}")

        print(f"\n{'DRAWDOWN':^30}")
        print("-" * 30)
        print(f"Max Drawdown:          {metrics['max_drawdown']:>12.2%}")

        print(f"\n{'RISK':^30}")
        print("-" * 30)
        print(f"VaR (95%):             {metrics['var_95']:>12.2%}")
        print(f"CVaR (95%):            {metrics['cvar_95']:>12.2%}")

        print(f"\n{'TRADE STATS':^30}")
        print("-" * 30)
        print(f"Win Rate:              {metrics['win_rate']:>12.2%}")
        print(f"Profit Factor:         {metrics['profit_factor']:>12.2f}")
        print(f"Best Day:              {metrics['best_day']:>12.2%}")
        print(f"Worst Day:             {metrics['worst_day']:>12.2%}")

        if metrics.get("alpha") is not None:
            print(f"\n{'BENCHMARK RELATIVE':^30}")
            print("-" * 30)
            print(f"Alpha:                 {metrics['alpha']:>12.2%}")
            print(f"Beta:                  {metrics['beta']:>12.2f}")
            print(f"Information Ratio:     {metrics['information_ratio']:>12.2f}")

        print("\n" + "=" * 60)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_full_tear_sheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    positions: Optional[pd.DataFrame] = None,
    **kwargs,
) -> Any:
    """
    Create a full tear sheet.

    Args:
        returns: Daily returns series
        benchmark: Optional benchmark returns
        positions: Optional positions DataFrame
        **kwargs: Additional arguments for TearSheet

    Returns:
        matplotlib figure
    """
    ts = TearSheet(returns, benchmark=benchmark, positions=positions, **kwargs)
    return ts.create_full_tear_sheet()


def create_returns_tear_sheet(
    returns: pd.Series, benchmark: Optional[pd.Series] = None, **kwargs
) -> Any:
    """
    Create a returns-focused tear sheet.

    Args:
        returns: Daily returns series
        benchmark: Optional benchmark returns
        **kwargs: Additional arguments for TearSheet

    Returns:
        matplotlib figure
    """
    ts = TearSheet(returns, benchmark=benchmark, **kwargs)
    return ts.create_returns_tear_sheet()


def create_position_tear_sheet(
    returns: pd.Series, positions: pd.DataFrame, **kwargs
) -> Any:
    """
    Create a position-focused tear sheet.

    Args:
        returns: Daily returns series
        positions: Positions DataFrame (date x asset)
        **kwargs: Additional arguments

    Returns:
        matplotlib figure
    """
    ts = TearSheet(returns, positions=positions, **kwargs)
    # For now, just use full tear sheet
    # TODO: Add position-specific plots
    return ts.create_full_tear_sheet()


def create_round_trip_tear_sheet(
    returns: pd.Series, transactions: pd.DataFrame, **kwargs
) -> Any:
    """
    Create a round-trip analysis tear sheet.

    Args:
        returns: Daily returns series
        transactions: Transactions DataFrame
        **kwargs: Additional arguments

    Returns:
        matplotlib figure
    """
    ts = TearSheet(returns, transactions=transactions, **kwargs)
    # TODO: Add round-trip specific analysis
    return ts.create_full_tear_sheet()


def create_bayesian_tear_sheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    n_samples: int = 2000,
    **kwargs,
) -> Any:
    """
    Create a Bayesian tear sheet with uncertainty estimates.

    Args:
        returns: Daily returns series
        benchmark: Optional benchmark returns
        n_samples: Number of posterior samples
        **kwargs: Additional arguments

    Returns:
        matplotlib figure
    """
    # TODO: Implement Bayesian analysis
    ts = TearSheet(returns, benchmark=benchmark, **kwargs)
    return ts.create_full_tear_sheet()
