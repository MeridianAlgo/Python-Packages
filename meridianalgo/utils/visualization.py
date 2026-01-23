"""
Visualization utilities for MeridianAlgo.
"""

from typing import Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class PortfolioVisualizer:
    """Visualize portfolio performance and characteristics."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer with default figure size."""
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)

    def plot_portfolio_performance(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Portfolio Performance",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot cumulative portfolio performance."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate cumulative returns
        cumulative = (1 + portfolio_returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label="Portfolio", linewidth=2)

        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values,
                label="Benchmark",
                alpha=0.7,
                linestyle="--",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_drawdown(
        self,
        portfolio_returns: pd.Series,
        title: str = "Portfolio Drawdown",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot portfolio drawdown over time."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        # Fill area for drawdown
        ax.fill_between(
            drawdown.index, drawdown.values, 0, color="red", alpha=0.3, label="Drawdown"
        )
        ax.plot(drawdown.index, drawdown.values, color="red", linewidth=1)

        # Highlight maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.scatter(max_dd_idx, max_dd_value, color="darkred", s=100, zorder=5)
        ax.annotate(
            f"Max DD: {max_dd_value:.2%}",
            xy=(max_dd_idx, max_dd_value),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_risk_return_scatter(
        self,
        returns_data: pd.DataFrame,
        title: str = "Risk-Return Profile",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot risk-return scatter plot for multiple assets."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate annualized returns and volatilities
        annual_returns = returns_data.mean() * 252
        annual_vols = returns_data.std() * np.sqrt(252)

        # Create scatter plot
        scatter = ax.scatter(
            annual_vols,
            annual_returns,
            s=100,
            alpha=0.7,
            c=range(len(annual_returns)),
            cmap="viridis",
        )

        # Add labels for each point
        for i, (vol, ret) in enumerate(zip(annual_vols, annual_returns)):
            ax.annotate(
                returns_data.columns[i],
                (vol, ret),
                xytext=(5, 5),
                textcoords="offset points",
            )

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Asset Index")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Return")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_correlation_heatmap(
        self,
        returns_data: pd.DataFrame,
        title: str = "Asset Correlation Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate correlation matrix
        corr_matrix = returns_data.corr()

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
            fmt=".2f",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


class TechnicalAnalysisVisualizer:
    """Visualize technical analysis indicators."""

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot_price_with_indicators(
        self,
        prices: pd.Series,
        indicators: Dict[str, pd.Series],
        title: str = "Price with Technical Indicators",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot price with multiple technical indicators."""
        n_indicators = len(indicators)
        fig, axes = plt.subplots(
            n_indicators + 1,
            1,
            figsize=(self.figsize[0], self.figsize[1] * (n_indicators + 1) / 2),
        )

        # Plot price
        axes[0].plot(
            prices.index, prices.values, label="Price", linewidth=2, color="blue"
        )
        axes[0].set_title(f"{title} - Price", fontweight="bold")
        axes[0].set_ylabel("Price")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot indicators
        for i, (name, values) in enumerate(indicators.items(), 1):
            axes[i].plot(values.index, values.values, label=name, linewidth=1.5)
            axes[i].set_title(f"{title} - {name}", fontweight="bold")
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

            # Add special formatting for specific indicators
            if name == "RSI":
                axes[i].axhline(y=70, color="r", linestyle="--", alpha=0.5)
                axes[i].axhline(y=30, color="g", linestyle="--", alpha=0.5)
                axes[i].set_ylim(0, 100)

        # Format x-axis for all subplots
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_candlestick(
        self,
        ohlc_data: pd.DataFrame,
        volume: Optional[pd.Series] = None,
        title: str = "Candlestick Chart",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot candlestick chart with optional volume."""
        from mplfinance import plot

        # Prepare data for mplfinance
        ohlc_data.index.name = "Date"

        # Create subplots
        if volume is not None:
            fig, axes = plt.subplots(
                2, 1, figsize=self.figsize, gridspec_kw={"height_ratios": [3, 1]}
            )

            # Plot candlestick
            plot(ohlc_data, type="candle", style="yahoo", ax=axes[0], volume=False)

            # Plot volume
            axes[1].bar(volume.index, volume.values, alpha=0.7)
            axes[1].set_title("Volume")
            axes[1].set_ylabel("Volume")
            axes[1].grid(True, alpha=0.3)

            axes[0].set_title(title, fontsize=14, fontweight="bold")
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            plot(ohlc_data, type="candle", style="yahoo", ax=ax)
            ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


class RiskVisualizer:
    """Visualize risk metrics and distributions."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot returns distribution with statistics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Histogram
        ax1.hist(
            returns,
            bins=50,
            alpha=0.7,
            density=True,
            color="skyblue",
            edgecolor="black",
        )
        ax1.axvline(
            returns.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {returns.mean():.3f}",
        )
        ax1.axvline(
            returns.median(),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {returns.median():.3f}",
        )
        ax1.set_title("Returns Distribution")
        ax1.set_xlabel("Returns")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats

        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal Distribution)")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_rolling_risk_metrics(
        self,
        returns: pd.Series,
        window: int = 21,
        title: str = "Rolling Risk Metrics",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot rolling volatility and drawdown."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        # Rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        ax1.plot(
            rolling_vol.index,
            rolling_vol.values,
            label=f"{window}-day Rolling Volatility",
            color="orange",
        )
        ax1.set_title("Rolling Volatility (Annualized)")
        ax1.set_ylabel("Volatility")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Rolling drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
        ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=1)
        ax2.set_title("Rolling Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.xticks(rotation=45)
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_risk_contribution(
        self,
        risk_contributions: Dict[str, float],
        title: str = "Portfolio Risk Contribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot risk contribution by asset."""
        fig, ax = plt.subplots(figsize=self.figsize)

        assets = list(risk_contributions.keys())
        contributions = list(risk_contributions.values())

        bars = ax.bar(
            assets, contributions, color=sns.color_palette("viridis", len(assets))
        )

        # Add value labels on bars
        for bar, value in zip(bars, contributions):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2%}",
                ha="center",
                va="bottom",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Risk Contribution")
        ax.set_xlabel("Assets")
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def create_dashboard(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    weights: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a comprehensive dashboard."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid specification
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    cumulative = (1 + portfolio_returns).cumprod()
    ax1.plot(cumulative.index, cumulative.values, label="Portfolio", linewidth=2)

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        ax1.plot(
            benchmark_cumulative.index,
            benchmark_cumulative.values,
            label="Benchmark",
            alpha=0.7,
            linestyle="--",
        )

    ax1.set_title("Portfolio Performance", fontweight="bold")
    ax1.set_ylabel("Cumulative Returns")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)

    # 3. Returns distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(portfolio_returns, bins=30, alpha=0.7, density=True)
    ax3.set_title("Returns Distribution")
    ax3.set_xlabel("Returns")
    ax3.grid(True, alpha=0.3)

    # 4. Rolling volatility
    ax4 = fig.add_subplot(gs[1, 2])
    rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
    ax4.plot(rolling_vol.index, rolling_vol.values, color="orange")
    ax4.set_title("21-Day Rolling Volatility")
    ax4.set_ylabel("Volatility")
    ax4.grid(True, alpha=0.3)

    # 5. Portfolio weights (if provided)
    if weights:
        ax5 = fig.add_subplot(gs[2, :])
        assets = list(weights.keys())
        values = list(weights.values())
        bars = ax5.bar(assets, values, color=sns.color_palette("viridis", len(assets)))

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2%}",
                ha="center",
                va="bottom",
            )

        ax5.set_title("Portfolio Weights")
        ax5.set_ylabel("Weight")
        ax5.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Portfolio Dashboard", fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
