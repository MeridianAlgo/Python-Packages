"""
Performance Analytics Module

Comprehensive performance analysis including returns calculation, risk-adjusted metrics,
rolling statistics, and benchmark comparison.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    omega_ratio: float
    information_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    treynor_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies and portfolios.

    Provides institutional-grade performance metrics, rolling analysis,
    benchmark comparison, and statistical testing.

    Example:
        >>> analyzer = PerformanceAnalyzer(returns, benchmark=spy_returns)
        >>> metrics = analyzer.calculate_all_metrics()
        >>> analyzer.plot_cumulative_returns()
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """
        Initialize PerformanceAnalyzer.

        Args:
            returns: Daily returns series or DataFrame of returns
            benchmark: Optional benchmark returns for comparison
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Number of periods per year (252 for daily)
        """
        self.returns = self._validate_returns(returns)
        self.benchmark = (
            self._validate_returns(benchmark) if benchmark is not None else None
        )
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._daily_rf = risk_free_rate / periods_per_year

    def _validate_returns(self, returns: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Validate and clean returns data."""
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] == 1:
                returns = returns.iloc[:, 0]
            else:
                raise ValueError("For multi-asset returns, use PortfolioAnalyzer")

        returns = returns.dropna()

        # Check for extreme values
        if (returns.abs() > 1).any():
            import warnings

            warnings.warn(
                "Returns contain values > 100%, verify data is in decimal format"
            )

        return returns

    # =========================================================================
    # BASIC RETURNS METRICS
    # =========================================================================

    def total_return(self) -> float:
        """Calculate total cumulative return."""
        return (1 + self.returns).prod() - 1

    def annualized_return(self) -> float:
        """Calculate annualized return (CAGR)."""
        total = self.total_return()
        n_years = len(self.returns) / self.periods_per_year
        if n_years <= 0:
            return 0.0
        return (1 + total) ** (1 / n_years) - 1

    def annualized_volatility(self) -> float:
        """Calculate annualized volatility."""
        return self.returns.std() * np.sqrt(self.periods_per_year)

    def downside_volatility(self, threshold: float = 0.0) -> float:
        """
        Calculate downside volatility (semi-deviation).

        Args:
            threshold: Minimum acceptable return (default 0)
        """
        downside_returns = self.returns[self.returns < threshold]
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std() * np.sqrt(self.periods_per_year)

    # =========================================================================
    # RISK-ADJUSTED RETURNS
    # =========================================================================

    def sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_return = self.annualized_return() - self.risk_free_rate
        vol = self.annualized_volatility()
        return excess_return / vol if vol > 0 else 0.0

    def sortino_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Sortino ratio.

        Args:
            threshold: Minimum acceptable return (default 0)
        """
        excess_return = self.annualized_return() - self.risk_free_rate
        downside_vol = self.downside_volatility(threshold)
        return excess_return / downside_vol if downside_vol > 0 else 0.0

    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        ann_return = self.annualized_return()
        max_dd = abs(self.max_drawdown())
        return ann_return / max_dd if max_dd > 0 else 0.0

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Args:
            threshold: Return threshold (default 0)
        """
        excess = self.returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        return gains / losses if losses > 0 else np.inf

    def information_ratio(self) -> Optional[float]:
        """Calculate Information ratio vs benchmark."""
        if self.benchmark is None:
            return None

        active_return = self.returns - self.benchmark
        tracking_error = active_return.std() * np.sqrt(self.periods_per_year)

        if tracking_error == 0:
            return 0.0

        ann_active = active_return.mean() * self.periods_per_year
        return ann_active / tracking_error

    def treynor_ratio(self) -> Optional[float]:
        """Calculate Treynor ratio."""
        beta = self.beta()
        if beta is None or beta == 0:
            return None

        excess_return = self.annualized_return() - self.risk_free_rate
        return excess_return / beta

    # =========================================================================
    # BENCHMARK-RELATIVE METRICS
    # =========================================================================

    def alpha(self) -> Optional[float]:
        """Calculate Jensen's alpha."""
        if self.benchmark is None:
            return None

        # Align returns
        aligned = pd.concat([self.returns, self.benchmark], axis=1).dropna()
        if len(aligned) < 2:
            return None

        strategy_returns = aligned.iloc[:, 0].values
        benchmark_returns = aligned.iloc[:, 1].values

        # Simple regression
        cov = np.cov(strategy_returns, benchmark_returns)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

        alpha_daily = (
            strategy_returns.mean()
            - self._daily_rf
            - beta * (benchmark_returns.mean() - self._daily_rf)
        )

        return alpha_daily * self.periods_per_year

    def beta(self) -> Optional[float]:
        """Calculate beta vs benchmark."""
        if self.benchmark is None:
            return None

        aligned = pd.concat([self.returns, self.benchmark], axis=1).dropna()
        if len(aligned) < 2:
            return None

        cov = np.cov(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
        return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

    def tracking_error(self) -> Optional[float]:
        """Calculate annualized tracking error."""
        if self.benchmark is None:
            return None

        active_return = self.returns - self.benchmark
        return active_return.std() * np.sqrt(self.periods_per_year)

    def correlation(self) -> Optional[float]:
        """Calculate correlation with benchmark."""
        if self.benchmark is None:
            return None

        aligned = pd.concat([self.returns, self.benchmark], axis=1).dropna()
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

    def up_capture(self) -> Optional[float]:
        """Calculate up-market capture ratio."""
        if self.benchmark is None:
            return None

        aligned = pd.concat([self.returns, self.benchmark], axis=1).dropna()
        up_market = aligned[aligned.iloc[:, 1] > 0]

        if len(up_market) == 0:
            return None

        strategy_up = (1 + up_market.iloc[:, 0]).prod() - 1
        benchmark_up = (1 + up_market.iloc[:, 1]).prod() - 1

        return strategy_up / benchmark_up if benchmark_up != 0 else 0

    def down_capture(self) -> Optional[float]:
        """Calculate down-market capture ratio."""
        if self.benchmark is None:
            return None

        aligned = pd.concat([self.returns, self.benchmark], axis=1).dropna()
        down_market = aligned[aligned.iloc[:, 1] < 0]

        if len(down_market) == 0:
            return None

        strategy_down = (1 + down_market.iloc[:, 0]).prod() - 1
        benchmark_down = (1 + down_market.iloc[:, 1]).prod() - 1

        return strategy_down / benchmark_down if benchmark_down != 0 else 0

    # =========================================================================
    # DRAWDOWN ANALYSIS
    # =========================================================================

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def drawdown_series(self) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max

    def top_drawdowns(self, n: int = 5) -> pd.DataFrame:
        """
        Get the top N drawdowns with their characteristics.

        Args:
            n: Number of top drawdowns to return

        Returns:
            DataFrame with drawdown start, end, trough, depth, duration, recovery
        """
        drawdown = self.drawdown_series()

        # Find drawdown periods
        is_drawdown = drawdown < 0

        # Find start and end of each drawdown
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1).fillna(False)

        starts = drawdown_starts[drawdown_starts].index.tolist()
        ends = drawdown_ends[drawdown_ends].index.tolist()

        # If still in drawdown, use last date
        if len(starts) > len(ends):
            ends.append(drawdown.index[-1])

        drawdowns = []
        for start, end in zip(starts, ends):
            period_dd = drawdown[start:end]
            if len(period_dd) == 0:
                continue

            trough_date = period_dd.idxmin()
            depth = period_dd.min()

            drawdowns.append(
                {
                    "start": start,
                    "trough": trough_date,
                    "end": end,
                    "depth": depth,
                    "duration_days": (
                        (end - start).days
                        if hasattr(end - start, "days")
                        else len(period_dd)
                    ),
                    "recovery_days": (
                        (end - trough_date).days
                        if hasattr(end - trough_date, "days")
                        else 0
                    ),
                }
            )

        df = pd.DataFrame(drawdowns)
        if len(df) > 0:
            df = df.sort_values("depth").head(n)

        return df

    def underwater_chart_data(self) -> pd.Series:
        """Get data for underwater chart."""
        return self.drawdown_series()

    # =========================================================================
    # ROLLING ANALYSIS
    # =========================================================================

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_return = self.returns.rolling(window).mean() * self.periods_per_year
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(
            self.periods_per_year
        )
        return (rolling_return - self.risk_free_rate) / rolling_vol

    def rolling_sortino(self, window: int = 63) -> pd.Series:
        """Calculate rolling Sortino ratio."""

        def _sortino(returns):
            if len(returns) == 0:
                return 0
            ann_return = returns.mean() * self.periods_per_year
            downside = returns[returns < 0]
            downside_vol = (
                downside.std() * np.sqrt(self.periods_per_year)
                if len(downside) > 0
                else 0
            )
            return ann_return / downside_vol if downside_vol > 0 else 0

        return self.returns.rolling(window).apply(_sortino)

    def rolling_volatility(self, window: int = 21) -> pd.Series:
        """Calculate rolling volatility."""
        return self.returns.rolling(window).std() * np.sqrt(self.periods_per_year)

    def rolling_beta(self, window: int = 63) -> Optional[pd.Series]:
        """Calculate rolling beta vs benchmark."""
        if self.benchmark is None:
            return None

        def _beta(aligned):
            if len(aligned) < 2:
                return np.nan
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

        aligned = pd.concat([self.returns, self.benchmark], axis=1).dropna()
        return aligned.rolling(window).apply(lambda x: _beta(x.values.reshape(-1, 2)))

    def rolling_returns(self, window: int = 21) -> pd.Series:
        """Calculate rolling returns."""
        return self.returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================

    def skewness(self) -> float:
        """Calculate return skewness."""
        return self.returns.skew()

    def kurtosis(self) -> float:
        """Calculate return kurtosis (excess)."""
        return self.returns.kurtosis()

    def var(self, confidence: float = 0.95, method: str = "historical") -> float:
        """
        Calculate Value at Risk.

        Args:
            confidence: Confidence level (e.g., 0.95)
            method: 'historical', 'parametric', or 'cornish_fisher'
        """
        if method == "historical":
            return self.returns.quantile(1 - confidence)
        elif method == "parametric":
            from scipy.stats import norm

            return self.returns.mean() + norm.ppf(1 - confidence) * self.returns.std()
        elif method == "cornish_fisher":
            from scipy.stats import norm

            z = norm.ppf(1 - confidence)
            s = self.skewness()
            k = self.kurtosis()
            z_cf = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * (k - 3) / 24
                - (2 * z**3 - 5 * z) * s**2 / 36
            )
            return self.returns.mean() + z_cf * self.returns.std()
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.var(confidence)
        return self.returns[self.returns <= var].mean()

    def tail_ratio(self, percentile: float = 0.05) -> float:
        """Calculate tail ratio (right tail / left tail)."""
        right_tail = self.returns.quantile(1 - percentile)
        left_tail = abs(self.returns.quantile(percentile))
        return right_tail / left_tail if left_tail != 0 else np.inf

    def gain_to_pain_ratio(self) -> float:
        """Calculate Gain-to-Pain ratio."""
        total_gain = self.returns[self.returns > 0].sum()
        total_pain = abs(self.returns[self.returns < 0].sum())
        return total_gain / total_pain if total_pain > 0 else np.inf

    # =========================================================================
    # TRADE STATISTICS
    # =========================================================================

    def win_rate(self) -> float:
        """Calculate win rate (% of positive days)."""
        return (self.returns > 0).mean()

    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = self.returns[self.returns > 0].sum()
        gross_loss = abs(self.returns[self.returns < 0].sum())
        return gross_profit / gross_loss if gross_loss > 0 else np.inf

    def payoff_ratio(self) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        avg_win = self.returns[self.returns > 0].mean()
        avg_loss = abs(self.returns[self.returns < 0].mean())
        return avg_win / avg_loss if avg_loss > 0 else np.inf

    def expectancy(self) -> float:
        """Calculate expectancy per trade."""
        win_rate = self.win_rate()
        payoff = self.payoff_ratio()
        return (win_rate * payoff) - (1 - win_rate)

    def best_day(self) -> float:
        """Get best single day return."""
        return self.returns.max()

    def worst_day(self) -> float:
        """Get worst single day return."""
        return self.returns.min()

    def avg_return(self) -> float:
        """Get average daily return."""
        return self.returns.mean()

    def positive_days(self) -> int:
        """Count positive return days."""
        return (self.returns > 0).sum()

    def negative_days(self) -> int:
        """Count negative return days."""
        return (self.returns < 0).sum()

    # =========================================================================
    # MONTHLY/YEARLY ANALYSIS
    # =========================================================================

    def monthly_returns(self) -> pd.Series:
        """Calculate monthly returns."""
        return self.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

    def yearly_returns(self) -> pd.Series:
        """Calculate yearly returns."""
        return self.returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)

    def monthly_returns_table(self) -> pd.DataFrame:
        """Create monthly returns table (years x months)."""
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return pd.DataFrame()

        monthly = self.monthly_returns()

        # Create table
        years = monthly.index.year
        months = monthly.index.month

        table = pd.DataFrame(index=sorted(years.unique()), columns=range(1, 13))

        for year in years.unique():
            for month in range(1, 13):
                mask = (years == year) & (months == month)
                if mask.any():
                    table.loc[year, month] = monthly[mask].values[0]

        return table.astype(float)

    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================

    def calculate_all_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        return PerformanceMetrics(
            total_return=self.total_return(),
            annualized_return=self.annualized_return(),
            annualized_volatility=self.annualized_volatility(),
            sharpe_ratio=self.sharpe_ratio(),
            sortino_ratio=self.sortino_ratio(),
            calmar_ratio=self.calmar_ratio(),
            max_drawdown=self.max_drawdown(),
            omega_ratio=self.omega_ratio(),
            information_ratio=self.information_ratio(),
            alpha=self.alpha(),
            beta=self.beta(),
            treynor_ratio=self.treynor_ratio(),
            tracking_error=self.tracking_error(),
        )

    def summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all metrics."""
        return {
            # Returns
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "annualized_volatility": self.annualized_volatility(),
            # Risk-adjusted
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "omega_ratio": self.omega_ratio(),
            # Drawdown
            "max_drawdown": self.max_drawdown(),
            # Benchmark-relative
            "alpha": self.alpha(),
            "beta": self.beta(),
            "information_ratio": self.information_ratio(),
            "tracking_error": self.tracking_error(),
            "up_capture": self.up_capture(),
            "down_capture": self.down_capture(),
            # Statistical
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
            "var_95": self.var(0.95),
            "cvar_95": self.cvar(0.95),
            "tail_ratio": self.tail_ratio(),
            # Trade statistics
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "payoff_ratio": self.payoff_ratio(),
            "expectancy": self.expectancy(),
            "best_day": self.best_day(),
            "worst_day": self.worst_day(),
            # Counts
            "total_days": len(self.returns),
            "positive_days": self.positive_days(),
            "negative_days": self.negative_days(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert summary to DataFrame."""
        summary = self.summary()
        return pd.DataFrame({"Value": summary.values()}, index=summary.keys())


def calculate_returns_metrics(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Convenience function to calculate all returns metrics.

    Args:
        returns: Return series
        benchmark: Optional benchmark returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of all performance metrics
    """
    analyzer = PerformanceAnalyzer(returns, benchmark, risk_free_rate)
    return analyzer.summary()
