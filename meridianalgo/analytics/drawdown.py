"""
Drawdown Analysis Module

Comprehensive drawdown analysis including drawdown duration, underwater periods,
recovery analysis, and drawdown-based risk metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


@dataclass
class DrawdownPeriod:
    """Container for a single drawdown period."""

    start: Any  # Date
    trough: Any  # Date of max drawdown
    end: Any  # Recovery date (or None if ongoing)
    depth: float  # Maximum drawdown during period
    duration: int  # Days from start to end
    recovery: int  # Days from trough to end
    is_recovered: bool


class DrawdownAnalyzer:
    """
    Comprehensive drawdown analysis.

    Provides detailed analysis of portfolio drawdowns including:
    - Maximum drawdown and timing
    - Top N drawdowns with characteristics
    - Time underwater analysis
    - Drawdown duration statistics
    - Recovery analysis

    Example:
        >>> analyzer = DrawdownAnalyzer(returns)
        >>> max_dd = analyzer.max_drawdown()
        >>> top_5 = analyzer.top_drawdowns(5)
    """

    def __init__(
        self, returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252
    ):
        """
        Initialize DrawdownAnalyzer.

        Args:
            returns: Return series (daily returns in decimal format)
            periods_per_year: Number of trading periods per year
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year

        # Pre-calculate cumulative returns and drawdown series
        self._cumulative = (1 + self.returns).cumprod()
        self._running_max = self._cumulative.cummax()
        self._drawdown = (self._cumulative - self._running_max) / self._running_max

    # =========================================================================
    # BASIC DRAWDOWN METRICS
    # =========================================================================

    def max_drawdown(self) -> float:
        """Get maximum drawdown."""
        return self._drawdown.min()

    def max_drawdown_date(self) -> Any:
        """Get date of maximum drawdown."""
        return self._drawdown.idxmin()

    def drawdown_series(self) -> pd.Series:
        """Get full drawdown series."""
        return self._drawdown

    def current_drawdown(self) -> float:
        """Get current drawdown level."""
        return self._drawdown.iloc[-1]

    def average_drawdown(self) -> float:
        """Get average drawdown (only negative values)."""
        negative_dd = self._drawdown[self._drawdown < 0]
        return negative_dd.mean() if len(negative_dd) > 0 else 0

    # =========================================================================
    # DRAWDOWN PERIODS
    # =========================================================================

    def identify_drawdown_periods(self) -> List[DrawdownPeriod]:
        """
        Identify all drawdown periods.

        Returns:
            List of DrawdownPeriod objects
        """
        periods = []

        # Find drawdown starts and ends
        in_drawdown = self._drawdown < 0

        current_start = None
        current_trough = None
        current_depth = 0

        for date, is_dd in in_drawdown.items():
            if is_dd:
                if current_start is None:
                    # Start of new drawdown
                    current_start = date
                    current_trough = date
                    current_depth = self._drawdown[date]
                else:
                    # Update trough if deeper
                    if self._drawdown[date] < current_depth:
                        current_trough = date
                        current_depth = self._drawdown[date]
            else:
                if current_start is not None:
                    # End of drawdown - recovered
                    duration = self._calculate_duration(current_start, date)
                    recovery = self._calculate_duration(current_trough, date)

                    periods.append(
                        DrawdownPeriod(
                            start=current_start,
                            trough=current_trough,
                            end=date,
                            depth=current_depth,
                            duration=duration,
                            recovery=recovery,
                            is_recovered=True,
                        )
                    )

                    current_start = None
                    current_trough = None
                    current_depth = 0

        # Handle ongoing drawdown
        if current_start is not None:
            last_date = self._drawdown.index[-1]
            duration = self._calculate_duration(current_start, last_date)
            recovery = self._calculate_duration(current_trough, last_date)

            periods.append(
                DrawdownPeriod(
                    start=current_start,
                    trough=current_trough,
                    end=last_date,
                    depth=current_depth,
                    duration=duration,
                    recovery=recovery,
                    is_recovered=False,
                )
            )

        return periods

    def _calculate_duration(self, start, end) -> int:
        """Calculate duration in days between two dates."""
        if hasattr(end - start, "days"):
            return (end - start).days
        else:
            # Assume index positions
            return len(self._drawdown[start:end])

    def top_drawdowns(self, n: int = 5) -> pd.DataFrame:
        """
        Get top N largest drawdowns.

        Args:
            n: Number of drawdowns to return

        Returns:
            DataFrame with drawdown details
        """
        periods = self.identify_drawdown_periods()

        if len(periods) == 0:
            return pd.DataFrame()

        # Sort by depth and take top N
        periods.sort(key=lambda x: x.depth)
        top_periods = periods[:n]

        data = []
        for i, p in enumerate(top_periods, 1):
            data.append(
                {
                    "Rank": i,
                    "Start": p.start,
                    "Trough": p.trough,
                    "End": p.end if p.is_recovered else "Ongoing",
                    "Depth": p.depth,
                    "Duration (days)": p.duration,
                    "Recovery (days)": p.recovery if p.is_recovered else None,
                    "Recovered": p.is_recovered,
                }
            )

        return pd.DataFrame(data)

    # =========================================================================
    # DURATION ANALYSIS
    # =========================================================================

    def time_underwater(self) -> float:
        """
        Calculate percentage of time in drawdown.

        Returns:
            Fraction of time spent in drawdown
        """
        return (self._drawdown < 0).mean()

    def average_drawdown_duration(self) -> float:
        """Get average duration of drawdown periods."""
        periods = self.identify_drawdown_periods()
        if len(periods) == 0:
            return 0

        durations = [p.duration for p in periods]
        return np.mean(durations)

    def max_drawdown_duration(self) -> int:
        """Get maximum drawdown duration (consecutive days underwater)."""
        periods = self.identify_drawdown_periods()
        if len(periods) == 0:
            return 0

        return max(p.duration for p in periods)

    def average_recovery_time(self) -> float:
        """Get average time to recover from drawdowns."""
        periods = self.identify_drawdown_periods()
        recovered = [p for p in periods if p.is_recovered]

        if len(recovered) == 0:
            return np.nan

        return np.mean([p.recovery for p in recovered])

    # =========================================================================
    # RISK METRICS
    # =========================================================================

    def ulcer_index(self) -> float:
        """
        Calculate Ulcer Index.

        Square root of the mean of squared drawdowns.
        Measures both depth and duration of drawdowns.
        """
        return np.sqrt(np.mean(self._drawdown**2))

    def pain_index(self) -> float:
        """
        Calculate Pain Index.

        Mean absolute drawdown.
        """
        return abs(self._drawdown.mean())

    def pain_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Pain Ratio.

        Excess return divided by Pain Index.
        """
        annual_return = (1 + self.returns.mean()) ** self.periods_per_year - 1
        excess_return = annual_return - risk_free_rate
        pain = self.pain_index()

        return excess_return / pain if pain > 0 else np.inf

    def ulcer_performance_index(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Ulcer Performance Index.

        Excess return divided by Ulcer Index.
        """
        annual_return = (1 + self.returns.mean()) ** self.periods_per_year - 1
        excess_return = annual_return - risk_free_rate
        ulcer = self.ulcer_index()

        return excess_return / ulcer if ulcer > 0 else np.inf

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio.

        Annualized return divided by max drawdown.
        """
        annual_return = (1 + self.returns.mean()) ** self.periods_per_year - 1
        max_dd = abs(self.max_drawdown())

        return annual_return / max_dd if max_dd > 0 else np.inf

    def sterling_ratio(self, years: int = 3) -> float:
        """
        Calculate Sterling Ratio.

        Return divided by average of largest drawdowns.
        """
        annual_return = (1 + self.returns.mean()) ** self.periods_per_year - 1

        # Get average of annual max drawdowns
        if isinstance(self.returns.index, pd.DatetimeIndex):
            yearly_dd = self._drawdown.resample("Y").min()
            avg_dd = abs(yearly_dd.tail(years).mean())
        else:
            avg_dd = abs(self.max_drawdown())

        return annual_return / avg_dd if avg_dd > 0 else np.inf

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Generate comprehensive drawdown summary."""
        periods = self.identify_drawdown_periods()

        return {
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_date": self.max_drawdown_date(),
            "current_drawdown": self.current_drawdown(),
            "average_drawdown": self.average_drawdown(),
            "time_underwater": self.time_underwater(),
            "num_drawdowns": len(periods),
            "avg_duration": self.average_drawdown_duration(),
            "max_duration": self.max_drawdown_duration(),
            "avg_recovery": self.average_recovery_time(),
            "ulcer_index": self.ulcer_index(),
            "pain_index": self.pain_index(),
            "calmar_ratio": self.calmar_ratio(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert summary to DataFrame."""
        summary = self.summary()
        return pd.DataFrame({"Value": summary.values()}, index=summary.keys())


def calculate_drawdown_series(returns: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Calculate drawdown series from returns.

    Args:
        returns: Return series

    Returns:
        Drawdown series
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max
