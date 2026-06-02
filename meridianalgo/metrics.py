"""
Top-level performance metrics helpers.

One-call utilities that summarize a returns series without having to construct
an analyzer first. Built on :class:`PerformanceAnalyzer`, pure numpy/pandas.

Example:
    >>> import meridianalgo as m
    >>> stats = m.summary_stats(returns)
    >>> print(m.tearsheet(returns))
"""

from typing import Optional, Union

import pandas as pd

from .analytics.performance import PerformanceAnalyzer

__all__ = ["summary_stats", "tearsheet"]


def summary_stats(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    as_series: bool = False,
) -> Union[dict, pd.Series]:
    """Compute a full set of performance and risk statistics in one call.

    Args:
        returns: Periodic returns (decimal, e.g. 0.01 for 1%).
        benchmark: Optional benchmark returns for relative metrics.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Periods per year (252 for daily data).
        as_series: Return a pandas Series instead of a dict.

    Returns:
        Mapping of metric name to value (dict, or Series if ``as_series``).
    """
    analyzer = PerformanceAnalyzer(
        returns,
        benchmark=benchmark,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    stats = analyzer.summary()
    return pd.Series(stats, name="value") if as_series else stats


def tearsheet(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> str:
    """Return a formatted plain-text performance summary.

    Args:
        returns: Periodic returns (decimal).
        benchmark: Optional benchmark returns for relative metrics.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Periods per year (252 for daily data).

    Returns:
        Aligned, human-readable summary table as a string.
    """
    stats = summary_stats(
        returns,
        benchmark=benchmark,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    width = max(len(k) for k in stats)
    lines = ["Performance Summary", "=" * (width + 16)]
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        if value is None:
            shown = "n/a"
        elif isinstance(value, float):
            shown = f"{value:,.4f}"
        else:
            shown = str(value)
        lines.append(f"{label:<{width}}  {shown:>12}")
    return "\n".join(lines)
