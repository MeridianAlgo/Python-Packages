"""
Spread Analysis Module

Comprehensive bid-ask spread analysis including realized spread, effective spread,
price improvement, and spread decomposition.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class SpreadAnalyzer:
    """
    Analyze bid-ask spreads and execution quality.

    Provides:
    - Quoted vs effective vs realized spread
    - Spread decomposition
    - Price improvement analysis
    - Spread volatility and patterns
    """

    def __init__(self, quotes: pd.DataFrame, trades: Optional[pd.DataFrame] = None):
        """
        Initialize SpreadAnalyzer.

        Args:
            quotes: Quote data with 'bid', 'ask' columns
            trades: Optional trade data for effective spread
        """
        self.quotes = quotes.copy()
        self.trades = trades.copy() if trades is not None else None

        # Pre-calculate common metrics
        self.quotes["mid"] = (self.quotes["bid"] + self.quotes["ask"]) / 2
        self.quotes["quoted_spread"] = self.quotes["ask"] - self.quotes["bid"]
        self.quotes["spread_bps"] = (
            self.quotes["quoted_spread"] / self.quotes["mid"]
        ) * 10000

    def quoted_spread(self) -> pd.Series:
        """Get quoted spread series."""
        return self.quotes["quoted_spread"]

    def quoted_spread_bps(self) -> pd.Series:
        """Get quoted spread in basis points."""
        return self.quotes["spread_bps"]

    def average_spread(self) -> float:
        """Get time-weighted average spread."""
        if "duration" in self.quotes.columns:
            return (
                self.quotes["quoted_spread"] * self.quotes["duration"]
            ).sum() / self.quotes["duration"].sum()
        return self.quotes["quoted_spread"].mean()

    def average_spread_bps(self) -> float:
        """Get average spread in basis points."""
        return self.quotes["spread_bps"].mean()

    def spread_volatility(self) -> float:
        """Get spread volatility."""
        return self.quotes["quoted_spread"].std()

    def spread_percentiles(self) -> Dict[str, float]:
        """Get spread percentiles."""
        return {
            "p10": self.quotes["spread_bps"].quantile(0.10),
            "p25": self.quotes["spread_bps"].quantile(0.25),
            "p50": self.quotes["spread_bps"].quantile(0.50),
            "p75": self.quotes["spread_bps"].quantile(0.75),
            "p90": self.quotes["spread_bps"].quantile(0.90),
        }

    def intraday_pattern(self) -> pd.Series:
        """Analyze intraday spread pattern."""
        if "timestamp" not in self.quotes.columns:
            return pd.Series()

        self.quotes["hour"] = pd.to_datetime(self.quotes["timestamp"]).dt.hour
        return self.quotes.groupby("hour")["spread_bps"].mean()

    def summary(self) -> Dict[str, Any]:
        """Generate spread analysis summary."""
        return {
            "avg_quoted_spread": self.average_spread(),
            "avg_spread_bps": self.average_spread_bps(),
            "spread_volatility": self.spread_volatility(),
            "min_spread": self.quotes["quoted_spread"].min(),
            "max_spread": self.quotes["quoted_spread"].max(),
            **self.spread_percentiles(),
        }


class RealizedSpread:
    """
    Calculate and analyze realized spreads.

    Realized spread measures market maker profitability
    after information is revealed.
    """

    def __init__(
        self, trades: pd.DataFrame, quotes: pd.DataFrame, lag_periods: int = 5
    ):
        """
        Initialize RealizedSpread.

        Args:
            trades: Trade data
            quotes: Quote data
            lag_periods: Number of periods to measure future mid
        """
        self.trades = trades.copy()
        self.quotes = quotes.copy()
        self.lag = lag_periods

        self._calculate()

    def _calculate(self):
        """Calculate realized spreads."""
        # Merge trades with quotes
        if "timestamp" in self.trades.columns and "timestamp" in self.quotes.columns:
            merged = pd.merge_asof(
                self.trades.sort_values("timestamp"),
                self.quotes.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
        else:
            # Assume aligned indices
            merged = pd.concat([self.trades, self.quotes], axis=1)

        # Calculate midpoint
        merged["mid"] = (merged["bid"] + merged["ask"]) / 2

        # Calculate future midpoint
        merged["future_mid"] = merged["mid"].shift(-self.lag)

        # Determine trade direction
        if "side" in merged.columns:
            merged["direction"] = merged["side"].map({"buy": 1, "sell": -1}).fillna(0)
        else:
            merged["direction"] = np.sign(merged["price"] - merged["mid"])

        # Realized spread = 2 * direction * (price - future_mid)
        merged["realized_spread"] = (
            2 * merged["direction"] * (merged["price"] - merged["future_mid"])
        )

        self.data = merged

    def get_realized_spreads(self) -> pd.Series:
        """Get realized spread series."""
        return self.data["realized_spread"].dropna()

    def average(self) -> float:
        """Get average realized spread."""
        return self.get_realized_spreads().mean()

    def by_side(self) -> Dict[str, float]:
        """Get average realized spread by trade side."""
        if "side" not in self.data.columns:
            return {}

        return self.data.groupby("side")["realized_spread"].mean().to_dict()

    def by_size_bucket(self, n_buckets: int = 5) -> pd.Series:
        """Get average realized spread by trade size."""
        self.data["size_bucket"] = pd.qcut(
            self.data["size"], n_buckets, labels=False, duplicates="drop"
        )
        return self.data.groupby("size_bucket")["realized_spread"].mean()


class EffectiveSpread:
    """
    Calculate and analyze effective spreads.

    Effective spread measures the actual cost of trading
    compared to the quoted midpoint.
    """

    def __init__(self, trades: pd.DataFrame, quotes: pd.DataFrame):
        """
        Initialize EffectiveSpread.

        Args:
            trades: Trade data
            quotes: Quote data
        """
        self.trades = trades.copy()
        self.quotes = quotes.copy()

        self._calculate()

    def _calculate(self):
        """Calculate effective spreads."""
        # Merge trades with quotes
        if "timestamp" in self.trades.columns and "timestamp" in self.quotes.columns:
            merged = pd.merge_asof(
                self.trades.sort_values("timestamp"),
                self.quotes.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
        else:
            merged = pd.concat([self.trades, self.quotes], axis=1)

        # Calculate midpoint
        merged["mid"] = (merged["bid"] + merged["ask"]) / 2
        merged["quoted_spread"] = merged["ask"] - merged["bid"]

        # Effective spread = 2 * |price - mid|
        merged["effective_spread"] = 2 * abs(merged["price"] - merged["mid"])

        # Relative effective spread (bps)
        merged["effective_spread_bps"] = (
            merged["effective_spread"] / merged["mid"]
        ) * 10000

        # Price improvement
        merged["price_improvement"] = merged["quoted_spread"] / 2 - abs(
            merged["price"] - merged["mid"]
        )

        self.data = merged

    def get_effective_spreads(self) -> pd.Series:
        """Get effective spread series."""
        return self.data["effective_spread"]

    def get_effective_spreads_bps(self) -> pd.Series:
        """Get effective spread series in basis points."""
        return self.data["effective_spread_bps"]

    def average(self) -> float:
        """Get volume-weighted average effective spread."""
        if "size" in self.data.columns:
            total_value = (self.data["effective_spread"] * self.data["size"]).sum()
            total_size = self.data["size"].sum()
            return total_value / total_size if total_size > 0 else 0
        return self.data["effective_spread"].mean()

    def average_bps(self) -> float:
        """Get average effective spread in basis points."""
        return self.data["effective_spread_bps"].mean()

    def price_improvement(self) -> float:
        """Get average price improvement (positive = trader benefited)."""
        if "price_improvement" in self.data.columns:
            return self.data["price_improvement"].mean()
        return 0

    def price_improvement_rate(self) -> float:
        """Get fraction of trades with price improvement."""
        if "price_improvement" in self.data.columns:
            return (self.data["price_improvement"] > 0).mean()
        return 0

    def effective_over_quoted(self) -> float:
        """Get ratio of effective to quoted spread."""
        effective_avg = self.data["effective_spread"].mean()
        quoted_avg = self.data["quoted_spread"].mean()
        return effective_avg / quoted_avg if quoted_avg > 0 else 1

    def by_size(self, n_buckets: int = 5) -> pd.DataFrame:
        """Analyze effective spread by trade size."""
        if "size" not in self.data.columns:
            return pd.DataFrame()

        self.data["size_bucket"] = pd.qcut(
            self.data["size"],
            n_buckets,
            labels=range(1, n_buckets + 1),
            duplicates="drop",
        )

        return self.data.groupby("size_bucket").agg(
            {
                "effective_spread_bps": "mean",
                "price_improvement": "mean",
                "size": ["mean", "count"],
            }
        )

    def by_time(self, freq: str = "H") -> pd.Series:
        """Analyze effective spread by time."""
        if "timestamp" not in self.data.columns:
            return pd.Series()

        self.data["time_bucket"] = pd.to_datetime(self.data["timestamp"]).dt.floor(freq)
        return self.data.groupby("time_bucket")["effective_spread_bps"].mean()

    def summary(self) -> Dict[str, Any]:
        """Generate effective spread summary."""
        return {
            "avg_effective_spread": self.average(),
            "avg_effective_spread_bps": self.average_bps(),
            "avg_price_improvement": self.price_improvement(),
            "price_improvement_rate": self.price_improvement_rate(),
            "effective_over_quoted": self.effective_over_quoted(),
            "min_spread_bps": self.data["effective_spread_bps"].min(),
            "max_spread_bps": self.data["effective_spread_bps"].max(),
            "spread_std": self.data["effective_spread_bps"].std(),
        }
