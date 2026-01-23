"""
Volume Analysis Module

Volume profile analysis, institutional flow detection, and VPIN calculation.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class VolumeProfile:
    """
    Analyze volume distribution across price and time.

    Creates volume profiles for:
    - Intraday patterns
    - Price level distribution
    - Value areas
    - POC (Point of Control)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "price",
        volume_col: str = "volume",
        time_col: str = "timestamp",
    ):
        """
        Initialize VolumeProfile.

        Args:
            data: DataFrame with price, volume, and time data
            price_col: Name of price column
            volume_col: Name of volume column
            time_col: Name of timestamp column
        """
        self.data = data.copy()
        self.price_col = price_col
        self.volume_col = volume_col
        self.time_col = time_col

        if time_col in self.data.columns:
            self.data[time_col] = pd.to_datetime(self.data[time_col])

    def price_profile(
        self, n_bins: int = 50, price_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Create volume profile by price level.

        Args:
            n_bins: Number of price bins
            price_range: Optional (min, max) price range

        Returns:
            DataFrame with price levels and volume
        """
        prices = self.data[self.price_col]

        if price_range:
            bins = np.linspace(price_range[0], price_range[1], n_bins + 1)
        else:
            bins = np.linspace(prices.min(), prices.max(), n_bins + 1)

        # Assign prices to bins
        bin_labels = list(range(n_bins))
        self.data["price_bin"] = pd.cut(
            prices, bins=bins, labels=bin_labels, include_lowest=True
        )

        # Aggregate volume by price bin
        profile = self.data.groupby("price_bin", observed=True)[self.volume_col].sum()

        # Reindex to ensure all bins are present
        profile = profile.reindex(bin_labels, fill_value=0)

        # Create price level labels (bin centers)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        total_vol = profile.sum()

        return pd.DataFrame(
            {
                "price": bin_centers,
                "volume": profile.values,
                "volume_pct": (
                    (profile.values / total_vol * 100) if total_vol > 0 else 0
                ),
            }
        )

    def intraday_profile(self, freq: str = "30min") -> pd.Series:
        """
        Create intraday volume profile.

        Args:
            freq: Time frequency for aggregation

        Returns:
            Series with volume by time of day
        """
        if self.time_col not in self.data.columns:
            return pd.Series()

        self.data["time_bucket"] = self.data[self.time_col].dt.floor(freq)
        self.data["time_of_day"] = self.data[self.time_col].dt.time

        # Aggregate by time of day (average across days)
        profile = self.data.groupby("time_of_day")[self.volume_col].mean()

        return profile

    def point_of_control(self) -> float:
        """
        Find Point of Control (price with highest volume).

        Returns:
            Price level with highest volume
        """
        profile = self.price_profile()
        poc_idx = profile["volume"].idxmax()
        return profile.loc[poc_idx, "price"]

    def value_area(self, pct: float = 0.70) -> Tuple[float, float]:
        """
        Calculate Value Area (price range containing X% of volume).

        Args:
            pct: Percentage of volume to include (default 70%)

        Returns:
            (low_price, high_price) tuple
        """
        profile = self.price_profile()
        profile = profile.sort_values("price")

        total_volume = profile["volume"].sum()
        target_volume = total_volume * pct

        # Start from POC and expand outward
        profile["volume"].idxmax()
        profile = profile.reset_index(drop=True)
        poc_row = profile["volume"].idxmax()

        included = [poc_row]
        current_volume = profile.loc[poc_row, "volume"]

        low_idx = poc_row - 1
        high_idx = poc_row + 1

        while current_volume < target_volume:
            low_vol = profile.loc[low_idx, "volume"] if low_idx >= 0 else 0
            high_vol = profile.loc[high_idx, "volume"] if high_idx < len(profile) else 0

            if low_vol >= high_vol and low_idx >= 0:
                included.append(low_idx)
                current_volume += low_vol
                low_idx -= 1
            elif high_idx < len(profile):
                included.append(high_idx)
                current_volume += high_vol
                high_idx += 1
            else:
                break

        value_area_profile = profile.loc[included]
        return (value_area_profile["price"].min(), value_area_profile["price"].max())

    def vwap(self) -> float:
        """Calculate Volume Weighted Average Price."""
        return (
            self.data[self.price_col] * self.data[self.volume_col]
        ).sum() / self.data[self.volume_col].sum()

    def cumulative_vwap(self) -> pd.Series:
        """Calculate cumulative VWAP."""
        cum_value = (self.data[self.price_col] * self.data[self.volume_col]).cumsum()
        cum_volume = self.data[self.volume_col].cumsum()
        return cum_value / cum_volume

    def summary(self) -> Dict[str, Any]:
        """Generate volume profile summary."""
        va_low, va_high = self.value_area()

        return {
            "total_volume": self.data[self.volume_col].sum(),
            "avg_volume": self.data[self.volume_col].mean(),
            "vwap": self.vwap(),
            "poc": self.point_of_control(),
            "value_area_low": va_low,
            "value_area_high": va_high,
            "value_area_range": va_high - va_low,
        }


class InstitutionalFlow:
    """
    Detect and analyze institutional order flow.

    Uses order size analysis and flow persistence
    to identify likely institutional activity.
    """

    def __init__(self, trades: pd.DataFrame, size_threshold_pct: float = 0.90):
        """
        Initialize InstitutionalFlow.

        Args:
            trades: Trade data with 'size', 'price', 'side' columns
            size_threshold_pct: Percentile threshold for "large" trades
        """
        self.trades = trades.copy()
        self.threshold_pct = size_threshold_pct

        # Calculate size threshold
        self.size_threshold = trades["size"].quantile(size_threshold_pct)

        # Classify trades
        self.trades["is_large"] = self.trades["size"] >= self.size_threshold

    def large_trade_ratio(self) -> float:
        """Get ratio of volume from large trades."""
        large_volume = self.trades[self.trades["is_large"]]["size"].sum()
        total_volume = self.trades["size"].sum()
        return large_volume / total_volume if total_volume > 0 else 0

    def large_trade_imbalance(self) -> float:
        """
        Calculate imbalance of large trades.

        Returns:
            Ratio from -1 (all sells) to +1 (all buys)
        """
        large_trades = self.trades[self.trades["is_large"]]

        if "side" not in large_trades.columns:
            return 0

        buy_volume = large_trades[large_trades["side"] == "buy"]["size"].sum()
        sell_volume = large_trades[large_trades["side"] == "sell"]["size"].sum()
        total = buy_volume + sell_volume

        if total == 0:
            return 0

        return (buy_volume - sell_volume) / total

    def iceberg_detection(self, child_threshold: float = 0.5) -> pd.DataFrame:
        """
        Detect potential iceberg/hidden orders.

        Looks for clusters of similar-sized trades at same price.

        Args:
            child_threshold: Max size variation for "similar" trades

        Returns:
            DataFrame with potential iceberg clusters
        """
        # Group trades by price and time proximity
        trades = self.trades.copy()
        if "timestamp" in trades.columns:
            trades = trades.sort_values("timestamp")

        # Find clusters of similar-sized trades
        clusters = []
        current_cluster = []

        for idx, row in trades.iterrows():
            if not current_cluster:
                current_cluster.append(row)
            else:
                last = current_cluster[-1]
                size_similar = (
                    abs(row["size"] - last["size"]) / last["size"] < child_threshold
                )
                price_same = row["price"] == last["price"]

                if size_similar and price_same:
                    current_cluster.append(row)
                else:
                    if len(current_cluster) >= 3:  # Minimum cluster size
                        clusters.append(
                            {
                                "price": current_cluster[0]["price"],
                                "avg_size": np.mean(
                                    [t["size"] for t in current_cluster]
                                ),
                                "total_size": sum([t["size"] for t in current_cluster]),
                                "n_trades": len(current_cluster),
                            }
                        )
                    current_cluster = [row]

        return pd.DataFrame(clusters)

    def flow_persistence(self, window: int = 10) -> pd.Series:
        """
        Calculate flow persistence (are large trades in same direction?).

        Args:
            window: Rolling window for calculation

        Returns:
            Series of persistence scores
        """
        if "side" not in self.trades.columns:
            return pd.Series()

        large = self.trades[self.trades["is_large"]].copy()
        large["direction"] = large["side"].map({"buy": 1, "sell": -1})

        # Rolling sum of directions (persistence = consistent direction)
        return large["direction"].rolling(window).sum() / window

    def summary(self) -> Dict[str, Any]:
        """Generate institutional flow summary."""
        return {
            "size_threshold": self.size_threshold,
            "large_trade_ratio": self.large_trade_ratio(),
            "large_trade_imbalance": self.large_trade_imbalance(),
            "n_large_trades": self.trades["is_large"].sum(),
            "large_trade_pct": self.trades["is_large"].mean(),
        }


class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading.

    Based on Easley, Lopez de Prado, and O'Hara (2012).
    Measures toxicity of order flow in real-time.
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        bucket_size: Optional[float] = None,
        n_buckets: int = 50,
    ):
        """
        Initialize VPIN calculator.

        Args:
            trades: Trade data with 'price', 'size', optionally 'side'
            bucket_size: Volume per bucket (auto-calculated if None)
            n_buckets: Number of buckets for rolling VPIN
        """
        self.trades = trades.copy()
        self.n_buckets = n_buckets

        # Calculate bucket size
        total_volume = self.trades["size"].sum()
        if bucket_size is None:
            n_target_buckets = max(50, len(self.trades) // 50)
            self.bucket_size = total_volume / n_target_buckets
        else:
            self.bucket_size = bucket_size

        # Classify trades if needed
        if "side" not in self.trades.columns:
            self._classify_trades_bulk()

        # Calculate VPIN
        self._calculate_vpin()

    def _classify_trades_bulk(self):
        """Classify trades using bulk volume classification."""
        # Use tick rule
        self.trades["price_change"] = self.trades["price"].diff()

        # Bulk classification: split volume proportionally
        # Simplified version - use tick direction
        self.trades["buy_volume"] = np.where(
            self.trades["price_change"] >= 0, self.trades["size"], 0
        )
        self.trades["sell_volume"] = np.where(
            self.trades["price_change"] < 0, self.trades["size"], 0
        )

    def _calculate_vpin(self):
        """Calculate VPIN time series."""
        trades = self.trades.copy()

        # Create volume buckets
        trades["cum_volume"] = trades["size"].cumsum()
        trades["bucket"] = (trades["cum_volume"] / self.bucket_size).astype(int)

        # Aggregate buy/sell volume per bucket
        if "buy_volume" in trades.columns:
            bucket_buy = trades.groupby("bucket")["buy_volume"].sum()
            bucket_sell = trades.groupby("bucket")["sell_volume"].sum()
        else:
            buy_mask = trades["side"] == "buy"
            bucket_buy = trades[buy_mask].groupby("bucket")["size"].sum()
            bucket_sell = trades[~buy_mask].groupby("bucket")["size"].sum()

        # Align indices
        all_buckets = pd.Index(range(trades["bucket"].max() + 1))
        bucket_buy = bucket_buy.reindex(all_buckets, fill_value=0)
        bucket_sell = bucket_sell.reindex(all_buckets, fill_value=0)

        # Calculate VPIN
        imbalance = abs(bucket_buy - bucket_sell)
        volume = bucket_buy + bucket_sell

        self.vpin_series = (
            imbalance.rolling(self.n_buckets).sum()
            / volume.rolling(self.n_buckets).sum()
        )
        self.bucket_data = pd.DataFrame(
            {
                "buy_volume": bucket_buy,
                "sell_volume": bucket_sell,
                "imbalance": imbalance,
                "total_volume": volume,
            }
        )

    def get_vpin(self) -> pd.Series:
        """Get VPIN time series."""
        return self.vpin_series.dropna()

    def current_vpin(self) -> float:
        """Get current VPIN value."""
        clean = self.vpin_series.dropna()
        return clean.iloc[-1] if len(clean) > 0 else 0

    def average_vpin(self) -> float:
        """Get average VPIN."""
        return self.vpin_series.mean()

    def vpin_percentile(self, current: bool = True) -> float:
        """
        Get percentile rank of VPIN.

        Args:
            current: If True, rank current value; else return distribution

        Returns:
            Percentile (0-100) of current VPIN
        """
        clean = self.vpin_series.dropna()
        if len(clean) == 0:
            return 50

        if current:
            current_val = clean.iloc[-1]
            return (clean < current_val).mean() * 100

        return clean.rank(pct=True) * 100

    def toxicity_regime(self) -> str:
        """
        Classify current toxicity regime.

        Returns:
            'low', 'normal', 'elevated', or 'high'
        """
        percentile = self.vpin_percentile()

        if percentile < 25:
            return "low"
        elif percentile < 75:
            return "normal"
        elif percentile < 90:
            return "elevated"
        else:
            return "high"

    def summary(self) -> Dict[str, Any]:
        """Generate VPIN summary."""
        return {
            "current_vpin": self.current_vpin(),
            "average_vpin": self.average_vpin(),
            "vpin_std": self.vpin_series.std(),
            "vpin_percentile": self.vpin_percentile(),
            "toxicity_regime": self.toxicity_regime(),
            "n_buckets": len(self.bucket_data),
            "bucket_size": self.bucket_size,
        }
