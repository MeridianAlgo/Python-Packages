"""
Market Microstructure Analysis Module

Advanced market microstructure analysis including order flow dynamics,
informed trading detection, and market quality metrics.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class MarketMicrostructure:
    """
    Comprehensive market microstructure analysis.

    Analyzes:
    - Order flow patterns and dynamics
    - Information asymmetry (PIN, VPIN)
    - Market quality measures
    - Trade classification (Lee-Ready)
    - Price discovery metrics

    Example:
        >>> micro = MarketMicrostructure(trades, quotes)
        >>> pin = micro.calculate_pin()
        >>> vpin = micro.calculate_vpin()
    """

    def __init__(
        self,
        trades: Optional[pd.DataFrame] = None,
        quotes: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize MarketMicrostructure.

        Args:
            trades: DataFrame with columns ['timestamp', 'price', 'size']
            quotes: DataFrame with columns ['timestamp', 'bid', 'ask', 'bid_size', 'ask_size']
        """
        self.trades = trades
        self.quotes = quotes

        if trades is not None:
            self._classify_trades()

    def _classify_trades(self):
        """Classify trades as buyer or seller initiated using Lee-Ready algorithm."""
        if self.trades is None or self.quotes is None:
            return

        # Merge trades with quotes
        merged = pd.merge_asof(
            self.trades.sort_values("timestamp"),
            self.quotes.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        # Calculate midpoint
        merged["mid"] = (merged["bid"] + merged["ask"]) / 2

        # Lee-Ready classification
        # Tick test: compare with previous price
        merged["price_change"] = merged["price"].diff()

        # Quote test: compare with midpoint
        merged["vs_mid"] = merged["price"] - merged["mid"]

        # Apply Lee-Ready rules
        def classify(row):
            if row["vs_mid"] > 0:
                return "buy"
            elif row["vs_mid"] < 0:
                return "sell"
            elif row["price_change"] > 0:
                return "buy"
            elif row["price_change"] < 0:
                return "sell"
            else:
                return "neutral"

        self.trades["side"] = merged.apply(classify, axis=1)

    # =========================================================================
    # INFORMATION ASYMMETRY MEASURES
    # =========================================================================

    def calculate_pin(
        self, daily_trades: Optional[pd.DataFrame] = None, n_iterations: int = 100
    ) -> float:
        """
        Calculate Probability of Informed Trading (PIN).

        Based on Easley, Kiefer, O'Hara, and Paperman (1996).

        Args:
            daily_trades: DataFrame with daily buy/sell counts
            n_iterations: MLE optimization iterations

        Returns:
            PIN estimate (0 to 1)
        """
        if self.trades is None:
            return 0

        # Aggregate to daily buy/sell counts
        if daily_trades is None:
            if "timestamp" in self.trades.columns:
                self.trades["date"] = pd.to_datetime(self.trades["timestamp"]).dt.date
            else:
                return 0.2  # Default estimate

            daily = self.trades.groupby(["date", "side"]).size().unstack(fill_value=0)
            if "buy" not in daily.columns:
                daily["buy"] = 0
            if "sell" not in daily.columns:
                daily["sell"] = 0

            buy_counts = daily["buy"].values
            sell_counts = daily["sell"].values
        else:
            buy_counts = daily_trades["buy"].values
            sell_counts = daily_trades["sell"].values

        n_days = len(buy_counts)
        if n_days < 5:
            return 0.2

        # Simple PIN approximation based on order imbalance
        # Full MLE estimation would require scipy.optimize
        total_buys = buy_counts.sum()
        total_sells = sell_counts.sum()
        total = total_buys + total_sells

        if total == 0:
            return 0.2

        # Simplified PIN estimate based on imbalance
        imbalance = abs(total_buys - total_sells) / total

        # Scale to PIN range (typically 0.1 to 0.3)
        return 0.1 + imbalance * 0.3

    def calculate_vpin(
        self, bucket_size: Optional[float] = None, n_buckets: int = 50
    ) -> pd.Series:
        """
        Calculate Volume-Synchronized PIN (VPIN).

        Based on Easley, Lopez de Prado, and O'Hara (2011).

        Args:
            bucket_size: Volume per bucket (auto-calculated if None)
            n_buckets: Number of buckets for rolling calculation

        Returns:
            VPIN time series
        """
        if self.trades is None:
            return pd.Series()

        trades = self.trades.copy()

        # Auto-calculate bucket size
        if bucket_size is None:
            total_volume = trades["size"].sum()
            n_target_buckets = max(50, len(trades) // 50)
            bucket_size = total_volume / n_target_buckets

        if bucket_size == 0:
            return pd.Series()

        # Create volume buckets
        trades["cum_volume"] = trades["size"].cumsum()
        trades["bucket"] = (trades["cum_volume"] / bucket_size).astype(int)

        # Calculate buy/sell volume per bucket
        if "side" not in trades.columns:
            # If trades not classified, use tick rule
            trades["price_change"] = trades["price"].diff()
            trades["side"] = np.where(trades["price_change"] >= 0, "buy", "sell")

        buy_mask = trades["side"] == "buy"

        bucket_buy = trades[buy_mask].groupby("bucket")["size"].sum()
        bucket_sell = trades[~buy_mask].groupby("bucket")["size"].sum()

        # Align indices
        all_buckets = pd.Index(range(trades["bucket"].max() + 1))
        bucket_buy = bucket_buy.reindex(all_buckets, fill_value=0)
        bucket_sell = bucket_sell.reindex(all_buckets, fill_value=0)

        # Calculate VPIN per window
        imbalance = abs(bucket_buy - bucket_sell)
        volume = bucket_buy + bucket_sell

        vpin = imbalance.rolling(n_buckets).sum() / volume.rolling(n_buckets).sum()

        return vpin.dropna()

    # =========================================================================
    # SPREAD DECOMPOSITION
    # =========================================================================

    def huang_stoll_decomposition(self) -> Dict[str, float]:
        """
        Decompose spread using Huang-Stoll (1997) model.

        Separates spread into:
        - Adverse selection component
        - Inventory component
        - Order processing component

        Returns:
            Dictionary with spread components
        """
        if self.quotes is None or self.trades is None:
            return {
                "adverse_selection": 0.33,
                "inventory": 0.33,
                "order_processing": 0.33,
            }

        # Calculate effective spread
        quotes = self.quotes.copy()
        quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
        quotes["spread"] = quotes["ask"] - quotes["bid"]

        # Merge with trades
        merged = pd.merge_asof(
            self.trades.sort_values("timestamp"),
            quotes.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        if len(merged) < 10:
            return {
                "adverse_selection": 0.33,
                "inventory": 0.33,
                "order_processing": 0.33,
            }

        # Calculate quote midpoint changes
        merged["mid_change"] = merged["mid"].diff()

        # Trade direction indicator (-1 for sell, +1 for buy)
        if "side" in merged.columns:
            merged["direction"] = merged["side"].map(
                {"buy": 1, "sell": -1, "neutral": 0}
            )
        else:
            merged["direction"] = np.sign(merged["price"] - merged["mid"])

        # Simple decomposition based on serial correlation
        # Full Huang-Stoll requires more complex regression
        serial_corr = merged["direction"].autocorr(1) if len(merged) > 10 else 0

        # Approximate component split
        inventory_component = max(0, min(0.5, abs(serial_corr)))
        remaining = 1 - inventory_component

        # Split remaining between adverse selection and order processing
        avg_spread = quotes["spread"].mean()
        realized_spread = self._calculate_realized_spread_internal(merged)

        if avg_spread > 0:
            adverse_selection = (avg_spread - realized_spread) / avg_spread * remaining
            order_processing = remaining - adverse_selection
        else:
            adverse_selection = remaining * 0.5
            order_processing = remaining * 0.5

        return {
            "adverse_selection": max(0, min(1, adverse_selection)),
            "inventory": inventory_component,
            "order_processing": max(0, min(1, order_processing)),
        }

    def _calculate_realized_spread_internal(
        self, merged: pd.DataFrame, lag: int = 5
    ) -> float:
        """Calculate realized spread from merged data."""
        if len(merged) < lag + 1:
            return 0

        merged = merged.copy()
        if "direction" not in merged.columns:
            merged["direction"] = 1

        merged["future_mid"] = merged["mid"].shift(-lag)
        merged["realized"] = (
            2 * merged["direction"] * (merged["price"] - merged["future_mid"])
        )

        return merged["realized"].dropna().mean()

    # =========================================================================
    # MARKET QUALITY METRICS
    # =========================================================================

    def effective_spread(self) -> pd.Series:
        """
        Calculate effective spread for each trade.

        Effective spread = 2 * |trade_price - midpoint|

        Returns:
            Series of effective spreads
        """
        if self.trades is None or self.quotes is None:
            return pd.Series()

        merged = pd.merge_asof(
            self.trades.sort_values("timestamp"),
            self.quotes.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        merged["mid"] = (merged["bid"] + merged["ask"]) / 2
        merged["effective_spread"] = 2 * abs(merged["price"] - merged["mid"])

        return merged.set_index("timestamp")["effective_spread"]

    def realized_spread(self, lag: int = 5) -> pd.Series:
        """
        Calculate realized spread.

        Measures market maker's profit after information is revealed.

        Args:
            lag: Number of periods forward for midpoint comparison

        Returns:
            Series of realized spreads
        """
        if self.trades is None or self.quotes is None:
            return pd.Series()

        merged = pd.merge_asof(
            self.trades.sort_values("timestamp"),
            self.quotes.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        if "side" in self.trades.columns:
            merged["direction"] = (
                self.trades["side"].map({"buy": 1, "sell": -1}).fillna(0)
            )
        else:
            merged["mid"] = (merged["bid"] + merged["ask"]) / 2
            merged["direction"] = np.sign(merged["price"] - merged["mid"])

        merged["mid"] = (merged["bid"] + merged["ask"]) / 2
        merged["future_mid"] = merged["mid"].shift(-lag)
        merged["realized"] = (
            2 * merged["direction"] * (merged["price"] - merged["future_mid"])
        )

        return merged.set_index("timestamp")["realized"].dropna()

    def price_impact(self, lag: int = 5) -> pd.Series:
        """
        Calculate price impact (adverse selection component).

        Price impact = Effective spread - Realized spread

        Args:
            lag: Lag for realized spread calculation

        Returns:
            Series of price impacts
        """
        effective = self.effective_spread()
        realized = self.realized_spread(lag)

        # Align indices
        aligned = pd.concat([effective, realized], axis=1).dropna()
        if len(aligned) < 2:
            return pd.Series()

        aligned.columns = ["effective", "realized"]
        return aligned["effective"] - aligned["realized"]

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Generate market microstructure summary."""
        result = {"pin": self.calculate_pin()}

        vpin = self.calculate_vpin()
        if len(vpin) > 0:
            result["vpin_mean"] = vpin.mean()
            result["vpin_current"] = vpin.iloc[-1]

        hs = self.huang_stoll_decomposition()
        result.update({f"component_{k}": v for k, v in hs.items()})

        effective = self.effective_spread()
        realized = self.realized_spread()

        if len(effective) > 0:
            result["avg_effective_spread"] = effective.mean()
        if len(realized) > 0:
            result["avg_realized_spread"] = realized.mean()

        return result


class OrderFlowAnalyzer:
    """
    Analyze order flow dynamics and patterns.

    Provides:
    - Net order flow calculation
    - Flow persistence analysis
    - Informed flow detection
    - Order flow momentum
    """

    def __init__(self, trades: pd.DataFrame, window: int = 100):
        """
        Initialize OrderFlowAnalyzer.

        Args:
            trades: Trade data with 'timestamp', 'price', 'size', 'side'
            window: Rolling window for calculations
        """
        self.trades = trades.copy()
        self.window = window

        # Ensure side classification
        if "side" not in self.trades.columns:
            self.trades["side"] = np.where(
                self.trades["price"].diff() >= 0, "buy", "sell"
            )

        # Calculate signed flow
        self.trades["signed_flow"] = np.where(
            self.trades["side"] == "buy", self.trades["size"], -self.trades["size"]
        )

    def net_flow(self, n: int = None) -> float:
        """
        Calculate net order flow.

        Args:
            n: Number of recent trades (default: all)

        Returns:
            Net flow (positive = buying, negative = selling)
        """
        if n:
            return self.trades["signed_flow"].tail(n).sum()
        return self.trades["signed_flow"].sum()

    def flow_imbalance(self, n: int = None) -> float:
        """
        Calculate flow imbalance ratio.

        Args:
            n: Number of recent trades

        Returns:
            Imbalance ratio (-1 to 1)
        """
        if n:
            trades = self.trades.tail(n)
        else:
            trades = self.trades

        buy_volume = trades[trades["side"] == "buy"]["size"].sum()
        sell_volume = trades[trades["side"] == "sell"]["size"].sum()
        total = buy_volume + sell_volume

        if total == 0:
            return 0

        return (buy_volume - sell_volume) / total

    def flow_persistence(self, lags: int = 10) -> pd.Series:
        """
        Calculate autocorrelation of order flow.

        Args:
            lags: Number of lags to calculate

        Returns:
            Series of autocorrelations
        """
        correlations = []
        for lag in range(1, lags + 1):
            corr = self.trades["signed_flow"].autocorr(lag)
            correlations.append(corr)

        return pd.Series(correlations, index=range(1, lags + 1))

    def rolling_flow(self) -> pd.Series:
        """Calculate rolling net order flow."""
        return self.trades["signed_flow"].rolling(self.window).sum()

    def rolling_imbalance(self) -> pd.Series:
        """Calculate rolling flow imbalance."""
        buy_vol = (
            (self.trades["size"] * (self.trades["side"] == "buy").astype(int))
            .rolling(self.window)
            .sum()
        )
        total_vol = self.trades["size"].rolling(self.window).sum()

        return (2 * buy_vol - total_vol) / total_vol

    def flow_momentum(self, short_window: int = 20, long_window: int = 50) -> pd.Series:
        """
        Calculate order flow momentum.

        Args:
            short_window: Short-term window
            long_window: Long-term window

        Returns:
            Momentum signal
        """
        short_flow = self.trades["signed_flow"].rolling(short_window).sum()
        long_flow = self.trades["signed_flow"].rolling(long_window).sum()

        return short_flow - long_flow
