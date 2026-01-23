"""
Order Book Analysis Module

Comprehensive order book analytics including depth analysis, imbalance detection,
price impact estimation, and order flow analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Level2Data:
    """Container for Level 2 order book data."""

    timestamp: datetime
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray

    @property
    def best_bid(self) -> float:
        """Get best bid price."""
        return self.bid_prices[0] if len(self.bid_prices) > 0 else 0

    @property
    def best_ask(self) -> float:
        """Get best ask price."""
        return self.ask_prices[0] if len(self.ask_prices) > 0 else 0

    @property
    def mid_price(self) -> float:
        """Get mid price."""
        return (
            (self.best_bid + self.best_ask) / 2
            if self.best_bid and self.best_ask
            else 0
        )

    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        """Get spread in basis points."""
        mid = self.mid_price
        return (self.spread / mid) * 10000 if mid > 0 else 0


@dataclass
class OrderBook:
    """
    Full order book representation.

    Supports multiple levels, time series analysis, and reconstruction.
    """

    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    asks: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    timestamp: Optional[datetime] = None
    symbol: Optional[str] = None

    def update(self, side: str, price: float, size: float):
        """
        Update order book with new order.

        Args:
            side: 'bid' or 'ask'
            price: Price level
            size: New size (0 to remove level)
        """
        if side == "bid":
            self._update_side(self.bids, price, size, reverse=True)
        else:
            self._update_side(self.asks, price, size, reverse=False)

    def _update_side(
        self,
        orders: List[Tuple[float, float]],
        price: float,
        size: float,
        reverse: bool,
    ):
        """Update one side of the order book."""
        # Remove existing price level
        orders[:] = [(p, s) for p, s in orders if p != price]

        # Add new level if size > 0
        if size > 0:
            orders.append((price, size))
            orders.sort(key=lambda x: x[0], reverse=reverse)

    @property
    def best_bid(self) -> Tuple[float, float]:
        """Get best bid (price, size)."""
        return self.bids[0] if self.bids else (0, 0)

    @property
    def best_ask(self) -> Tuple[float, float]:
        """Get best ask (price, size)."""
        return self.asks[0] if self.asks else (0, 0)

    @property
    def mid_price(self) -> float:
        """Get mid price."""
        bid = self.best_bid[0]
        ask = self.best_ask[0]
        return (bid + ask) / 2 if bid and ask else 0

    @property
    def microprice(self) -> float:
        """
        Get microprice (size-weighted mid price).

        Adjusts mid price based on relative sizes at best bid/ask.
        """
        bid_price, bid_size = self.best_bid
        ask_price, ask_size = self.best_ask

        if bid_size + ask_size == 0:
            return self.mid_price

        # Weight towards the side with more size
        total_size = bid_size + ask_size
        return (bid_price * ask_size + ask_price * bid_size) / total_size

    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        return self.best_ask[0] - self.best_bid[0]

    @property
    def spread_bps(self) -> float:
        """Get spread in basis points."""
        mid = self.mid_price
        return (self.spread / mid) * 10000 if mid > 0 else 0

    def depth(self, levels: int = 5) -> Dict[str, float]:
        """
        Calculate order book depth.

        Args:
            levels: Number of levels to consider

        Returns:
            Dictionary with bid/ask depth
        """
        bid_depth = sum(s for _, s in self.bids[:levels])
        ask_depth = sum(s for _, s in self.asks[:levels])

        return {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": bid_depth + ask_depth,
            "depth_imbalance": (
                (bid_depth - ask_depth) / (bid_depth + ask_depth)
                if (bid_depth + ask_depth) > 0
                else 0
            ),
        }

    def price_impact(self, order_size: float, side: str) -> float:
        """
        Estimate price impact for a given order size.

        Args:
            order_size: Size of order to execute
            side: 'buy' or 'sell'

        Returns:
            Estimated price impact in basis points
        """
        if side == "buy":
            levels = self.asks
        else:
            levels = self.bids

        if not levels:
            return 0

        remaining = order_size
        total_value = 0
        total_size = 0

        for price, size in levels:
            if remaining <= 0:
                break

            fill_size = min(remaining, size)
            total_value += fill_size * price
            total_size += fill_size
            remaining -= fill_size

        if total_size == 0:
            return 0

        avg_price = total_value / total_size
        mid = self.mid_price

        return abs((avg_price - mid) / mid) * 10000


class OrderBookAnalyzer:
    """
    Comprehensive order book analysis.

    Provides:
    - Depth analysis and imbalance detection
    - Order flow toxicity (VPIN-style)
    - Price impact estimation
    - Liquidity metrics
    - Market maker detection

    Example:
        >>> analyzer = OrderBookAnalyzer()
        >>> analyzer.update(order_book)
        >>> imbalance = analyzer.order_imbalance()
        >>> toxicity = analyzer.flow_toxicity()
    """

    def __init__(self, levels: int = 10, window: int = 100):
        """
        Initialize OrderBookAnalyzer.

        Args:
            levels: Number of price levels to track
            window: Rolling window for calculations
        """
        self.levels = levels
        self.window = window
        self.history: List[OrderBook] = []
        self.trades: List[Dict[str, Any]] = []

    def update(self, order_book: OrderBook):
        """
        Update analyzer with new order book snapshot.

        Args:
            order_book: Current order book state
        """
        self.history.append(order_book)

        # Keep only recent history
        if len(self.history) > self.window * 10:
            self.history = self.history[-self.window * 10 :]

    def add_trade(
        self, price: float, size: float, side: str, timestamp: Optional[datetime] = None
    ):
        """
        Add trade for order flow analysis.

        Args:
            price: Trade price
            size: Trade size
            side: 'buy' or 'sell'
            timestamp: Trade timestamp
        """
        self.trades.append(
            {
                "price": price,
                "size": size,
                "side": side,
                "timestamp": timestamp or datetime.now(),
            }
        )

        # Keep only recent trades
        if len(self.trades) > self.window * 100:
            self.trades = self.trades[-self.window * 100 :]

    # =========================================================================
    # ORDER BOOK METRICS
    # =========================================================================

    def current_spread(self) -> float:
        """Get current bid-ask spread in bps."""
        if not self.history:
            return 0
        return self.history[-1].spread_bps

    def average_spread(self, n: int = None) -> float:
        """
        Calculate average spread over recent history.

        Args:
            n: Number of snapshots to average (default: window)
        """
        n = n or self.window
        snapshots = self.history[-n:]

        if not snapshots:
            return 0

        return np.mean([ob.spread_bps for ob in snapshots])

    def spread_volatility(self, n: int = None) -> float:
        """Calculate spread volatility."""
        n = n or self.window
        snapshots = self.history[-n:]

        if len(snapshots) < 2:
            return 0

        spreads = [ob.spread_bps for ob in snapshots]
        return np.std(spreads)

    def order_imbalance(self, levels: int = None) -> float:
        """
        Calculate order imbalance.

        Positive = more bids (buying pressure)
        Negative = more asks (selling pressure)

        Args:
            levels: Number of levels to consider

        Returns:
            Imbalance ratio (-1 to 1)
        """
        if not self.history:
            return 0

        levels = levels or self.levels
        depth = self.history[-1].depth(levels)

        return depth["depth_imbalance"]

    def depth_ratio(self, levels: int = None) -> float:
        """
        Calculate bid depth / ask depth ratio.

        Args:
            levels: Number of levels to consider

        Returns:
            Ratio of bid to ask depth
        """
        if not self.history:
            return 1

        levels = levels or self.levels
        depth = self.history[-1].depth(levels)

        if depth["ask_depth"] == 0:
            return np.inf

        return depth["bid_depth"] / depth["ask_depth"]

    def microprice(self) -> float:
        """Get current microprice."""
        if not self.history:
            return 0
        return self.history[-1].microprice

    def mid_price(self) -> float:
        """Get current mid price."""
        if not self.history:
            return 0
        return self.history[-1].mid_price

    # =========================================================================
    # ORDER FLOW ANALYSIS
    # =========================================================================

    def order_flow_imbalance(self, n: int = None) -> float:
        """
        Calculate Order Flow Imbalance (OFI).

        Based on changes in best bid/ask quantities.

        Args:
            n: Number of snapshots to analyze

        Returns:
            OFI value
        """
        n = n or self.window
        snapshots = self.history[-n:]

        if len(snapshots) < 2:
            return 0

        ofi = 0
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]

            # Bid contribution
            if curr.best_bid[0] > prev.best_bid[0]:
                ofi += curr.best_bid[1]
            elif curr.best_bid[0] == prev.best_bid[0]:
                ofi += curr.best_bid[1] - prev.best_bid[1]
            else:
                ofi -= prev.best_bid[1]

            # Ask contribution
            if curr.best_ask[0] < prev.best_ask[0]:
                ofi -= curr.best_ask[1]
            elif curr.best_ask[0] == prev.best_ask[0]:
                ofi -= curr.best_ask[1] - prev.best_ask[1]
            else:
                ofi += prev.best_ask[1]

        return ofi

    def trade_imbalance(self, n: int = None) -> float:
        """
        Calculate trade imbalance (buy volume - sell volume).

        Args:
            n: Number of trades to analyze

        Returns:
            Imbalance ratio (-1 to 1)
        """
        n = n or len(self.trades)
        trades = self.trades[-n:]

        if not trades:
            return 0

        buy_volume = sum(t["size"] for t in trades if t["side"] == "buy")
        sell_volume = sum(t["size"] for t in trades if t["side"] == "sell")
        total = buy_volume + sell_volume

        if total == 0:
            return 0

        return (buy_volume - sell_volume) / total

    def flow_toxicity(self, bucket_size: float = 0.01) -> float:
        """
        Calculate order flow toxicity (VPIN-like).

        Higher values indicate more informed trading.

        Args:
            bucket_size: Volume bucket size as fraction of daily volume

        Returns:
            Toxicity measure (0 to 1)
        """
        if len(self.trades) < 10:
            return 0

        trades = pd.DataFrame(self.trades)
        total_volume = trades["size"].sum()
        bucket_vol = total_volume * bucket_size

        if bucket_vol == 0:
            return 0

        # Create volume buckets
        trades["cum_volume"] = trades["size"].cumsum()
        trades["bucket"] = (trades["cum_volume"] / bucket_vol).astype(int)

        # Calculate imbalance per bucket
        buy_mask = trades["side"] == "buy"
        bucket_buy = trades[buy_mask].groupby("bucket")["size"].sum()
        bucket_sell = trades[~buy_mask].groupby("bucket")["size"].sum()

        imbalances = abs(bucket_buy.subtract(bucket_sell, fill_value=0))
        volumes = bucket_buy.add(bucket_sell, fill_value=0)

        # VPIN = average absolute imbalance / average bucket volume
        if volumes.sum() == 0:
            return 0

        return imbalances.sum() / volumes.sum()

    # =========================================================================
    # PRICE IMPACT
    # =========================================================================

    def estimate_impact(self, order_size: float, side: str) -> float:
        """
        Estimate price impact for order.

        Args:
            order_size: Size of order
            side: 'buy' or 'sell'

        Returns:
            Estimated impact in basis points
        """
        if not self.history:
            return 0

        return self.history[-1].price_impact(order_size, side)

    def kyle_lambda(self) -> float:
        """
        Estimate Kyle's lambda (price impact coefficient).

        Measures how much price moves per unit of signed order flow.
        """
        if len(self.trades) < 20 or len(self.history) < 20:
            return 0

        trades = pd.DataFrame(self.trades[-100:])

        # Signed order flow
        trades["signed_flow"] = trades.apply(
            lambda x: x["size"] if x["side"] == "buy" else -x["size"], axis=1
        )

        # Price changes (need to align with order book history)
        if len(self.history) < 2:
            return 0

        prices = [ob.mid_price for ob in self.history[-100:]]
        returns = np.diff(prices) / prices[:-1]

        n = min(len(returns), len(trades) - 1)
        if n < 10:
            return 0

        # Regression: return = lambda * signed_flow
        flow = trades["signed_flow"].values[:n]
        ret = returns[:n]

        cov = np.cov(flow, ret)
        if cov[0, 0] == 0:
            return 0

        return cov[0, 1] / cov[0, 0]

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Generate comprehensive order book analysis summary."""
        return {
            "current_spread_bps": self.current_spread(),
            "average_spread_bps": self.average_spread(),
            "spread_volatility": self.spread_volatility(),
            "order_imbalance": self.order_imbalance(),
            "depth_ratio": self.depth_ratio(),
            "mid_price": self.mid_price(),
            "microprice": self.microprice(),
            "order_flow_imbalance": self.order_flow_imbalance(),
            "trade_imbalance": self.trade_imbalance(),
            "flow_toxicity": self.flow_toxicity(),
            "kyle_lambda": self.kyle_lambda(),
            "num_snapshots": len(self.history),
            "num_trades": len(self.trades),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert summary to DataFrame."""
        summary = self.summary()
        return pd.DataFrame({"Value": summary.values()}, index=summary.keys())
