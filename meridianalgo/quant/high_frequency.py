"""
High-Frequency Trading Algorithms

Advanced HFT strategies including market making, latency arbitrage,
and liquidity provision algorithms.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class OrderBookLevel:
    """Single level in the order book."""

    price: float
    volume: float
    timestamp: float


@dataclass
class OrderBook:
    """Full order book state."""

    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: float = 0.0

    def get_mid_price(self) -> float:
        """Get mid price."""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    def get_spread(self) -> float:
        """Get bid-ask spread."""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    def get_depth(self, side: str, levels: int = 5) -> float:
        """Get total volume for top N levels."""
        if side == "bid":
            return sum(level.volume for level in self.bids[:levels])
        else:
            return sum(level.volume for level in self.asks[:levels])


class MarketMaking:
    """
    Market making strategy with inventory management.
    """

    def __init__(
        self,
        target_spread_bps: float = 5.0,
        max_inventory: int = 1000,
        inventory_penalty: float = 0.01,
        tick_size: float = 0.01,
    ):
        """
        Initialize market maker.

        Parameters:
        -----------
        target_spread_bps : float
            Target spread in basis points
        max_inventory : int
            Maximum inventory position
        inventory_penalty : float
            Penalty coefficient for inventory deviation
        tick_size : float
            Minimum price increment
        """
        self.target_spread_bps = target_spread_bps
        self.max_inventory = max_inventory
        self.inventory_penalty = inventory_penalty
        self.tick_size = tick_size

        self.position = 0
        self.cash = 0.0
        self.trades = []

    def calculate_quotes(
        self, mid_price: float, volatility: float, order_flow_imbalance: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask quotes using Avellaneda-Stoikov model.

        Parameters:
        -----------
        mid_price : float
            Current mid price
        volatility : float
            Current volatility estimate
        order_flow_imbalance : float
            Order flow imbalance (-1 to 1)

        Returns:
        --------
        Tuple[float, float]
            (bid_price, ask_price)
        """
        # Base spread (in price units)
        base_spread = mid_price * (self.target_spread_bps / 10000)

        # Inventory adjustment (Avellaneda-Stoikov)
        inventory_adjustment = self.inventory_penalty * self.position * volatility

        # Order flow adjustment
        flow_adjustment = 0.5 * base_spread * order_flow_imbalance

        # Calculate quotes
        reservation_price = mid_price - inventory_adjustment
        half_spread = base_spread / 2 + abs(flow_adjustment)

        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread

        # Round to tick size
        bid_price = np.floor(bid_price / self.tick_size) * self.tick_size
        ask_price = np.ceil(ask_price / self.tick_size) * self.tick_size

        # Apply inventory limits
        if self.position >= self.max_inventory:
            # Stop buying, widen bid
            bid_price = mid_price * 0.95
        elif self.position <= -self.max_inventory:
            # Stop selling, widen ask
            ask_price = mid_price * 1.05

        return bid_price, ask_price

    def calculate_quote_sizes(self, base_size: int = 100) -> Tuple[int, int]:
        """
        Calculate optimal quote sizes based on inventory.

        Parameters:
        -----------
        base_size : int
            Base order size

        Returns:
        --------
        Tuple[int, int]
            (bid_size, ask_size)
        """
        # Reduce size as we approach inventory limits
        inventory_pct = abs(self.position) / self.max_inventory

        size_multiplier = max(0.1, 1.0 - inventory_pct)

        bid_size = int(base_size * size_multiplier)
        ask_size = int(base_size * size_multiplier)

        # Adjust based on position
        if self.position > 0:
            # Long inventory, prefer selling
            ask_size = int(bid_size * 1.5)
        elif self.position < 0:
            # Short inventory, prefer buying
            bid_size = int(ask_size * 1.5)

        return bid_size, ask_size

    def on_fill(self, side: str, price: float, quantity: int):
        """
        Handle order fill.

        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        price : float
            Fill price
        quantity : int
            Fill quantity
        """
        if side == "buy":
            self.position += quantity
            self.cash -= price * quantity
        else:  # sell
            self.position -= quantity
            self.cash += price * quantity

        self.trades.append(
            {
                "side": side,
                "price": price,
                "quantity": quantity,
                "position": self.position,
                "cash": self.cash,
            }
        )

    def calculate_pnl(self, current_price: float) -> Dict[str, float]:
        """
        Calculate current P&L.

        Parameters:
        -----------
        current_price : float
            Current market price

        Returns:
        --------
        Dict
            P&L breakdown
        """
        position_value = self.position * current_price
        total_value = self.cash + position_value

        # Calculate realized P&L from trades
        realized_pnl = sum(
            (t["price"] if t["side"] == "sell" else -t["price"]) * t["quantity"]
            for t in self.trades
        )

        unrealized_pnl = position_value

        return {
            "total_pnl": total_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "position": self.position,
            "cash": self.cash,
            "position_value": position_value,
        }


class LatencyArbitrage:
    """
    Latency arbitrage strategy for exploiting price discrepancies
    across venues or due to information arrival.
    """

    def __init__(
        self, latency_threshold_us: float = 100.0, min_profit_bps: float = 1.0
    ):
        """
        Initialize latency arbitrage strategy.

        Parameters:
        -----------
        latency_threshold_us : float
            Maximum acceptable latency in microseconds
        min_profit_bps : float
            Minimum required profit in basis points
        """
        self.latency_threshold = latency_threshold_us
        self.min_profit_bps = min_profit_bps
        self.opportunities = []

    def detect_opportunity(
        self,
        venue1_price: float,
        venue1_time: float,
        venue2_price: float,
        venue2_time: float,
        side: str = "both",
    ) -> Optional[Dict]:
        """
        Detect arbitrage opportunity between two venues.

        Parameters:
        -----------
        venue1_price : float
            Price at venue 1
        venue1_time : float
            Timestamp at venue 1 (microseconds)
        venue2_price : float
            Price at venue 2
        venue2_time : float
            Timestamp at venue 2 (microseconds)
        side : str
            'buy', 'sell', or 'both'

        Returns:
        --------
        Optional[Dict]
            Arbitrage opportunity details if profitable
        """
        # Calculate latency
        latency = abs(venue1_time - venue2_time)

        if latency > self.latency_threshold:
            return None  # Too slow

        # Calculate profit opportunity
        price_diff = venue2_price - venue1_price
        profit_bps = (abs(price_diff) / min(venue1_price, venue2_price)) * 10000

        if profit_bps < self.min_profit_bps:
            return None  # Not profitable enough

        # Determine direction
        if price_diff > 0 and side in ["buy", "both"]:
            # Buy at venue 1, sell at venue 2
            opportunity = {
                "type": "buy_venue1_sell_venue2",
                "buy_price": venue1_price,
                "sell_price": venue2_price,
                "profit_bps": profit_bps,
                "latency_us": latency,
                "timestamp": max(venue1_time, venue2_time),
            }
            self.opportunities.append(opportunity)
            return opportunity

        elif price_diff < 0 and side in ["sell", "both"]:
            # Buy at venue 2, sell at venue 1
            opportunity = {
                "type": "buy_venue2_sell_venue1",
                "buy_price": venue2_price,
                "sell_price": venue1_price,
                "profit_bps": profit_bps,
                "latency_us": latency,
                "timestamp": max(venue1_time, venue2_time),
            }
            self.opportunities.append(opportunity)
            return opportunity

        return None

    def calculate_optimal_size(
        self, opportunity: Dict, max_position: int = 1000, liquidity_limit: int = 500
    ) -> int:
        """
        Calculate optimal trade size for arbitrage.

        Parameters:
        -----------
        opportunity : Dict
            Arbitrage opportunity
        max_position : int
            Maximum position size
        liquidity_limit : int
            Liquidity constraint

        Returns:
        --------
        int
            Optimal trade size
        """
        # Start with liquidity limit
        size = liquidity_limit

        # Reduce based on profit (lower profit = smaller size)
        if opportunity["profit_bps"] < 2.0:
            size = int(size * 0.5)

        # Cap at max position
        size = min(size, max_position)

        return size


class LiquidityProvision:
    """
    Passive liquidity provision strategy.
    """

    def __init__(self, target_fill_rate: float = 0.5, min_edge_bps: float = 2.5):
        """
        Initialize liquidity provision strategy.

        Parameters:
        -----------
        target_fill_rate : float
            Target fill rate (0 to 1)
        min_edge_bps : float
            Minimum edge in basis points
        """
        self.target_fill_rate = target_fill_rate
        self.min_edge_bps = min_edge_bps
        self.fill_history = deque(maxlen=1000)

    def calculate_optimal_skew(
        self, order_book: OrderBook, recent_trades: List[Dict]
    ) -> float:
        """
        Calculate optimal price skew based on order book imbalance.

        Parameters:
        -----------
        order_book : OrderBook
            Current order book
        recent_trades : List[Dict]
            Recent trade history

        Returns:
        --------
        float
            Optimal skew (positive = skew quotes higher, negative = lower)
        """
        # Order book imbalance
        bid_depth = order_book.get_depth("bid", levels=5)
        ask_depth = order_book.get_depth("ask", levels=5)

        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            imbalance = (bid_depth - ask_depth) / total_depth
        else:
            imbalance = 0.0

        # Trade flow imbalance
        if recent_trades:
            buy_volume = sum(t["volume"] for t in recent_trades if t["side"] == "buy")
            sell_volume = sum(t["volume"] for t in recent_trades if t["side"] == "sell")
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                trade_imbalance = (buy_volume - sell_volume) / total_volume
            else:
                trade_imbalance = 0.0
        else:
            trade_imbalance = 0.0

        # Combine signals (50/50 weight)
        skew = 0.5 * imbalance + 0.5 * trade_imbalance

        return skew

    def adjust_for_fill_rate(
        self, current_fill_rate: float, base_edge_bps: float
    ) -> float:
        """
        Adjust edge based on fill rate.

        If fill rate is too low, reduce edge to get more fills.
        If fill rate is too high, increase edge to improve profitability.
        """
        if current_fill_rate < self.target_fill_rate * 0.8:
            # Not enough fills, reduce edge
            return base_edge_bps * 0.8
        elif current_fill_rate > self.target_fill_rate * 1.2:
            # Too many fills, increase edge
            return base_edge_bps * 1.2
        else:
            return base_edge_bps


class HFTSignalGenerator:
    """
    Generate high-frequency trading signals from microstructure data.
    """

    @staticmethod
    def order_flow_toxicity(
        trade_prices: np.ndarray,
        trade_volumes: np.ndarray,
        trade_sides: np.ndarray,
        window: int = 100,
    ) -> np.ndarray:
        """
        Calculate order flow toxicity (informed trading measure).

        High toxicity suggests informed traders are active.
        """
        n = len(trade_prices)
        toxicity = np.zeros(n)

        for i in range(window, n):
            window_prices = trade_prices[i - window : i]
            window_volumes = trade_volumes[i - window : i]
            window_sides = trade_sides[i - window : i]  # 1 for buy, -1 for sell

            # Signed volume
            signed_volume = window_volumes * window_sides

            # Price change
            window_prices[-1] - window_prices[0]

            # Correlation between signed volume and price change
            if np.std(signed_volume) > 0:
                corr = np.corrcoef(signed_volume, window_prices)[0, 1]
                toxicity[i] = abs(corr)
            else:
                toxicity[i] = 0.0

        return toxicity

    @staticmethod
    def volume_clock_returns(
        prices: pd.Series, volumes: pd.Series, volume_bucket: float = 10000
    ) -> pd.Series:
        """
        Calculate returns on volume clock instead of time clock.

        Sample at regular volume intervals rather than time intervals.
        """
        cumul_volume = volumes.cumsum()

        # Find volume bucket boundaries
        volume_levels = np.arange(0, cumul_volume.iloc[-1], volume_bucket)

        sampled_prices = []
        sampled_times = []

        for vol_level in volume_levels:
            # Find first timestamp where cumulative volume exceeds level
            idx = (cumul_volume >= vol_level).idxmax()
            if idx:
                sampled_prices.append(prices.loc[idx])
                sampled_times.append(idx)

        # Calculate returns
        sampled_prices = pd.Series(sampled_prices, index=sampled_times)
        returns = sampled_prices.pct_change()

        return returns

    @staticmethod
    def microstructure_noise_ratio(
        prices: pd.Series, sampling_freq: str = "1S"
    ) -> float:
        """
        Estimate microstructure noise ratio.

        Higher ratio suggests more noise relative to signal.
        """
        # Resample to desired frequency
        resampled = prices.resample(sampling_freq).last().dropna()

        # Calculate realized variance
        returns = resampled.pct_change().dropna()
        rv = np.sum(returns**2)

        # Calculate first-order autocovariance
        autocov = np.cov(returns[1:], returns[:-1])[0, 1]

        # Noise ratio estimate
        if rv > 0:
            noise_ratio = -2 * autocov / rv
            return max(0, min(1, noise_ratio))  # Bound between 0 and 1
        else:
            return 0.0


class MicropriceEstimator:
    """
    Estimate microprice (best price estimate from order book).
    """

    @staticmethod
    def simple_microprice(
        bid_price: float, ask_price: float, bid_volume: float, ask_volume: float
    ) -> float:
        """
        Simple volume-weighted microprice.
        """
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return (bid_price + ask_price) / 2

        return (bid_price * ask_volume + ask_price * bid_volume) / total_volume

    @staticmethod
    def depth_weighted_microprice(
        order_book: OrderBook, depth_levels: int = 3
    ) -> float:
        """
        Microprice using multiple depth levels.
        """
        bid_prices = [level.price for level in order_book.bids[:depth_levels]]
        bid_volumes = [level.volume for level in order_book.bids[:depth_levels]]
        ask_prices = [level.price for level in order_book.asks[:depth_levels]]
        ask_volumes = [level.volume for level in order_book.asks[:depth_levels]]

        total_bid_volume = sum(bid_volumes)
        total_ask_volume = sum(ask_volumes)
        total_volume = total_bid_volume + total_ask_volume

        if total_volume == 0:
            return order_book.get_mid_price()

        # Weighted average
        bid_contribution = sum(p * v for p, v in zip(bid_prices, bid_volumes))
        ask_contribution = sum(p * v for p, v in zip(ask_prices, ask_volumes))

        microprice = (
            bid_contribution * total_ask_volume + ask_contribution * total_bid_volume
        ) / (total_volume * max(total_bid_volume, total_ask_volume))

        return microprice


__all__ = [
    "MarketMaking",
    "LatencyArbitrage",
    "LiquidityProvision",
    "HFTSignalGenerator",
    "MicropriceEstimator",
    "OrderBook",
    "OrderBookLevel",
]
