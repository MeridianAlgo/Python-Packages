"""
Market simulator with realistic market conditions including bid-ask spreads and market impact.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from .events import FillEvent, FillStatus, MarketEvent, OrderEvent, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current market state for a symbol."""

    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    volatility: float = 0.02  # Daily volatility

    @property
    def mid_price(self) -> float:
        """Mid price between bid and ask."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid_price) * 10000


class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def calculate_slippage(self, order: OrderEvent, market_state: MarketState) -> float:
        """Calculate slippage for an order."""
        pass


class LinearSlippageModel(SlippageModel):
    """Linear slippage model based on order size and volatility."""

    def __init__(
        self, base_slippage_bps: float = 5.0, volume_impact_factor: float = 0.1
    ):
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact_factor = volume_impact_factor

    def calculate_slippage(self, order: OrderEvent, market_state: MarketState) -> float:
        """
        Calculate linear slippage.

        Args:
            order: Order to calculate slippage for
            market_state: Current market state

        Returns:
            Slippage as fraction of price
        """
        # Base slippage
        base_slippage = self.base_slippage_bps / 10000.0

        # Volume impact (larger orders have more impact)
        volume_ratio = order.quantity / max(market_state.volume, 1)
        volume_impact = self.volume_impact_factor * volume_ratio

        # Volatility impact
        volatility_impact = market_state.volatility * 0.1

        total_slippage = base_slippage + volume_impact + volatility_impact

        # Apply direction (buying increases price, selling decreases)
        if order.side == OrderSide.BUY:
            return total_slippage
        else:
            return -total_slippage


class SquareRootSlippageModel(SlippageModel):
    """Square-root slippage model (more realistic for large orders)."""

    def __init__(self, impact_coefficient: float = 0.1):
        self.impact_coefficient = impact_coefficient

    def calculate_slippage(self, order: OrderEvent, market_state: MarketState) -> float:
        """Calculate square-root slippage."""
        # Participation rate
        participation_rate = order.quantity / max(market_state.volume, 1)

        # Square-root impact
        impact = (
            self.impact_coefficient
            * np.sqrt(participation_rate)
            * market_state.volatility
        )

        # Apply direction
        if order.side == OrderSide.BUY:
            return impact
        else:
            return -impact


class MarketSimulator:
    """Realistic market simulator with bid-ask spreads and market impact."""

    def __init__(
        self,
        slippage_model: SlippageModel = None,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
    ):
        self.slippage_model = slippage_model or LinearSlippageModel()
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

        # Market state tracking
        self.market_states: Dict[str, MarketState] = {}
        self.order_book: Dict[str, List[OrderEvent]] = {}

        # Statistics
        self.total_trades = 0
        self.total_volume = 0
        self.total_commission = 0.0

    def update_market_state(self, market_event: MarketEvent) -> None:
        """Update market state from market event."""
        self.market_states[market_event.symbol] = MarketState(
            symbol=market_event.symbol,
            timestamp=market_event.timestamp,
            bid_price=market_event.bid_price,
            ask_price=market_event.ask_price,
            bid_size=market_event.bid_size,
            ask_size=market_event.ask_size,
            last_price=market_event.close_price,
            volume=market_event.volume,
            volatility=self._estimate_volatility(market_event),
        )

    def _estimate_volatility(self, market_event: MarketEvent) -> float:
        """Estimate volatility from price range."""
        if market_event.high_price > market_event.low_price:
            # Use Parkinson volatility estimator
            price_range = np.log(market_event.high_price / market_event.low_price)
            return price_range / (2 * np.sqrt(2 * np.log(2)))
        else:
            return 0.02  # Default 2% daily volatility

    def execute_order(self, order: OrderEvent) -> List[FillEvent]:
        """
        Execute an order and return fill events.

        Args:
            order: Order to execute

        Returns:
            List of fill events (may be partial fills)
        """
        if order.symbol not in self.market_states:
            # No market data available, reject order
            return [self._create_rejection(order, "No market data available")]

        market_state = self.market_states[order.symbol]

        # Handle different order types
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, market_state)
        elif order.order_type == OrderType.LIMIT:
            return self._execute_limit_order(order, market_state)
        elif order.order_type == OrderType.STOP:
            return self._execute_stop_order(order, market_state)
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._execute_stop_limit_order(order, market_state)
        else:
            return [
                self._create_rejection(
                    order, f"Order type {order.order_type} not supported"
                )
            ]

    def _execute_market_order(
        self, order: OrderEvent, market_state: MarketState
    ) -> List[FillEvent]:
        """Execute a market order."""
        # Determine execution price based on side
        if order.side == OrderSide.BUY:
            base_price = market_state.ask_price
            available_size = market_state.ask_size
        else:
            base_price = market_state.bid_price
            available_size = market_state.bid_size

        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(order, market_state)
        execution_price = base_price * (1 + slippage)

        # Determine fill quantity (may be partial if not enough liquidity)
        fill_quantity = min(order.quantity, available_size)

        # Calculate commission
        commission = max(self.min_commission, fill_quantity * self.commission_per_share)

        # Create fill event
        fill_status = (
            FillStatus.FILLED
            if fill_quantity == order.quantity
            else FillStatus.PARTIAL_FILL
        )
        remaining_quantity = order.quantity - fill_quantity

        fill_event = FillEvent(
            timestamp=market_state.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=execution_price,
            fill_quantity=fill_quantity,
            remaining_quantity=remaining_quantity,
            commission=commission,
            fill_status=fill_status,
            metadata={
                "slippage": slippage,
                "base_price": base_price,
                "market_impact": slippage * base_price,
            },
        )

        # Update statistics
        self.total_trades += 1
        self.total_volume += fill_quantity
        self.total_commission += commission

        return [fill_event]

    def _execute_limit_order(
        self, order: OrderEvent, market_state: MarketState
    ) -> List[FillEvent]:
        """Execute a limit order."""
        if order.price is None:
            return [self._create_rejection(order, "Limit order requires price")]

        # Check if limit order can be filled immediately
        can_fill = False
        if order.side == OrderSide.BUY and order.price >= market_state.ask_price:
            can_fill = True
            execution_price = market_state.ask_price
            available_size = market_state.ask_size
        elif order.side == OrderSide.SELL and order.price <= market_state.bid_price:
            can_fill = True
            execution_price = market_state.bid_price
            available_size = market_state.bid_size

        if can_fill:
            # Execute immediately at market price (price improvement)
            fill_quantity = min(order.quantity, available_size)
            commission = max(
                self.min_commission, fill_quantity * self.commission_per_share
            )

            fill_status = (
                FillStatus.FILLED
                if fill_quantity == order.quantity
                else FillStatus.PARTIAL_FILL
            )
            remaining_quantity = order.quantity - fill_quantity

            fill_event = FillEvent(
                timestamp=market_state.timestamp,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=execution_price,
                fill_quantity=fill_quantity,
                remaining_quantity=remaining_quantity,
                commission=commission,
                fill_status=fill_status,
                metadata={"limit_price": order.price, "price_improvement": True},
            )

            self.total_trades += 1
            self.total_volume += fill_quantity
            self.total_commission += commission

            return [fill_event]
        else:
            # Order goes to book (in real backtesting, this would be tracked)
            # For simplicity, we'll reject orders that can't be filled immediately
            return [self._create_rejection(order, "Limit order not marketable")]

    def _execute_stop_order(
        self, order: OrderEvent, market_state: MarketState
    ) -> List[FillEvent]:
        """Execute a stop order."""
        if order.stop_price is None:
            return [self._create_rejection(order, "Stop order requires stop price")]

        # Check if stop is triggered
        triggered = False
        if order.side == OrderSide.BUY and market_state.last_price >= order.stop_price:
            triggered = True
        elif (
            order.side == OrderSide.SELL and market_state.last_price <= order.stop_price
        ):
            triggered = True

        if triggered:
            # Convert to market order
            market_order = OrderEvent(
                timestamp=order.timestamp,
                symbol=order.symbol,
                order_type=OrderType.MARKET,
                side=order.side,
                quantity=order.quantity,
                order_id=order.order_id,
            )
            return self._execute_market_order(market_order, market_state)
        else:
            # Stop not triggered, order remains pending
            return [self._create_rejection(order, "Stop order not triggered")]

    def _execute_stop_limit_order(
        self, order: OrderEvent, market_state: MarketState
    ) -> List[FillEvent]:
        """Execute a stop-limit order."""
        if order.stop_price is None or order.price is None:
            return [
                self._create_rejection(
                    order, "Stop-limit order requires both stop and limit price"
                )
            ]

        # Check if stop is triggered
        triggered = False
        if order.side == OrderSide.BUY and market_state.last_price >= order.stop_price:
            triggered = True
        elif (
            order.side == OrderSide.SELL and market_state.last_price <= order.stop_price
        ):
            triggered = True

        if triggered:
            # Convert to limit order
            limit_order = OrderEvent(
                timestamp=order.timestamp,
                symbol=order.symbol,
                order_type=OrderType.LIMIT,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                order_id=order.order_id,
            )
            return self._execute_limit_order(limit_order, market_state)
        else:
            return [self._create_rejection(order, "Stop-limit order not triggered")]

    def _create_rejection(self, order: OrderEvent, reason: str) -> FillEvent:
        """Create a rejection fill event."""
        return FillEvent(
            timestamp=order.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=0.0,
            fill_quantity=0,
            remaining_quantity=order.quantity,
            commission=0.0,
            fill_status=FillStatus.REJECTED,
            metadata={"rejection_reason": reason},
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_trades": self.total_trades,
            "total_volume": self.total_volume,
            "total_commission": self.total_commission,
            "avg_commission_per_trade": self.total_commission
            / max(1, self.total_trades),
            "avg_volume_per_trade": self.total_volume / max(1, self.total_trades),
        }

    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self.total_trades = 0
        self.total_volume = 0
        self.total_commission = 0.0


# Transaction cost models for different asset classes
class AssetClassCostModel:
    """Transaction cost model for different asset classes."""

    def __init__(self):
        # Default cost parameters by asset class
        self.cost_models = {
            "equity": {
                "commission_per_share": 0.005,
                "min_commission": 1.0,
                "spread_bps": 5.0,
                "market_impact_factor": 0.1,
            },
            "etf": {
                "commission_per_share": 0.0,  # Many brokers offer free ETF trades
                "min_commission": 0.0,
                "spread_bps": 2.0,
                "market_impact_factor": 0.05,
            },
            "bond": {
                "commission_per_share": 0.01,
                "min_commission": 5.0,
                "spread_bps": 20.0,
                "market_impact_factor": 0.2,
            },
            "forex": {
                "commission_per_share": 0.0,
                "min_commission": 0.0,
                "spread_bps": 1.0,
                "market_impact_factor": 0.02,
            },
            "commodity": {
                "commission_per_share": 2.0,
                "min_commission": 10.0,
                "spread_bps": 10.0,
                "market_impact_factor": 0.15,
            },
        }

    def get_cost_model(self, asset_class: str) -> Dict[str, float]:
        """Get cost model parameters for asset class."""
        return self.cost_models.get(asset_class, self.cost_models["equity"])

    def create_simulator(self, asset_class: str) -> MarketSimulator:
        """Create market simulator for asset class."""
        params = self.get_cost_model(asset_class)

        slippage_model = LinearSlippageModel(
            base_slippage_bps=params["spread_bps"],
            volume_impact_factor=params["market_impact_factor"],
        )

        return MarketSimulator(
            slippage_model=slippage_model,
            commission_per_share=params["commission_per_share"],
            min_commission=params["min_commission"],
        )


if __name__ == "__main__":
    # Example usage
    from datetime import datetime

    # Create market simulator
    simulator = MarketSimulator()

    # Create sample market event
    market_event = MarketEvent(
        timestamp=datetime.now(),
        symbol="AAPL",
        open_price=150.0,
        high_price=152.0,
        low_price=149.0,
        close_price=151.0,
        volume=1000000,
        bid_price=150.95,
        ask_price=151.05,
        bid_size=1000,
        ask_size=1000,
    )

    # Update market state
    simulator.update_market_state(market_event)

    # Create sample order
    order = OrderEvent(
        timestamp=datetime.now(),
        symbol="AAPL",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=100,
    )

    # Execute order
    fills = simulator.execute_order(order)

    print("Market Simulator Example:")
    print(f"Market Event: {market_event}")
    print(f"Order: {order}")
    print(f"Fills: {[str(fill) for fill in fills]}")
    print(f"Statistics: {simulator.get_statistics()}")
