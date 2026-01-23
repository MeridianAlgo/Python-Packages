"""
Comprehensive order management system with all order types and execution simulation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .events import FillEvent, FillStatus, OrderSide, OrderType

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Time in force enumeration."""

    DAY = "DAY"  # Good for day
    GTC = "GTC"  # Good till cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill
    GTD = "GTD"  # Good till date


@dataclass
class Order:
    """Comprehensive order representation."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Order state
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    remaining_quantity: int = field(init=False)
    avg_fill_price: float = 0.0

    # Timestamps
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    expiry_time: Optional[datetime] = None

    # Bracket order components
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)

    # OCO (One-Cancels-Other) group
    oco_group_id: Optional[str] = None

    # Execution details
    fills: List[FillEvent] = field(default_factory=list)
    total_commission: float = 0.0

    # Metadata
    strategy_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.remaining_quantity = self.quantity

    @property
    def is_active(self) -> bool:
        """Check if order is active (can be filled)."""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]

    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled or cancelled)."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0

    def add_fill(self, fill_event: FillEvent) -> None:
        """Add a fill to the order."""
        self.fills.append(fill_event)
        self.filled_quantity += fill_event.fill_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.total_commission += fill_event.commission
        self.last_update_time = fill_event.timestamp

        # Update average fill price
        if self.filled_quantity > 0:
            total_value = sum(
                fill.fill_quantity * fill.fill_price for fill in self.fills
            )
            self.avg_fill_price = total_value / self.filled_quantity

        # Update status
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Cancel the order."""
        if self.is_active:
            self.status = OrderStatus.CANCELLED
            self.last_update_time = datetime.now()

    def reject(self, reason: str = "") -> None:
        """Reject the order."""
        self.status = OrderStatus.REJECTED
        self.last_update_time = datetime.now()
        if reason:
            self.tags["rejection_reason"] = reason


class BracketOrderBuilder:
    """Builder for bracket orders (parent + stop loss + take profit)."""

    def __init__(self, symbol: str, side: OrderSide, quantity: int):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.parent_order: Optional[Order] = None
        self.stop_loss_order: Optional[Order] = None
        self.take_profit_order: Optional[Order] = None

    def set_parent_order(
        self, order_type: OrderType, price: Optional[float] = None
    ) -> "BracketOrderBuilder":
        """Set the parent (entry) order."""
        order_id = str(uuid.uuid4())
        self.parent_order = Order(
            order_id=order_id,
            symbol=self.symbol,
            side=self.side,
            order_type=order_type,
            quantity=self.quantity,
            price=price,
        )
        return self

    def set_stop_loss(
        self, stop_price: float, limit_price: Optional[float] = None
    ) -> "BracketOrderBuilder":
        """Set the stop loss order."""
        if not self.parent_order:
            raise ValueError("Parent order must be set first")

        order_id = str(uuid.uuid4())
        opposite_side = OrderSide.SELL if self.side == OrderSide.BUY else OrderSide.BUY
        order_type = OrderType.STOP_LIMIT if limit_price else OrderType.STOP

        self.stop_loss_order = Order(
            order_id=order_id,
            symbol=self.symbol,
            side=opposite_side,
            order_type=order_type,
            quantity=self.quantity,
            price=limit_price,
            stop_price=stop_price,
            parent_order_id=self.parent_order.order_id,
        )

        self.parent_order.child_orders.append(order_id)
        return self

    def set_take_profit(self, limit_price: float) -> "BracketOrderBuilder":
        """Set the take profit order."""
        if not self.parent_order:
            raise ValueError("Parent order must be set first")

        order_id = str(uuid.uuid4())
        opposite_side = OrderSide.SELL if self.side == OrderSide.BUY else OrderSide.BUY

        self.take_profit_order = Order(
            order_id=order_id,
            symbol=self.symbol,
            side=opposite_side,
            order_type=OrderType.LIMIT,
            quantity=self.quantity,
            price=limit_price,
            parent_order_id=self.parent_order.order_id,
        )

        self.parent_order.child_orders.append(order_id)
        return self

    def build(self) -> List[Order]:
        """Build the bracket order set."""
        if not self.parent_order:
            raise ValueError("Parent order must be set")

        orders = [self.parent_order]
        if self.stop_loss_order:
            orders.append(self.stop_loss_order)
        if self.take_profit_order:
            orders.append(self.take_profit_order)

        return orders


class OrderValidator:
    """Order validation logic."""

    def __init__(self):
        self.validation_rules: List[Callable[[Order], Tuple[bool, str]]] = [
            self._validate_basic_fields,
            self._validate_price_fields,
            self._validate_quantity,
            self._validate_order_type_consistency,
        ]

    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate an order against all rules."""
        for rule in self.validation_rules:
            is_valid, message = rule(order)
            if not is_valid:
                return False, message
        return True, "Order is valid"

    def _validate_basic_fields(self, order: Order) -> Tuple[bool, str]:
        """Validate basic required fields."""
        if not order.symbol:
            return False, "Symbol is required"
        if not order.side:
            return False, "Order side is required"
        if not order.order_type:
            return False, "Order type is required"
        return True, ""

    def _validate_price_fields(self, order: Order) -> Tuple[bool, str]:
        """Validate price fields based on order type."""
        if order.order_type == OrderType.LIMIT and order.price is None:
            return False, "Limit orders require a price"

        if (
            order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]
            and order.stop_price is None
        ):
            return False, "Stop orders require a stop price"

        if order.order_type == OrderType.STOP_LIMIT and order.price is None:
            return False, "Stop-limit orders require both stop price and limit price"

        if order.price is not None and order.price <= 0:
            return False, "Price must be positive"

        if order.stop_price is not None and order.stop_price <= 0:
            return False, "Stop price must be positive"

        return True, ""

    def _validate_quantity(self, order: Order) -> Tuple[bool, str]:
        """Validate order quantity."""
        if order.quantity <= 0:
            return False, "Quantity must be positive"
        return True, ""

    def _validate_order_type_consistency(self, order: Order) -> Tuple[bool, str]:
        """Validate order type consistency."""
        # Add more sophisticated validation logic here
        return True, ""


class PositionTracker:
    """Track positions and calculate margin requirements."""

    def __init__(self):
        self.positions: Dict[str, int] = {}  # symbol -> net position
        self.avg_costs: Dict[str, float] = {}  # symbol -> average cost
        self.realized_pnl: Dict[str, float] = {}  # symbol -> realized P&L

    def get_position(self, symbol: str) -> int:
        """Get current position for symbol."""
        return self.positions.get(symbol, 0)

    def get_avg_cost(self, symbol: str) -> float:
        """Get average cost for symbol."""
        return self.avg_costs.get(symbol, 0.0)

    def update_position(self, fill_event: FillEvent) -> float:
        """
        Update position from fill and return realized P&L.

        Args:
            fill_event: Fill event to process

        Returns:
            Realized P&L from this fill
        """
        symbol = fill_event.symbol
        current_pos = self.positions.get(symbol, 0)
        current_avg = self.avg_costs.get(symbol, 0.0)

        realized_pnl = 0.0

        if fill_event.side == OrderSide.BUY:
            if current_pos >= 0:
                # Adding to long or starting long
                total_cost = (current_pos * current_avg) + (
                    fill_event.fill_quantity * fill_event.fill_price
                )
                new_pos = current_pos + fill_event.fill_quantity
                new_avg = total_cost / new_pos if new_pos > 0 else 0.0
            else:
                # Covering short
                cover_qty = min(fill_event.fill_quantity, abs(current_pos))
                realized_pnl = cover_qty * (current_avg - fill_event.fill_price)

                remaining_qty = fill_event.fill_quantity - cover_qty
                if remaining_qty > 0:
                    new_pos = remaining_qty
                    new_avg = fill_event.fill_price
                else:
                    new_pos = current_pos + fill_event.fill_quantity
                    new_avg = current_avg
        else:  # SELL
            if current_pos > 0:
                # Selling from long
                sell_qty = min(fill_event.fill_quantity, current_pos)
                realized_pnl = sell_qty * (fill_event.fill_price - current_avg)

                remaining_qty = fill_event.fill_quantity - sell_qty
                if remaining_qty > 0:
                    new_pos = -remaining_qty
                    new_avg = fill_event.fill_price
                else:
                    new_pos = current_pos - fill_event.fill_quantity
                    new_avg = current_avg
            else:
                # Adding to short or starting short
                total_cost = (abs(current_pos) * current_avg) + (
                    fill_event.fill_quantity * fill_event.fill_price
                )
                new_pos = current_pos - fill_event.fill_quantity
                new_avg = total_cost / abs(new_pos) if new_pos != 0 else 0.0

        self.positions[symbol] = new_pos
        self.avg_costs[symbol] = new_avg

        # Update realized P&L
        if symbol not in self.realized_pnl:
            self.realized_pnl[symbol] = 0.0
        self.realized_pnl[symbol] += realized_pnl

        return realized_pnl

    def calculate_margin_requirement(
        self, symbol: str, price: float, margin_rate: float = 0.5
    ) -> float:
        """Calculate margin requirement for position."""
        position = abs(self.get_position(symbol))
        return position * price * margin_rate

    def get_position_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all positions."""
        summary = {}
        for symbol in self.positions:
            if self.positions[symbol] != 0:
                summary[symbol] = {
                    "position": self.positions[symbol],
                    "avg_cost": self.avg_costs[symbol],
                    "realized_pnl": self.realized_pnl.get(symbol, 0.0),
                }
        return summary


class OrderManager:
    """Comprehensive order management system."""

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.oco_groups: Dict[str, List[str]] = {}  # OCO group ID -> order IDs

        self.validator = OrderValidator()
        self.position_tracker = PositionTracker()

        # Statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.cancelled_orders = 0
        self.rejected_orders = 0

    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """
        Submit an order to the system.

        Args:
            order: Order to submit

        Returns:
            Tuple of (success, message)
        """
        # Validate order
        is_valid, message = self.validator.validate_order(order)
        if not is_valid:
            order.reject(message)
            self.orders[order.order_id] = order
            self.rejected_orders += 1
            return False, message

        # Check for duplicate order ID
        if order.order_id in self.orders:
            return False, f"Order ID {order.order_id} already exists"

        # Submit order
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = datetime.now()

        self.orders[order.order_id] = order
        self.active_orders[order.order_id] = order
        self.total_orders += 1

        # Handle OCO groups
        if order.oco_group_id:
            if order.oco_group_id not in self.oco_groups:
                self.oco_groups[order.oco_group_id] = []
            self.oco_groups[order.oco_group_id].append(order.order_id)

        logger.info(f"Order submitted: {order.order_id}")
        return True, "Order submitted successfully"

    def submit_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> Tuple[bool, str, List[str]]:
        """
        Submit a bracket order (entry + stop loss + take profit).

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            entry_price: Entry price (None for market order)
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price

        Returns:
            Tuple of (success, message, order_ids)
        """
        try:
            builder = BracketOrderBuilder(symbol, side, quantity)

            # Set parent order
            entry_type = OrderType.LIMIT if entry_price else OrderType.MARKET
            builder.set_parent_order(entry_type, entry_price)

            # Set stop loss
            if stop_loss_price:
                builder.set_stop_loss(stop_loss_price)

            # Set take profit
            if take_profit_price:
                builder.set_take_profit(take_profit_price)

            orders = builder.build()
            order_ids = []

            # Submit all orders
            for order in orders:
                success, message = self.submit_order(order)
                if not success:
                    # Cancel previously submitted orders
                    for prev_id in order_ids:
                        self.cancel_order(prev_id)
                    return False, f"Failed to submit bracket order: {message}", []
                order_ids.append(order.order_id)

            return True, "Bracket order submitted successfully", order_ids

        except Exception as e:
            return False, f"Error submitting bracket order: {str(e)}", []

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel an order."""
        if order_id not in self.orders:
            return False, f"Order {order_id} not found"

        order = self.orders[order_id]

        if not order.is_active:
            return False, f"Order {order_id} is not active"

        order.cancel()

        # Remove from active orders
        if order_id in self.active_orders:
            del self.active_orders[order_id]

        # Handle OCO cancellation
        if order.oco_group_id and order.oco_group_id in self.oco_groups:
            for oco_order_id in self.oco_groups[order.oco_group_id]:
                if oco_order_id != order_id and oco_order_id in self.active_orders:
                    self.cancel_order(oco_order_id)

        # Cancel child orders for bracket orders
        for child_id in order.child_orders:
            if child_id in self.active_orders:
                self.cancel_order(child_id)

        self.cancelled_orders += 1
        logger.info(f"Order cancelled: {order_id}")
        return True, "Order cancelled successfully"

    def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[int] = None,
        new_price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Modify an existing order."""
        if order_id not in self.orders:
            return False, f"Order {order_id} not found"

        order = self.orders[order_id]

        if not order.is_active:
            return False, f"Order {order_id} is not active"

        # Modify quantity
        if new_quantity is not None:
            if new_quantity <= order.filled_quantity:
                return False, "New quantity must be greater than filled quantity"
            order.quantity = new_quantity
            order.remaining_quantity = new_quantity - order.filled_quantity

        # Modify price
        if new_price is not None:
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                order.price = new_price
            else:
                return False, f"Cannot modify price for {order.order_type} orders"

        order.last_update_time = datetime.now()
        logger.info(f"Order modified: {order_id}")
        return True, "Order modified successfully"

    def process_fill(self, fill_event: FillEvent) -> Tuple[bool, str]:
        """Process a fill event."""
        order_id = fill_event.order_id

        if order_id not in self.orders:
            return False, f"Order {order_id} not found"

        order = self.orders[order_id]

        if not order.is_active:
            return False, f"Order {order_id} is not active"

        # Add fill to order
        order.add_fill(fill_event)

        # Update position tracker
        self.position_tracker.update_position(fill_event)

        # Handle order completion
        if order.status == OrderStatus.FILLED:
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            self.filled_orders += 1

            # Handle bracket order logic
            if order.parent_order_id is None and order.child_orders:
                # Parent order filled, activate child orders
                for child_id in order.child_orders:
                    if child_id in self.orders:
                        child_order = self.orders[child_id]
                        child_order.status = OrderStatus.SUBMITTED
                        self.active_orders[child_id] = child_order

            # Handle OCO logic
            if order.oco_group_id and order.oco_group_id in self.oco_groups:
                for oco_order_id in self.oco_groups[order.oco_group_id]:
                    if oco_order_id != order_id and oco_order_id in self.active_orders:
                        self.cancel_order(oco_order_id)

        logger.info(f"Fill processed: {fill_event.execution_id}")
        return True, "Fill processed successfully"

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders, optionally filtered by symbol."""
        orders = list(self.active_orders.values())
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        return orders

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Order]:
        """Get order history with optional filters."""
        orders = list(self.orders.values())

        if symbol:
            orders = [order for order in orders if order.symbol == symbol]

        if start_date:
            orders = [order for order in orders if order.created_time >= start_date]

        if end_date:
            orders = [order for order in orders if order.created_time <= end_date]

        return sorted(orders, key=lambda x: x.created_time)

    def get_position(self, symbol: str) -> int:
        """Get current position for symbol."""
        return self.position_tracker.get_position(symbol)

    def get_positions_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all positions."""
        return self.position_tracker.get_position_summary()

    def get_statistics(self) -> Dict[str, Any]:
        """Get order management statistics."""
        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "cancelled_orders": self.cancelled_orders,
            "rejected_orders": self.rejected_orders,
            "active_orders": len(self.active_orders),
            "fill_rate": (self.filled_orders / max(1, self.total_orders)) * 100,
            "cancel_rate": (self.cancelled_orders / max(1, self.total_orders)) * 100,
            "reject_rate": (self.rejected_orders / max(1, self.total_orders)) * 100,
        }

    def reset(self) -> None:
        """Reset the order manager."""
        self.orders.clear()
        self.active_orders.clear()
        self.oco_groups.clear()
        self.position_tracker = PositionTracker()

        self.total_orders = 0
        self.filled_orders = 0
        self.cancelled_orders = 0
        self.rejected_orders = 0


if __name__ == "__main__":
    # Example usage
    print("Order Management System Example")
    print("=" * 40)

    # Create order manager
    om = OrderManager()

    # Create and submit a simple order
    order = Order(
        order_id="TEST_001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=150.0,
    )

    success, message = om.submit_order(order)
    print(f"Order submission: {success} - {message}")

    # Create a fill event
    fill = FillEvent(
        timestamp=datetime.now(),
        order_id="TEST_001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        fill_price=150.0,
        fill_quantity=100,
        remaining_quantity=0,
        commission=1.0,
        fill_status=FillStatus.FILLED,
    )

    # Process fill
    success, message = om.process_fill(fill)
    print(f"Fill processing: {success} - {message}")

    # Check position
    position = om.get_position("AAPL")
    print(f"AAPL position: {position}")

    # Get statistics
    stats = om.get_statistics()
    print(f"Statistics: {stats}")

    # Test bracket order
    success, message, order_ids = om.submit_bracket_order(
        symbol="GOOGL",
        side=OrderSide.BUY,
        quantity=50,
        entry_price=2800.0,
        stop_loss_price=2750.0,
        take_profit_price=2900.0,
    )
    print(f"Bracket order: {success} - {message}")
    print(f"Order IDs: {order_ids}")
