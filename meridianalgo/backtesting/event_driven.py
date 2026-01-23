"""
Event-driven backtesting framework with realistic market simulation.
"""

import heapq
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the backtesting system."""

    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    PORTFOLIO_UPDATE = "portfolio_update"
    CUSTOM = "custom"


@dataclass
class Event:
    """Base event class."""

    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.timestamp < other.timestamp


@dataclass
class MarketDataEvent(Event):
    """Market data event."""

    symbol: str = ""
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: int = 0
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None

    def __post_init__(self):
        self.event_type = EventType.MARKET_DATA
        self.data.update(
            {
                "symbol": self.symbol,
                "open": self.open_price,
                "high": self.high_price,
                "low": self.low_price,
                "close": self.close_price,
                "volume": self.volume,
                "bid": self.bid_price,
                "ask": self.ask_price,
            }
        )


@dataclass
class SignalEvent(Event):
    """Trading signal event."""

    symbol: str = ""
    signal_type: str = ""  # 'BUY', 'SELL', 'HOLD'
    strength: float = 0.0  # Signal strength (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.SIGNAL
        self.data.update(
            {
                "symbol": self.symbol,
                "signal_type": self.signal_type,
                "strength": self.strength,
                "metadata": self.metadata,
            }
        )


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, DAY, IOC, FOK
    status: OrderStatus = OrderStatus.PENDING
    created_time: Optional[datetime] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderEvent(Event):
    """Order event."""

    order: Order = None

    def __post_init__(self):
        self.event_type = EventType.ORDER
        self.data.update(
            {
                "order": self.order,
                "symbol": self.order.symbol,
                "order_type": self.order.order_type.value,
                "side": self.order.side,
                "quantity": self.order.quantity,
            }
        )


@dataclass
class Fill:
    """Trade fill representation."""

    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FillEvent(Event):
    """Fill event."""

    fill: Fill = None

    def __post_init__(self):
        self.event_type = EventType.FILL
        self.data.update(
            {
                "fill": self.fill,
                "symbol": self.fill.symbol,
                "side": self.fill.side,
                "quantity": self.fill.quantity,
                "price": self.fill.price,
            }
        )


class MarketSimulator:
    """Realistic market simulation with bid-ask spreads and market impact."""

    def __init__(
        self,
        spread_model: str = "proportional",
        base_spread: float = 0.001,  # 10 bps
        impact_model: str = "linear",
        impact_coefficient: float = 0.1,
        latency_model: str = "fixed",
        base_latency: float = 0.001,
    ):  # 1ms
        self.spread_model = spread_model
        self.base_spread = base_spread
        self.impact_model = impact_model
        self.impact_coefficient = impact_coefficient
        self.latency_model = latency_model
        self.base_latency = base_latency

        # Market state
        self.current_prices = {}
        self.bid_ask_spreads = {}
        self.volumes = {}
        self.volatilities = {}

    def update_market_data(self, market_event: MarketDataEvent):
        """Update market state with new data."""
        symbol = market_event.symbol

        self.current_prices[symbol] = market_event.close_price
        self.volumes[symbol] = market_event.volume

        # Calculate bid-ask spread
        if market_event.bid_price and market_event.ask_price:
            spread = market_event.ask_price - market_event.bid_price
            self.bid_ask_spreads[symbol] = spread
        else:
            # Estimate spread based on model
            spread = self._calculate_spread(
                symbol, market_event.close_price, market_event.volume
            )
            self.bid_ask_spreads[symbol] = spread

        # Update volatility estimate (simplified)
        if symbol not in self.volatilities:
            self.volatilities[symbol] = 0.02  # Default 2% daily vol

    def _calculate_spread(self, symbol: str, price: float, volume: int) -> float:
        """Calculate bid-ask spread based on model."""
        if self.spread_model == "fixed":
            return self.base_spread * price

        elif self.spread_model == "proportional":
            # Spread proportional to volatility and inverse to volume
            vol = self.volatilities.get(symbol, 0.02)
            volume_factor = max(0.1, 1.0 / np.sqrt(max(volume, 1000)))
            return self.base_spread * price * vol * volume_factor

        else:
            return self.base_spread * price

    def calculate_market_impact(self, symbol: str, quantity: float, side: str) -> float:
        """Calculate market impact for an order."""
        if symbol not in self.current_prices:
            return 0.0

        price = self.current_prices[symbol]
        volume = self.volumes.get(symbol, 1000000)
        volatility = self.volatilities.get(symbol, 0.02)

        # Participation rate
        participation_rate = abs(quantity) / max(volume, 1)

        if self.impact_model == "linear":
            impact_rate = self.impact_coefficient * volatility * participation_rate
        elif self.impact_model == "square_root":
            impact_rate = (
                self.impact_coefficient * volatility * np.sqrt(participation_rate)
            )
        elif self.impact_model == "power_law":
            impact_rate = (
                self.impact_coefficient * volatility * (participation_rate**0.6)
            )
        else:
            impact_rate = 0.0

        # Apply impact in direction of trade
        impact = impact_rate * price
        return impact if side == "BUY" else -impact

    def get_execution_price(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Get execution price and slippage for an order."""
        if symbol not in self.current_prices:
            raise ValueError(f"No market data for symbol {symbol}")

        mid_price = self.current_prices[symbol]
        spread = self.bid_ask_spreads.get(symbol, self.base_spread * mid_price)

        # Calculate bid/ask prices
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2

        # Market impact
        market_impact = self.calculate_market_impact(symbol, quantity, side)

        if order_type == OrderType.MARKET:
            # Market order - execute at bid/ask + impact
            if side == "BUY":
                execution_price = ask_price + market_impact
                slippage = execution_price - mid_price
            else:
                execution_price = bid_price + market_impact
                slippage = mid_price - execution_price

        elif order_type == OrderType.LIMIT:
            # Limit order - execute at limit price if possible
            if side == "BUY":
                if limit_price >= ask_price + market_impact:
                    execution_price = ask_price + market_impact
                    slippage = execution_price - mid_price
                else:
                    # Order not filled
                    return None, None
            else:
                if limit_price <= bid_price + market_impact:
                    execution_price = bid_price + market_impact
                    slippage = mid_price - execution_price
                else:
                    # Order not filled
                    return None, None

        else:
            # Other order types - simplified to market execution
            return self.get_execution_price(symbol, side, quantity, OrderType.MARKET)

        return execution_price, slippage


class EventQueue:
    """Priority queue for events ordered by timestamp."""

    def __init__(self):
        self._queue = []
        self._index = 0

    def put(self, event: Event):
        """Add event to queue."""
        heapq.heappush(self._queue, (event.timestamp, self._index, event))
        self._index += 1

    def get(self) -> Optional[Event]:
        """Get next event from queue."""
        if self._queue:
            _, _, event = heapq.heappop(self._queue)
            return event
        return None

    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)


class DataHandler(ABC):
    """Abstract base class for data handlers."""

    @abstractmethod
    def get_latest_data(self, symbol: str, n_bars: int = 1) -> pd.DataFrame:
        """Get latest market data."""
        pass

    @abstractmethod
    def update_bars(self) -> List[MarketDataEvent]:
        """Update to next bar and return market events."""
        pass

    @abstractmethod
    def continue_backtest(self) -> bool:
        """Check if backtest should continue."""
        pass


class HistoricalDataHandler(DataHandler):
    """Historical data handler for backtesting."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize with historical data.

        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
        """
        self.data = data
        self.symbols = list(data.keys())

        # Create unified timeline
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)

        self.dates = sorted(list(all_dates))
        self.current_date_index = 0

        # Current data state
        self.latest_data = {symbol: pd.DataFrame() for symbol in self.symbols}

    def get_latest_data(self, symbol: str, n_bars: int = 1) -> pd.DataFrame:
        """Get latest n bars of data for symbol."""
        if symbol not in self.latest_data:
            return pd.DataFrame()

        return self.latest_data[symbol].tail(n_bars)

    def update_bars(self) -> List[MarketDataEvent]:
        """Update to next bar and return market events."""
        if self.current_date_index >= len(self.dates):
            return []

        current_date = self.dates[self.current_date_index]
        events = []

        for symbol in self.symbols:
            if current_date in self.data[symbol].index:
                bar = self.data[symbol].loc[current_date]

                # Update latest data
                new_bar = pd.DataFrame([bar], index=[current_date])
                if self.latest_data[symbol].empty:
                    self.latest_data[symbol] = new_bar
                else:
                    self.latest_data[symbol] = pd.concat(
                        [self.latest_data[symbol], new_bar]
                    )

                # Keep only recent data for performance
                if len(self.latest_data[symbol]) > 1000:
                    self.latest_data[symbol] = self.latest_data[symbol].tail(1000)

                # Create market data event
                event = MarketDataEvent(
                    timestamp=current_date,
                    symbol=symbol,
                    open_price=bar["Open"],
                    high_price=bar["High"],
                    low_price=bar["Low"],
                    close_price=bar["Close"],
                    volume=int(bar["Volume"]) if "Volume" in bar else 0,
                )
                events.append(event)

        self.current_date_index += 1
        return events

    def continue_backtest(self) -> bool:
        """Check if backtest should continue."""
        return self.current_date_index < len(self.dates)


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name
        self.data_handler = None
        self.portfolio = None
        self.event_queue = None

    def set_handlers(
        self, data_handler: DataHandler, portfolio, event_queue: EventQueue
    ):
        """Set handler references."""
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.event_queue = event_queue

    @abstractmethod
    def calculate_signals(self, market_event: MarketDataEvent) -> List[SignalEvent]:
        """Calculate trading signals based on market data."""
        pass

    def on_market_data(self, market_event: MarketDataEvent):
        """Handle market data event."""
        signals = self.calculate_signals(market_event)
        for signal in signals:
            self.event_queue.put(signal)

    def on_fill(self, fill_event: FillEvent):
        """Handle fill event."""
        pass  # Override in subclass if needed


class SimpleMovingAverageStrategy(Strategy):
    """Simple moving average crossover strategy."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__("SMA_Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.signals = {}  # Track signals by symbol

    def calculate_signals(self, market_event: MarketDataEvent) -> List[SignalEvent]:
        """Calculate SMA crossover signals."""
        symbol = market_event.symbol

        # Get historical data
        data = self.data_handler.get_latest_data(symbol, self.long_window + 1)

        if len(data) < self.long_window:
            return []

        # Calculate moving averages
        short_ma = data["Close"].rolling(window=self.short_window).mean().iloc[-1]
        long_ma = data["Close"].rolling(window=self.long_window).mean().iloc[-1]

        # Previous values for crossover detection
        if len(data) > self.long_window:
            prev_short_ma = (
                data["Close"].rolling(window=self.short_window).mean().iloc[-2]
            )
            prev_long_ma = (
                data["Close"].rolling(window=self.long_window).mean().iloc[-2]
            )
        else:
            return []

        signals = []
        current_signal = self.signals.get(symbol, "HOLD")

        # Bullish crossover
        if (
            prev_short_ma <= prev_long_ma
            and short_ma > long_ma
            and current_signal != "BUY"
        ):
            signal = SignalEvent(
                timestamp=market_event.timestamp,
                symbol=symbol,
                signal_type="BUY",
                strength=0.8,
                metadata={
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "crossover_type": "bullish",
                },
            )
            signals.append(signal)
            self.signals[symbol] = "BUY"

        # Bearish crossover
        elif (
            prev_short_ma >= prev_long_ma
            and short_ma < long_ma
            and current_signal != "SELL"
        ):
            signal = SignalEvent(
                timestamp=market_event.timestamp,
                symbol=symbol,
                signal_type="SELL",
                strength=0.8,
                metadata={
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "crossover_type": "bearish",
                },
            )
            signals.append(signal)
            self.signals[symbol] = "SELL"

        return signals


class Portfolio:
    """Portfolio management for backtesting."""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Holdings
        self.positions = defaultdict(float)  # symbol -> quantity
        self.holdings = defaultdict(float)  # symbol -> market value

        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.returns = []

        # Current market prices
        self.current_prices = {}

    def update_market_value(self, market_event: MarketDataEvent):
        """Update portfolio market value based on current prices."""
        symbol = market_event.symbol
        self.current_prices[symbol] = market_event.close_price

        # Update holdings value
        if symbol in self.positions:
            self.holdings[symbol] = self.positions[symbol] * market_event.close_price

        # Calculate total portfolio value
        total_holdings_value = sum(self.holdings.values())
        cash = self.current_capital - sum(
            pos * self.current_prices.get(sym, 0) for sym, pos in self.positions.items()
        )

        total_value = cash + total_holdings_value

        # Record equity curve
        self.equity_curve.append(
            {
                "timestamp": market_event.timestamp,
                "total_value": total_value,
                "cash": cash,
                "holdings_value": total_holdings_value,
                "positions": dict(self.positions),
                "prices": dict(self.current_prices),
            }
        )

        # Calculate returns
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]["total_value"]
            current_return = (total_value - prev_value) / prev_value
            self.returns.append(current_return)

    def execute_fill(self, fill_event: FillEvent):
        """Execute a fill and update portfolio."""
        fill = fill_event.fill

        # Update positions
        if fill.side == "BUY":
            self.positions[fill.symbol] += fill.quantity
        else:  # SELL
            self.positions[fill.symbol] -= fill.quantity

        # Update cash (including commission and slippage costs)
        trade_value = fill.quantity * fill.price
        total_cost = trade_value + fill.commission + abs(fill.slippage * fill.quantity)

        if fill.side == "BUY":
            self.current_capital -= total_cost
        else:
            self.current_capital += (
                trade_value - fill.commission - abs(fill.slippage * fill.quantity)
            )

        # Record trade
        self.trades.append(
            {
                "timestamp": fill.timestamp,
                "symbol": fill.symbol,
                "side": fill.side,
                "quantity": fill.quantity,
                "price": fill.price,
                "commission": fill.commission,
                "slippage": fill.slippage,
                "trade_value": trade_value,
                "total_cost": total_cost,
            }
        )

        logger.debug(
            f"Executed {fill.side} {fill.quantity} {fill.symbol} @ {fill.price}"
        )

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        if not self.equity_curve:
            return {}

        latest = self.equity_curve[-1]
        total_return = (
            latest["total_value"] - self.initial_capital
        ) / self.initial_capital

        return {
            "initial_capital": self.initial_capital,
            "current_value": latest["total_value"],
            "cash": latest["cash"],
            "holdings_value": latest["holdings_value"],
            "total_return": total_return,
            "positions": dict(self.positions),
            "n_trades": len(self.trades),
            "equity_curve_length": len(self.equity_curve),
        }


class BacktestEngine:
    """Main backtesting engine."""

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        market_simulator: MarketSimulator = None,
    ):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.market_simulator = market_simulator or MarketSimulator()

        # Event handling
        self.event_queue = EventQueue()

        # Set up strategy handlers
        self.strategy.set_handlers(data_handler, portfolio, self.event_queue)

        # Order management
        self.orders = {}  # order_id -> Order
        self.order_counter = 0

        # Commission model
        self.commission_rate = 0.001  # 0.1%

        # Execution settings
        self.execution_delay = timedelta(seconds=0)  # Execution delay

    def run_backtest(self) -> Dict[str, Any]:
        """Run the complete backtest."""
        logger.info("Starting backtest...")

        while self.data_handler.continue_backtest():
            # Get new market data
            market_events = self.data_handler.update_bars()

            # Process market events
            for market_event in market_events:
                self.event_queue.put(market_event)

            # Process all events in queue
            while not self.event_queue.empty():
                event = self.event_queue.get()
                self._handle_event(event)

        logger.info("Backtest completed")
        return self._generate_results()

    def _handle_event(self, event: Event):
        """Handle different types of events."""
        if event.event_type == EventType.MARKET_DATA:
            self._handle_market_data(event)
        elif event.event_type == EventType.SIGNAL:
            self._handle_signal(event)
        elif event.event_type == EventType.ORDER:
            self._handle_order(event)
        elif event.event_type == EventType.FILL:
            self._handle_fill(event)

    def _handle_market_data(self, event: MarketDataEvent):
        """Handle market data event."""
        # Update market simulator
        self.market_simulator.update_market_data(event)

        # Update portfolio valuation
        self.portfolio.update_market_value(event)

        # Send to strategy
        self.strategy.on_market_data(event)

        # Check for order fills
        self._check_order_fills(event)

    def _handle_signal(self, event: SignalEvent):
        """Handle trading signal event."""
        # Convert signal to order (simplified)
        if event.signal_type in ["BUY", "SELL"]:
            # Calculate position size (simplified - fixed $10k per trade)
            target_value = 10000
            current_price = self.market_simulator.current_prices.get(event.symbol)

            if current_price:
                quantity = target_value / current_price

                # Create market order
                order = Order(
                    order_id=f"ORDER_{self.order_counter}",
                    symbol=event.symbol,
                    order_type=OrderType.MARKET,
                    side=event.signal_type,
                    quantity=quantity,
                    created_time=event.timestamp,
                )

                self.order_counter += 1

                # Submit order
                order_event = OrderEvent(
                    timestamp=event.timestamp + self.execution_delay, order=order
                )

                self.event_queue.put(order_event)

    def _handle_order(self, event: OrderEvent):
        """Handle order event."""
        order = event.order
        self.orders[order.order_id] = order

        # Try to execute order immediately (market orders)
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order, event.timestamp)

    def _handle_fill(self, event: FillEvent):
        """Handle fill event."""
        # Update portfolio
        self.portfolio.execute_fill(event)

        # Notify strategy
        self.strategy.on_fill(event)

        # Update order status
        fill = event.fill
        if fill.order_id in self.orders:
            order = self.orders[fill.order_id]
            order.filled_quantity += fill.quantity

            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

    def _execute_market_order(self, order: Order, timestamp: datetime):
        """Execute market order."""
        try:
            execution_price, slippage = self.market_simulator.get_execution_price(
                order.symbol, order.side, order.quantity, order.order_type
            )

            if execution_price is not None:
                # Calculate commission
                trade_value = order.quantity * execution_price
                commission = trade_value * self.commission_rate

                # Create fill
                fill = Fill(
                    fill_id=f"FILL_{len(self.portfolio.trades)}",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=execution_price,
                    timestamp=timestamp,
                    commission=commission,
                    slippage=slippage or 0.0,
                )

                # Create fill event
                fill_event = FillEvent(timestamp=timestamp, fill=fill)
                self.event_queue.put(fill_event)

                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.REJECTED

        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED

    def _check_order_fills(self, market_event: MarketDataEvent):
        """Check if any pending orders should be filled."""
        # For limit orders, stop orders, etc.
        # This is a simplified implementation
        pass

    def _generate_results(self) -> Dict[str, Any]:
        """Generate backtest results."""
        portfolio_summary = self.portfolio.get_portfolio_summary()

        # Calculate performance metrics
        if self.portfolio.returns:
            returns_series = pd.Series(self.portfolio.returns)

            # Basic metrics
            total_return = portfolio_summary["total_return"]
            annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1
            volatility = returns_series.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            # Drawdown
            equity_values = [
                point["total_value"] for point in self.portfolio.equity_curve
            ]
            equity_series = pd.Series(equity_values)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown = drawdown.min()

            performance_metrics = {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "n_trades": len(self.portfolio.trades),
                "win_rate": self._calculate_win_rate(),
            }
        else:
            performance_metrics = {}

        return {
            "portfolio_summary": portfolio_summary,
            "performance_metrics": performance_metrics,
            "equity_curve": self.portfolio.equity_curve,
            "trades": self.portfolio.trades,
            "returns": self.portfolio.returns,
        }

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades."""
        if not self.portfolio.trades:
            return 0.0

        # Group trades by symbol to calculate P&L
        # This is simplified - would need more sophisticated P&L calculation
        winning_trades = 0
        total_trades = 0

        positions = defaultdict(list)

        for trade in self.portfolio.trades:
            positions[trade["symbol"]].append(trade)

        for symbol, symbol_trades in positions.items():
            # Simple FIFO P&L calculation
            buys = [t for t in symbol_trades if t["side"] == "BUY"]
            sells = [t for t in symbol_trades if t["side"] == "SELL"]

            for sell in sells:
                if buys:
                    buy = buys.pop(0)
                    pnl = (sell["price"] - buy["price"]) * sell["quantity"]
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1

        return winning_trades / total_trades if total_trades > 0 else 0.0
