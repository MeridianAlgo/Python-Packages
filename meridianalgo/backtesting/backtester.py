"""
Event-driven backtesting framework with realistic market simulation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from .events import (
    Event,
    EventDispatcher,
    EventQueue,
    EventType,
    FillEvent,
    FillStatus,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalEvent,
    SignalType,
)
from .market_simulator import MarketSimulator

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Portfolio position for a symbol."""

    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized P&L."""
        self.market_value = self.quantity * current_price
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity

    def add_fill(self, fill_event: FillEvent) -> float:
        """
        Add a fill to the position and return realized P&L.

        Args:
            fill_event: Fill event to process

        Returns:
            Realized P&L from this fill
        """
        realized_pnl = 0.0

        if fill_event.side == OrderSide.BUY:
            # Buying shares
            if self.quantity >= 0:
                # Adding to long position or starting new long
                total_cost = (self.quantity * self.avg_cost) + (
                    fill_event.fill_quantity * fill_event.fill_price
                )
                self.quantity += fill_event.fill_quantity
                self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0.0
            else:
                # Covering short position
                cover_quantity = min(fill_event.fill_quantity, abs(self.quantity))
                realized_pnl = cover_quantity * (self.avg_cost - fill_event.fill_price)
                self.quantity += cover_quantity

                # If there are remaining shares, start new long position
                remaining_shares = fill_event.fill_quantity - cover_quantity
                if remaining_shares > 0:
                    self.quantity += remaining_shares
                    self.avg_cost = fill_event.fill_price

        else:  # SELL
            # Selling shares
            if self.quantity > 0:
                # Selling from long position
                sell_quantity = min(fill_event.fill_quantity, self.quantity)
                realized_pnl = sell_quantity * (fill_event.fill_price - self.avg_cost)
                self.quantity -= sell_quantity

                # If selling more than we have, start short position
                remaining_shares = fill_event.fill_quantity - sell_quantity
                if remaining_shares > 0:
                    self.quantity -= remaining_shares
                    self.avg_cost = fill_event.fill_price
            else:
                # Adding to short position or starting new short
                total_cost = (abs(self.quantity) * self.avg_cost) + (
                    fill_event.fill_quantity * fill_event.fill_price
                )
                self.quantity -= fill_event.fill_quantity
                self.avg_cost = (
                    total_cost / abs(self.quantity) if self.quantity != 0 else 0.0
                )

        self.realized_pnl += realized_pnl
        return realized_pnl


@dataclass
class Portfolio:
    """Portfolio tracking positions and cash."""

    initial_cash: float
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    total_realized_pnl: float = 0.0

    def __post_init__(self):
        self.cash = self.initial_cash

    def get_position(self, symbol: str) -> Position:
        """Get position for symbol (creates if doesn't exist)."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]

    def process_fill(self, fill_event: FillEvent) -> None:
        """Process a fill event and update portfolio."""
        if fill_event.fill_status == FillStatus.REJECTED:
            return

        position = self.get_position(fill_event.symbol)

        # Calculate trade value
        trade_value = fill_event.fill_quantity * fill_event.fill_price

        # Update cash
        if fill_event.side == OrderSide.BUY:
            self.cash -= trade_value + fill_event.commission
        else:
            self.cash += trade_value - fill_event.commission

        # Update position
        realized_pnl = position.add_fill(fill_event)
        self.total_realized_pnl += realized_pnl
        self.total_commission += fill_event.commission

    def update_market_values(self, market_prices: Dict[str, float]) -> None:
        """Update market values for all positions."""
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                position.update_market_value(market_prices[symbol])

    def get_total_value(self) -> float:
        """Get total portfolio value."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        total_value = self.get_total_value()
        total_unrealized = self.get_total_unrealized_pnl()

        return {
            "total_value": total_value,
            "cash": self.cash,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": total_unrealized,
            "total_pnl": self.total_realized_pnl + total_unrealized,
            "total_commission": self.total_commission,
            "num_positions": len(
                [p for p in self.positions.values() if p.quantity != 0]
            ),
            "return_pct": (total_value - self.initial_cash) / self.initial_cash * 100,
        }


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name
        self.portfolio: Optional[Portfolio] = None
        self.current_prices: Dict[str, float] = {}

    @abstractmethod
    def on_market_event(self, market_event: MarketEvent) -> List[SignalEvent]:
        """Handle market event and generate signals."""
        pass

    def on_fill_event(self, fill_event: FillEvent) -> None:
        """Handle fill event (optional override)."""
        pass

    def set_portfolio(self, portfolio: Portfolio) -> None:
        """Set portfolio reference."""
        self.portfolio = portfolio

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for symbol."""
        self.current_prices[symbol] = price


@dataclass
class BacktestResult:
    """Result of backtesting run."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_commission: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Detailed results
    portfolio_history: pd.DataFrame
    trade_history: pd.DataFrame
    position_history: pd.DataFrame

    # Performance metrics
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    drawdowns: pd.Series

    success: bool = True
    message: str = "Backtest completed successfully"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataHandler(ABC):
    """Abstract base class for market data handlers."""

    @abstractmethod
    def get_market_events(self) -> Iterator[MarketEvent]:
        """Get iterator of market events."""
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol."""
        pass


class PandasDataHandler(DataHandler):
    """Data handler for pandas DataFrame market data."""

    def __init__(self, data: pd.DataFrame, symbols: List[str]):
        """
        Initialize with market data.

        Args:
            data: DataFrame with OHLCV data, MultiIndex (date, symbol) or columns per symbol
            symbols: List of symbols to trade
        """
        self.data = data
        self.symbols = symbols
        self.current_index = 0
        self.latest_prices: Dict[str, float] = {}

        # Prepare data iterator
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data for iteration."""
        self.market_events = []

        if isinstance(self.data.index, pd.MultiIndex):
            # MultiIndex format (date, symbol)
            for date in self.data.index.get_level_values(0).unique():
                for symbol in self.symbols:
                    if (date, symbol) in self.data.index:
                        row = self.data.loc[(date, symbol)]
                        event = self._create_market_event(date, symbol, row)
                        self.market_events.append(event)
        else:
            # Single index format with symbol columns
            for date in self.data.index:
                for symbol in self.symbols:
                    if (
                        f"{symbol}_Close" in self.data.columns
                        or symbol in self.data.columns
                    ):
                        event = self._create_market_event_from_columns(
                            date, symbol, self.data.loc[date]
                        )
                        self.market_events.append(event)

        # Sort by timestamp
        self.market_events.sort(key=lambda x: x.timestamp)

    def _create_market_event(
        self, date: datetime, symbol: str, row: pd.Series
    ) -> MarketEvent:
        """Create market event from data row."""
        return MarketEvent(
            timestamp=date,
            symbol=symbol,
            open_price=row.get("Open", row.get("open", 0.0)),
            high_price=row.get("High", row.get("high", 0.0)),
            low_price=row.get("Low", row.get("low", 0.0)),
            close_price=row.get("Close", row.get("close", 0.0)),
            volume=int(row.get("Volume", row.get("volume", 0))),
        )

    def _create_market_event_from_columns(
        self, date: datetime, symbol: str, row: pd.Series
    ) -> MarketEvent:
        """Create market event from column-based data."""
        # Try different column naming conventions
        open_price = row.get(
            f"{symbol}_Open", row.get(f"Open_{symbol}", row.get("Open", 0.0))
        )
        high_price = row.get(
            f"{symbol}_High", row.get(f"High_{symbol}", row.get("High", 0.0))
        )
        low_price = row.get(
            f"{symbol}_Low", row.get(f"Low_{symbol}", row.get("Low", 0.0))
        )
        close_price = row.get(
            f"{symbol}_Close", row.get(f"Close_{symbol}", row.get(symbol, 0.0))
        )
        volume = int(
            row.get(
                f"{symbol}_Volume", row.get(f"Volume_{symbol}", row.get("Volume", 0))
            )
        )

        return MarketEvent(
            timestamp=date,
            symbol=symbol,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
        )

    def get_market_events(self) -> Iterator[MarketEvent]:
        """Get iterator of market events."""
        for event in self.market_events:
            self.latest_prices[event.symbol] = event.close_price
            yield event

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol."""
        return self.latest_prices.get(symbol)


class EventDrivenBacktester:
    """Main event-driven backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        market_simulator: MarketSimulator = None,
    ):
        self.initial_capital = initial_capital
        self.market_simulator = market_simulator or MarketSimulator()

        # Core components
        self.event_queue = EventQueue()
        self.event_dispatcher = EventDispatcher()
        self.portfolio = Portfolio(initial_capital)

        # Data and strategy
        self.data_handler: Optional[DataHandler] = None
        self.strategy: Optional[Strategy] = None

        # Tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.current_date: Optional[datetime] = None

        # Register event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Setup event handlers."""
        self.event_dispatcher.register_handler(EventType.MARKET, self)
        self.event_dispatcher.register_handler(EventType.SIGNAL, self)
        self.event_dispatcher.register_handler(EventType.ORDER, self)
        self.event_dispatcher.register_handler(EventType.FILL, self)

    def set_data_handler(self, data_handler: DataHandler) -> None:
        """Set data handler."""
        self.data_handler = data_handler

    def set_strategy(self, strategy: Strategy) -> None:
        """Set trading strategy."""
        self.strategy = strategy
        self.strategy.set_portfolio(self.portfolio)

    def handle_event(self, event: Event) -> None:
        """Handle events based on type."""
        if isinstance(event, MarketEvent):
            self._handle_market_event(event)
        elif isinstance(event, SignalEvent):
            self._handle_signal_event(event)
        elif isinstance(event, OrderEvent):
            self._handle_order_event(event)
        elif isinstance(event, FillEvent):
            self._handle_fill_event(event)

    def _handle_market_event(self, event: MarketEvent) -> None:
        """Handle market event."""
        self.current_date = event.timestamp

        # Update market simulator
        self.market_simulator.update_market_state(event)

        # Update strategy prices
        if self.strategy:
            self.strategy.update_price(event.symbol, event.close_price)

        # Update portfolio market values
        current_prices = {event.symbol: event.close_price}
        self.portfolio.update_market_values(current_prices)

        # Generate signals from strategy
        if self.strategy:
            signals = self.strategy.on_market_event(event)
            for signal in signals:
                self.event_queue.put(signal)

        # Record portfolio state
        self._record_portfolio_state()

    def _handle_signal_event(self, event: SignalEvent) -> None:
        """Handle signal event by generating orders."""
        if event.signal_type == SignalType.BUY:
            order = OrderEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=event.target_quantity or 100,
            )
            self.event_queue.put(order)

        elif event.signal_type == SignalType.SELL:
            # Check if we have position to sell
            position = self.portfolio.get_position(event.symbol)
            if position.quantity > 0:
                sell_quantity = event.target_quantity or position.quantity
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=min(sell_quantity, position.quantity),
                )
                self.event_queue.put(order)

    def _handle_order_event(self, event: OrderEvent) -> None:
        """Handle order event by executing through market simulator."""
        fills = self.market_simulator.execute_order(event)
        for fill in fills:
            self.event_queue.put(fill)

    def _handle_fill_event(self, event: FillEvent) -> None:
        """Handle fill event by updating portfolio."""
        self.portfolio.process_fill(event)

        # Notify strategy
        if self.strategy:
            self.strategy.on_fill_event(event)

        # Record trade
        self._record_trade(event)

    def _record_portfolio_state(self) -> None:
        """Record current portfolio state."""
        summary = self.portfolio.get_portfolio_summary()
        summary["timestamp"] = self.current_date
        self.portfolio_history.append(summary)

    def _record_trade(self, fill_event: FillEvent) -> None:
        """Record trade details."""
        if fill_event.fill_status != FillStatus.REJECTED:
            trade = {
                "timestamp": fill_event.timestamp,
                "symbol": fill_event.symbol,
                "side": fill_event.side.value,
                "quantity": fill_event.fill_quantity,
                "price": fill_event.fill_price,
                "commission": fill_event.commission,
                "order_id": fill_event.order_id,
            }
            self.trade_history.append(trade)

    def run_backtest(self) -> BacktestResult:
        """Run the backtest."""
        if not self.data_handler or not self.strategy:
            raise ValueError(
                "Data handler and strategy must be set before running backtest"
            )

        logger.info(f"Starting backtest for strategy: {self.strategy.name}")

        start_time = datetime.now()
        start_date = None
        end_date = None

        # Process all market events
        for market_event in self.data_handler.get_market_events():
            if start_date is None:
                start_date = market_event.timestamp
            end_date = market_event.timestamp

            # Add market event to queue
            self.event_queue.put(market_event)

            # Process all events in queue
            while not self.event_queue.empty():
                event = self.event_queue.get()
                self.event_dispatcher.dispatch_event(event)

        # Calculate final results
        return self._calculate_results(start_date, end_date, start_time)

    def _calculate_results(
        self, start_date: datetime, end_date: datetime, start_time: datetime
    ) -> BacktestResult:
        """Calculate backtest results."""
        # Convert history to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_history)
        if len(portfolio_df) > 0:
            portfolio_df.set_index("timestamp", inplace=True)

        trade_df = pd.DataFrame(self.trade_history)

        # Calculate performance metrics
        final_value = self.portfolio.get_total_value()
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Calculate daily returns
        if len(portfolio_df) > 1:
            daily_returns = portfolio_df["total_value"].pct_change().dropna()
            cumulative_returns = (1 + daily_returns).cumprod() - 1

            # Drawdown calculation
            peak = portfolio_df["total_value"].cummax()
            drawdown = (portfolio_df["total_value"] - peak) / peak
            max_drawdown = drawdown.min()

            # Risk metrics
            if daily_returns.std() > 0:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (
                    daily_returns.mean() / downside_returns.std() * np.sqrt(252)
                )
            else:
                sortino_ratio = 0.0

            # Calmar ratio
            if max_drawdown < 0:
                calmar_ratio = (total_return * 252 / len(daily_returns)) / abs(
                    max_drawdown
                )
            else:
                calmar_ratio = 0.0
        else:
            daily_returns = pd.Series()
            cumulative_returns = pd.Series()
            drawdown = pd.Series()
            max_drawdown = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0

        # Trade statistics
        total_trades = len(trade_df)
        if total_trades > 0:
            # Calculate P&L per trade (simplified)
            winning_trades = 0  # Would need more complex calculation
            losing_trades = 0
        else:
            winning_trades = 0
            losing_trades = 0

        execution_time = datetime.now() - start_time

        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_commission=self.portfolio.total_commission,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            portfolio_history=portfolio_df,
            trade_history=trade_df,
            position_history=pd.DataFrame(),  # Would be implemented
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            drawdowns=drawdown,
            metadata={
                "execution_time": execution_time.total_seconds(),
                "market_simulator_stats": self.market_simulator.get_statistics(),
            },
        )


# Example strategy for testing
class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold strategy."""

    def __init__(self, symbols: List[str]):
        super().__init__("BuyAndHold")
        self.symbols = symbols
        self.bought = set()

    def on_market_event(self, market_event: MarketEvent) -> List[SignalEvent]:
        """Buy once and hold."""
        signals = []

        if (
            market_event.symbol in self.symbols
            and market_event.symbol not in self.bought
        ):
            # Buy signal
            signals.append(
                SignalEvent(
                    timestamp=market_event.timestamp,
                    symbol=market_event.symbol,
                    signal_type=SignalType.BUY,
                    target_quantity=100,
                )
            )
            self.bought.add(market_event.symbol)

        return signals


if __name__ == "__main__":
    # Example usage
    print("Event-Driven Backtesting Framework Example")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    np.random.seed(42)

    # Generate sample price data
    symbols = ["AAPL", "GOOGL"]
    data = {}

    for symbol in symbols:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        data[f"{symbol}_Close"] = prices
        data[f"{symbol}_Volume"] = np.random.randint(100000, 1000000, len(dates))

    market_data = pd.DataFrame(data, index=dates)

    # Create backtester
    backtester = EventDrivenBacktester(initial_capital=100000)

    # Set data handler and strategy
    data_handler = PandasDataHandler(market_data, symbols)
    strategy = BuyAndHoldStrategy(symbols)

    backtester.set_data_handler(data_handler)
    backtester.set_strategy(strategy)

    # Run backtest
    result = backtester.run_backtest()

    print(f"Strategy: {result.strategy_name}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Value: ${result.final_value:,.2f}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Execution Time: {result.metadata['execution_time']:.2f} seconds")
