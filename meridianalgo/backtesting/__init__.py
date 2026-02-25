"""
Backtesting module for MeridianAlgo.

Provides comprehensive backtesting framework with event-driven architecture,
order management, and performance analytics.
"""

from .backtester import Portfolio, Position
from .engine import BacktestEngine
from .event_driven import BacktestEngine as EventDrivenBacktestEngine
from .event_driven import (
    DataHandler,
    EventQueue,
    EventType,
    Fill,
    FillEvent,
    HistoricalDataHandler,
    MarketDataEvent,
    MarketSimulator,
    Order,
    OrderEvent,
    OrderStatus,
    OrderType,
    SignalEvent,
    SimpleMovingAverageStrategy,
    Strategy,
)
from .event_driven import Portfolio as EventDrivenPortfolio
from .events import (
    Event,
    EventDispatcher,
    EventHandler,
    FillStatus,
    MarketEvent,
    OrderSide,
    SignalType,
)
from .events import EventQueue as EventQueueV2
from .events import EventType as EventTypeV2
from .events import FillEvent as FillEventV2
from .events import OrderEvent as OrderEventV2
from .events import OrderType as OrderTypeV2
from .events import SignalEvent as SignalEventV2
from .market_simulator import (
    AssetClassCostModel,
    LinearSlippageModel,
    MarketState,
    SlippageModel,
    SquareRootSlippageModel,
)
from .market_simulator import MarketSimulator as MarketSimulatorV2
from .order_management import (
    BracketOrderBuilder,
    OrderManager,
    OrderValidator,
    PositionTracker,
    TimeInForce,
)
from .order_management import Order as OrderV2
from .order_management import OrderStatus as OrderStatusV2
from .performance_analytics import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    RollingPerformanceAnalyzer,
)

# Alias
Backtest = BacktestEngine

__all__ = [
    # Main engine
    "BacktestEngine",
    "Backtest",
    "EventDrivenBacktestEngine",
    # Events
    "Event",
    "EventType",
    "EventTypeV2",
    "MarketEvent",
    "SignalEvent",
    "SignalEventV2",
    "OrderEvent",
    "OrderEventV2",
    "FillEvent",
    "FillEventV2",
    "FillStatus",
    "OrderSide",
    "SignalType",
    "EventQueue",
    "EventQueueV2",
    "EventDispatcher",
    "EventHandler",
    # Orders
    "Order",
    "OrderV2",
    "OrderType",
    "OrderTypeV2",
    "OrderStatus",
    "OrderStatusV2",
    "OrderManager",
    "OrderValidator",
    "BracketOrderBuilder",
    "TimeInForce",
    # Market simulation
    "MarketSimulator",
    "MarketSimulatorV2",
    "MarketState",
    "SlippageModel",
    "LinearSlippageModel",
    "SquareRootSlippageModel",
    "AssetClassCostModel",
    # Portfolio and positions
    "Portfolio",
    "EventDrivenPortfolio",
    "Position",
    "PositionTracker",
    # Data handling
    "DataHandler",
    "HistoricalDataHandler",
    # Strategies
    "Strategy",
    "SimpleMovingAverageStrategy",
    # Performance analytics
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "RollingPerformanceAnalyzer",
    # Market data
    "MarketDataEvent",
    # Fill
    "Fill",
]
