"""
Backtesting module for MeridianAlgo.

Provides comprehensive backtesting framework with event-driven architecture,
order management, and performance analytics.
"""

from .backtester import Portfolio, Position
from .engine import BacktestEngine
from .event_driven import (
    BacktestEngine as EventDrivenBacktestEngine,
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
    Portfolio as EventDrivenPortfolio,
    SignalEvent,
    SimpleMovingAverageStrategy,
    Strategy,
)
from .events import (
    Event,
    EventDispatcher,
    EventHandler,
    EventQueue as EventQueueV2,
    EventType as EventTypeV2,
    FillEvent as FillEventV2,
    FillStatus,
    MarketEvent,
    OrderEvent as OrderEventV2,
    OrderSide,
    OrderType as OrderTypeV2,
    SignalEvent as SignalEventV2,
    SignalType,
)
from .market_simulator import (
    AssetClassCostModel,
    LinearSlippageModel,
    MarketSimulator as MarketSimulatorV2,
    MarketState,
    SlippageModel,
    SquareRootSlippageModel,
)
from .order_management import (
    BracketOrderBuilder,
    Order as OrderV2,
    OrderManager,
    OrderStatus as OrderStatusV2,
    OrderValidator,
    PositionTracker,
    TimeInForce,
)
from .performance_analytics import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    RollingPerformanceAnalyzer,
)

__all__ = [
    # Main engine
    "BacktestEngine",
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
