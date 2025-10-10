"""
Production-ready backtesting engine for MeridianAlgo.

This module provides comprehensive backtesting capabilities including:
- Event-driven backtesting architecture with realistic market simulation
- Comprehensive order management system with all order types
- Parallel processing and GPU acceleration for large-scale testing
- Detailed performance analytics and risk reporting
"""

from .events import (
    Event,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent
)

from .backtester import (
    EventDrivenBacktester,
    BacktestResult
)

try:
    from .order_management import (
        OrderManager,
        Order,
        OrderType,
        OrderStatus
    )
    ORDER_MANAGEMENT_AVAILABLE = True
except ImportError:
    ORDER_MANAGEMENT_AVAILABLE = False

try:
    from .market_simulator import (
        MarketSimulator,
        SlippageModel,
        LinearSlippageModel,
        SquareRootSlippageModel
    )
    MARKET_SIMULATOR_AVAILABLE = True
except ImportError:
    MARKET_SIMULATOR_AVAILABLE = False

__all__ = [
    # Events
    'Event',
    'MarketEvent', 
    'SignalEvent',
    'OrderEvent',
    'FillEvent',
    
    # Backtester
    'EventDrivenBacktester',
    'BacktestResult'
]

# Add available modules to __all__
if ORDER_MANAGEMENT_AVAILABLE:
    __all__.extend(['OrderManager', 'Order', 'OrderType', 'OrderStatus'])

if MARKET_SIMULATOR_AVAILABLE:
    __all__.extend(['MarketSimulator', 'SlippageModel', 'LinearSlippageModel', 'SquareRootSlippageModel'])