"""
Technical Analysis module for MeridianAlgo.

This module provides comprehensive technical analysis capabilities including:
- TA-Lib indicator integration with pandas DataFrame interface
- Custom indicator framework with Numba JIT compilation
- Pattern recognition for candlestick and chart patterns
- Interactive visualization system
"""

from .framework import (
    CustomIndicatorFramework,
    IndicatorBuilder,
    IndicatorCompiler,
    IndicatorMetadata,
    IndicatorRegistry,
    IndicatorValidator,
    indicator,
    indicator_registry,
)
from .indicators import (
    BaseIndicator,
    CustomIndicator,
    IndicatorManager,
    TALibIndicators,
)
from .patterns import CandlestickPatterns, ChartPatterns, PatternRecognizer
from .visualization import (
    ChartAnnotationTool,
    ChartTemplate,
    InteractiveDashboard,
    TechnicalChart,
)

__all__ = [
    # Indicators
    "TALibIndicators",
    "IndicatorManager",
    "BaseIndicator",
    "CustomIndicator",
    # Patterns
    "CandlestickPatterns",
    "ChartPatterns",
    "PatternRecognizer",
    # Visualization
    "TechnicalChart",
    "InteractiveDashboard",
    "ChartTemplate",
    "ChartAnnotationTool",
    # Framework
    "IndicatorMetadata",
    "IndicatorValidator",
    "IndicatorCompiler",
    "IndicatorBuilder",
    "CustomIndicatorFramework",
    "IndicatorRegistry",
    "indicator_registry",
    "indicator",
]
