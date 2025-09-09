"""
Performance Analysis Module for Portfolio Management

This module provides performance analysis tools for portfolios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

class PerformanceAnalyzer:
    """Portfolio Performance Analyzer."""
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'total_return': (1 + self.portfolio_returns).prod() - 1,
            'annualized_return': (1 + self.portfolio_returns).prod() ** (252 / len(self.portfolio_returns)) - 1,
            'volatility': self.portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': self.portfolio_returns.mean() / self.portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown()
        }
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min()

class AttributionAnalysis:
    """Portfolio Attribution Analysis."""
    
    def __init__(self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame):
        self.portfolio_returns = portfolio_returns
        self.factor_returns = factor_returns
    
    def analyze_attribution(self) -> Dict[str, float]:
        """Analyze portfolio attribution."""
        # Simplified attribution analysis
        return {
            'factor_contribution': self.factor_returns.mean().sum(),
            'residual_return': self.portfolio_returns.mean() - self.factor_returns.mean().sum()
        }

class BenchmarkComparison:
    """Benchmark Comparison Analysis."""
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
    
    def compare_performance(self) -> Dict[str, float]:
        """Compare portfolio vs benchmark performance."""
        portfolio_metrics = PerformanceAnalyzer(self.portfolio_returns).calculate_metrics()
        benchmark_metrics = PerformanceAnalyzer(self.benchmark_returns).calculate_metrics()
        
        return {
            'excess_return': portfolio_metrics['annualized_return'] - benchmark_metrics['annualized_return'],
            'tracking_error': (self.portfolio_returns - self.benchmark_returns).std() * np.sqrt(252),
            'information_ratio': (portfolio_metrics['annualized_return'] - benchmark_metrics['annualized_return']) / 
                               ((self.portfolio_returns - self.benchmark_returns).std() * np.sqrt(252))
        }
