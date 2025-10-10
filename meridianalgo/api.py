"""
Unified API for MeridianAlgo - Ultimate Quantitative Development Platform.

This module provides a consistent, high-level interface to all MeridianAlgo functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import warnings

# Core imports
from .core import PortfolioOptimizer, TimeSeriesAnalyzer
from .statistics import StatisticalArbitrage, calculate_metrics

# Data infrastructure
try:
    from .data.providers import YahooFinanceProvider, AlphaVantageProvider
    from .data.processing import DataPipeline, DataValidator
    from .data.streaming import RealTimeDataStream
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False

# Technical analysis
try:
    from .technical_analysis.indicators import RSI, MACD, BollingerBands
    from .technical_analysis.patterns import CandlestickPatterns
    from .technical_analysis.framework import BaseIndicator
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False

# Portfolio management
try:
    from .portfolio.optimization import PortfolioOptimizer as AdvancedPortfolioOptimizer
    from .portfolio.risk_management import RiskManager
    from .portfolio.performance import PerformanceAnalyzer
    from .portfolio.transaction_costs import TransactionCostOptimizer
    from .portfolio.rebalancing import CalendarRebalancer, ThresholdRebalancer
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False

# Backtesting
try:
    from .backtesting.backtester import EventDrivenBacktester
    from .backtesting.performance_analytics import PerformanceAnalyzer as BacktestAnalyzer
    from .backtesting.order_management import OrderManager
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

# Machine Learning
try:
    from .machine_learning.models import ModelFactory, LSTMPredictor
    from .machine_learning.feature_engineering import FinancialFeatureEngineer
    from .machine_learning.validation import WalkForwardValidator
    from .machine_learning.deployment import ModelDeploymentPipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Fixed Income
try:
    from .fixed_income.bonds import BondPricer, YieldCurve
    from .fixed_income.options import BlackScholesModel, BinomialTreeModel
    FIXED_INCOME_AVAILABLE = True
except ImportError:
    FIXED_INCOME_AVAILABLE = False

# Risk Analysis
try:
    from .risk_analysis.var_es import VaRCalculator
    from .risk_analysis.stress_testing import StressTester
    from .risk_analysis.real_time_monitor import RealTimeRiskMonitor
    RISK_ANALYSIS_AVAILABLE = True
except ImportError:
    RISK_ANALYSIS_AVAILABLE = False

# High-Performance Computing
try:
    from .computing.distributed import HighPerformanceComputing
    HPC_AVAILABLE = True
except ImportError:
    HPC_AVAILABLE = False

logger = logging.getLogger(__name__)

class MeridianAlgoAPI:
    """
    Unified API for MeridianAlgo platform.
    
    This class provides a single entry point to all MeridianAlgo functionality
    with consistent interfaces and intelligent defaults.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MeridianAlgo API.
        
        Args:
            config: Configuration dictionary for various components
        """
        self.config = config or {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize available components."""
        
        # Data components
        if DATA_AVAILABLE:
            self.data_provider = YahooFinanceProvider()  # Default provider
            self.data_pipeline = DataPipeline()
        
        # Portfolio components
        if PORTFOLIO_AVAILABLE:
            self.portfolio_optimizer = AdvancedPortfolioOptimizer()
            self.risk_manager = RiskManager()
            self.performance_analyzer = PerformanceAnalyzer()
        
        # ML components
        if ML_AVAILABLE:
            self.feature_engineer = FinancialFeatureEngineer()
            self.model_deployment = ModelDeploymentPipeline()
        
        # Risk monitoring
        if RISK_ANALYSIS_AVAILABLE:
            self.risk_monitor = RealTimeRiskMonitor()
        
        # HPC
        if HPC_AVAILABLE:
            self.hpc = HighPerformanceComputing(self.config.get('hpc', {}))
    
    # ==================== DATA METHODS ====================
    
    def get_market_data(self, symbols: Union[str, List[str]], 
                       start_date: Union[str, datetime] = None,
                       end_date: Union[str, datetime] = None,
                       provider: str = "yahoo") -> pd.DataFrame:
        """
        Get market data for symbols.
        
        Args:
            symbols: Symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            provider: Data provider to use
            
        Returns:
            DataFrame with market data
        """
        if not DATA_AVAILABLE:
            raise ImportError("Data module not available")
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        
        if end_date is None:
            end_date = datetime.now()
        
        return self.data_provider.get_historical_data(symbols, start_date, end_date)
    
    def clean_data(self, data: pd.DataFrame, 
                   methods: List[str] = None) -> pd.DataFrame:
        """
        Clean financial data using processing pipeline.
        
        Args:
            data: Raw data to clean
            methods: List of cleaning methods to apply
            
        Returns:
            Cleaned data
        """
        if not DATA_AVAILABLE:
            return data
        
        return self.data_pipeline.fit_transform(data)
    
    # ==================== TECHNICAL ANALYSIS METHODS ====================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        if not TECHNICAL_ANALYSIS_AVAILABLE:
            # Fallback implementation
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        rsi_indicator = RSI(period=period)
        return rsi_indicator.calculate(prices)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        if not TECHNICAL_ANALYSIS_AVAILABLE:
            # Fallback implementation
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        macd_indicator = MACD(fast=fast, slow=slow, signal=signal)
        return macd_indicator.calculate(prices)
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        if not TECHNICAL_ANALYSIS_AVAILABLE:
            # Fallback implementation
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        
        bb_indicator = BollingerBands(period=period, std_dev=std_dev)
        return bb_indicator.calculate(prices)
    
    def detect_patterns(self, ohlc_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Detect candlestick patterns."""
        if not TECHNICAL_ANALYSIS_AVAILABLE:
            return {}
        
        pattern_detector = CandlestickPatterns()
        return pattern_detector.detect_all_patterns(ohlc_data)
    
    # ==================== PORTFOLIO METHODS ====================
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          method: str = "mean_variance",
                          **kwargs) -> Dict[str, float]:
        """
        Optimize portfolio weights.
        
        Args:
            returns: Historical returns data
            method: Optimization method
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of optimal weights
        """
        if PORTFOLIO_AVAILABLE:
            return self.portfolio_optimizer.optimize(returns, method=method, **kwargs)
        else:
            # Fallback to basic optimizer
            optimizer = PortfolioOptimizer(returns)
            return optimizer.optimize_portfolio(**kwargs)
    
    def calculate_risk_metrics(self, returns: pd.Series, 
                             confidence_levels: List[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Return series
            confidence_levels: VaR confidence levels
            
        Returns:
            Dictionary of risk metrics
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        
        metrics = {}
        
        if RISK_ANALYSIS_AVAILABLE:
            var_calc = VaRCalculator()
            for cl in confidence_levels:
                metrics[f'var_{int(cl*100)}'] = var_calc.calculate_var(returns, cl)
                metrics[f'es_{int(cl*100)}'] = var_calc.calculate_expected_shortfall(returns, cl)
        
        # Basic metrics
        metrics.update({
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        })
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def analyze_performance(self, returns: pd.Series, 
                           benchmark: pd.Series = None) -> Dict[str, Any]:
        """
        Analyze portfolio performance.
        
        Args:
            returns: Portfolio returns
            benchmark: Benchmark returns for comparison
            
        Returns:
            Performance analysis results
        """
        if PORTFOLIO_AVAILABLE:
            return self.performance_analyzer.analyze_returns(returns, benchmark)
        else:
            return calculate_metrics(returns)
    
    # ==================== BACKTESTING METHODS ====================
    
    def run_backtest(self, strategy, data: pd.DataFrame, 
                    initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run strategy backtest.
        
        Args:
            strategy: Trading strategy
            data: Historical data
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        if not BACKTESTING_AVAILABLE:
            raise ImportError("Backtesting module not available")
        
        backtester = EventDrivenBacktester(initial_capital=initial_capital)
        backtester.set_strategy(strategy)
        return backtester.run_backtest(data)
    
    # ==================== MACHINE LEARNING METHODS ====================
    
    def create_features(self, price_data: pd.DataFrame, 
                       feature_types: List[str] = None) -> pd.DataFrame:
        """
        Create financial features for ML.
        
        Args:
            price_data: OHLCV price data
            feature_types: Types of features to create
            
        Returns:
            Feature matrix
        """
        if not ML_AVAILABLE:
            # Basic feature creation
            features = pd.DataFrame(index=price_data.index)
            features['returns'] = price_data['Close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['sma_20'] = price_data['Close'].rolling(20).mean()
            return features.dropna()
        
        return self.feature_engineer.create_features(price_data, feature_types)
    
    def train_model(self, features: pd.DataFrame, targets: pd.Series,
                   model_type: str = "random_forest", **kwargs) -> Any:
        """
        Train ML model.
        
        Args:
            features: Feature matrix
            targets: Target values
            model_type: Type of model to train
            **kwargs: Model parameters
            
        Returns:
            Trained model
        """
        if not ML_AVAILABLE:
            raise ImportError("Machine learning module not available")
        
        model = ModelFactory.create_model(model_type, **kwargs)
        model.fit(features, targets)
        return model
    
    def deploy_model(self, model, model_name: str, 
                    performance_metrics: Dict[str, float]) -> str:
        """
        Deploy model to production.
        
        Args:
            model: Trained model
            model_name: Name for the model
            performance_metrics: Model performance metrics
            
        Returns:
            Model version ID
        """
        if not ML_AVAILABLE:
            raise ImportError("Machine learning module not available")
        
        return self.model_deployment.deploy_model(
            model_name, model, performance_metrics
        )
    
    # ==================== FIXED INCOME METHODS ====================
    
    def price_bond(self, face_value: float, coupon_rate: float, 
                  maturity_years: float, yield_rate: float) -> Dict[str, float]:
        """
        Price a bond.
        
        Args:
            face_value: Face value of bond
            coupon_rate: Annual coupon rate
            maturity_years: Years to maturity
            yield_rate: Yield to maturity
            
        Returns:
            Bond pricing results
        """
        if not FIXED_INCOME_AVAILABLE:
            # Simple bond pricing formula
            periods = int(maturity_years * 2)  # Semi-annual
            coupon = coupon_rate * face_value / 2
            
            pv_coupons = sum(coupon / (1 + yield_rate/2)**i for i in range(1, periods + 1))
            pv_principal = face_value / (1 + yield_rate/2)**periods
            
            return {'price': pv_coupons + pv_principal}
        
        from .fixed_income.bonds import Bond
        bond = Bond(
            face_value=face_value,
            coupon_rate=coupon_rate,
            maturity_date=datetime.now() + timedelta(days=int(maturity_years * 365))
        )
        
        return BondPricer.price_bond(bond, yield_rate)
    
    def price_option(self, spot: float, strike: float, expiry: float,
                    risk_free_rate: float, volatility: float, 
                    option_type: str = "call") -> Dict[str, Any]:
        """
        Price an option using Black-Scholes.
        
        Args:
            spot: Current underlying price
            strike: Strike price
            expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            volatility: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option pricing results
        """
        if not FIXED_INCOME_AVAILABLE:
            # Simplified Black-Scholes
            from scipy.stats import norm
            
            d1 = (np.log(spot/strike) + (risk_free_rate + 0.5*volatility**2)*expiry) / (volatility*np.sqrt(expiry))
            d2 = d1 - volatility*np.sqrt(expiry)
            
            if option_type == "call":
                price = spot*norm.cdf(d1) - strike*np.exp(-risk_free_rate*expiry)*norm.cdf(d2)
            else:
                price = strike*np.exp(-risk_free_rate*expiry)*norm.cdf(-d2) - spot*norm.cdf(-d1)
            
            return {'price': price}
        
        from .fixed_income.options import Option
        option = Option(
            underlying_price=spot,
            strike_price=strike,
            time_to_expiry=expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type
        )
        
        return BlackScholesModel.price_option(option)
    
    # ==================== RISK MONITORING METHODS ====================
    
    def start_risk_monitoring(self, portfolio_positions: List[Dict[str, Any]],
                            risk_limits: List[Dict[str, Any]] = None) -> None:
        """
        Start real-time risk monitoring.
        
        Args:
            portfolio_positions: List of portfolio positions
            risk_limits: List of risk limits to monitor
        """
        if not RISK_ANALYSIS_AVAILABLE:
            logger.warning("Risk monitoring not available")
            return
        
        # Add positions to monitor
        for pos_data in portfolio_positions:
            from .risk_analysis.real_time_monitor import PortfolioPosition
            position = PortfolioPosition(**pos_data)
            self.risk_monitor.update_position(position)
        
        # Add risk limits
        if risk_limits:
            for limit_data in risk_limits:
                from .risk_analysis.real_time_monitor import RiskLimit
                limit = RiskLimit(**limit_data)
                self.risk_monitor.add_risk_limit(limit)
        
        # Start monitoring
        self.risk_monitor.start_monitoring()
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get risk monitoring dashboard data."""
        if not RISK_ANALYSIS_AVAILABLE:
            return {}
        
        return self.risk_monitor.get_risk_dashboard_data()
    
    # ==================== UTILITY METHODS ====================
    
    def get_available_modules(self) -> Dict[str, bool]:
        """Get status of available modules."""
        return {
            'data': DATA_AVAILABLE,
            'technical_analysis': TECHNICAL_ANALYSIS_AVAILABLE,
            'portfolio': PORTFOLIO_AVAILABLE,
            'backtesting': BACKTESTING_AVAILABLE,
            'machine_learning': ML_AVAILABLE,
            'fixed_income': FIXED_INCOME_AVAILABLE,
            'risk_analysis': RISK_ANALYSIS_AVAILABLE,
            'hpc': HPC_AVAILABLE
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and performance stats."""
        info = {
            'version': '4.0.0',
            'available_modules': self.get_available_modules(),
            'config': self.config
        }
        
        if HPC_AVAILABLE and hasattr(self, 'hpc'):
            info['hpc_stats'] = self.hpc.get_performance_stats()
        
        return info
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'risk_monitor'):
            self.risk_monitor.stop_monitoring()
        
        if hasattr(self, 'hpc'):
            self.hpc.cleanup()
        
        logger.info("MeridianAlgo API cleaned up")

# Global API instance
_api_instance = None

def get_api(config: Dict[str, Any] = None) -> MeridianAlgoAPI:
    """Get global API instance."""
    global _api_instance
    
    if _api_instance is None:
        _api_instance = MeridianAlgoAPI(config)
    
    return _api_instance

# Convenience functions for direct access
def get_market_data(symbols: Union[str, List[str]], 
                   start_date: Union[str, datetime] = None,
                   end_date: Union[str, datetime] = None) -> pd.DataFrame:
    """Get market data."""
    return get_api().get_market_data(symbols, start_date, end_date)

def optimize_portfolio(returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
    """Optimize portfolio."""
    return get_api().optimize_portfolio(returns, **kwargs)

def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate risk metrics."""
    return get_api().calculate_risk_metrics(returns)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    return get_api().calculate_rsi(prices, period)

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                  signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD."""
    return get_api().calculate_macd(prices, fast, slow, signal)

def price_option(spot: float, strike: float, expiry: float,
                risk_free_rate: float, volatility: float, 
                option_type: str = "call") -> Dict[str, Any]:
    """Price option."""
    return get_api().price_option(spot, strike, expiry, risk_free_rate, volatility, option_type)

# Export main classes and functions
__all__ = [
    'MeridianAlgoAPI',
    'get_api',
    'get_market_data',
    'optimize_portfolio', 
    'calculate_risk_metrics',
    'calculate_rsi',
    'calculate_macd',
    'price_option'
]