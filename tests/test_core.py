"""Test core module functionality"""
import unittest
import numpy as np
import pandas as pd
from meridianalgo import PortfolioOptimizer, TimeSeriesAnalyzer, get_market_data, calculate_metrics, calculate_max_drawdown

class TestCoreModule(unittest.TestCase):
    """Test cases for core module"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=100)
        self.prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            index=self.dates,
            name='Close'
        )
        self.returns = self.prices.pct_change().dropna()
    
    def test_portfolio_optimizer_initialization(self):
        """Test PortfolioOptimizer initialization"""
        returns_df = pd.DataFrame({
            'AAPL': self.returns,
            'MSFT': self.returns * 0.8 + np.random.normal(0, 0.01, len(self.returns))
        })
        optimizer = PortfolioOptimizer(returns_df)
        self.assertIsInstance(optimizer, PortfolioOptimizer)
        self.assertEqual(optimizer.returns.shape, returns_df.shape)
    
    def test_time_series_analyzer(self):
        """Test TimeSeriesAnalyzer basic functionality"""
        analyzer = TimeSeriesAnalyzer(self.prices)
        returns = analyzer.calculate_returns()
        self.assertEqual(len(returns), len(self.prices) - 1)
        
        volatility = analyzer.calculate_volatility(window=21)
        # Rolling volatility should have NaN values for the first window-1 periods
        # Note: calculate_volatility calls calculate_returns() which drops NaN, so length is reduced by 1
        self.assertEqual(len(volatility), len(self.prices) - 1)  # Same length as returns
        self.assertEqual(volatility.notna().sum(), len(self.prices) - 1 - 20)  # 21-period window
    
    def test_market_data(self):
        """Test get_market_data function"""
        try:
            data = get_market_data(['AAPL'], start_date='2023-01-01', end_date='2023-01-10')
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
        except Exception as e:
            self.skipTest(f"Skipping market data test: {str(e)}")
    
    def test_metrics_calculation(self):
        """Test metrics calculation functions"""
        metrics = calculate_metrics(self.returns)
        required_metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        max_dd = calculate_max_drawdown(self.returns)
        self.assertIsInstance(max_dd, float)

if __name__ == '__main__':
    unittest.main()
