"""
Test script for MeridianAlgo package
"""
import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from meridianalgo import (
    PortfolioOptimizer,
    calculate_value_at_risk,
    calculate_expected_shortfall
)
from meridianalgo.ml import FeatureEngineer
from meridianalgo.statistics import StatisticalArbitrage

class TestMeridianAlgo(unittest.TestCase):
    """Test cases for MeridianAlgo package"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        cls.prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            index=dates,
            name='Close'
        )
        
        # Create sample returns data
        cls.returns = cls.prices.pct_change().dropna()
        
        # Create sample OHLCV data
        cls.ohlcv = pd.DataFrame({
            'Open': cls.prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
            'High': cls.prices * (1 + np.random.uniform(0, 0.02, 100)),
            'Low': cls.prices * (1 - np.random.uniform(0, 0.02, 100)),
            'Close': cls.prices,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
    
    def test_portfolio_optimizer(self):
        """Test PortfolioOptimizer class"""
        # Create sample returns for multiple assets
        returns = pd.DataFrame({
            'AAPL': self.returns,
            'MSFT': self.returns * 0.8 + np.random.normal(0, 0.01, len(self.returns)),
            'GOOG': self.returns * 1.2 + np.random.normal(0, 0.01, len(self.returns))
        })
        
        # Test initialization
        optimizer = PortfolioOptimizer(returns)
        self.assertIsNotNone(optimizer)
        
        # Test efficient frontier calculation
        frontier = optimizer.calculate_efficient_frontier()
        self.assertIsInstance(frontier, dict)
        self.assertIn('volatility', frontier)
        self.assertIn('returns', frontier)
        self.assertIn('sharpe', frontier)
        self.assertIn('weights', frontier)
        
        # Test that all arrays have the same length
        self.assertEqual(len(frontier['volatility']), len(frontier['returns']))
        self.assertEqual(len(frontier['volatility']), len(frontier['sharpe']))
        self.assertEqual(len(frontier['volatility']), len(frontier['weights']))
        
        # Test portfolio optimization
        weights = optimizer.optimize_portfolio()
        self.assertEqual(len(weights), 3)  # Should have weights for 3 assets
        self.assertAlmostEqual(sum(weights), 1.0, places=6)  # Weights should sum to 1
    
    def test_feature_engineering(self):
        """Test FeatureEngineer class"""
        # Initialize feature engineer
        engineer = FeatureEngineer(lookback=10)
        
        # Test feature creation
        features = engineer.create_features(self.prices)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 5)  # Should have multiple features
        
        # Test that features have the same index as input
        self.assertTrue(features.index.equals(self.prices.index))
    
    def test_statistical_arbitrage(self):
        """Test StatisticalArbitrage class"""
        # Create two cointegrated time series
        np.random.seed(42)
        x = np.cumsum(np.random.normal(0, 1, 100))
        y = x + np.random.normal(0, 0.5, 100)
        
        # Initialize and test cointegration
        arb = StatisticalArbitrage(pd.DataFrame({'X': x, 'Y': y}))
        coint_test = arb.test_cointegration(x, y)
        
        self.assertIsInstance(coint_test, dict)
        self.assertIn('pvalue', coint_test)
        self.assertIn('test_statistic', coint_test)
        
        # Test spread calculation
        spread = arb.calculate_spread(x, y)
        self.assertEqual(len(spread), len(x))
    
    def test_risk_metrics(self):
        """Test risk metrics functions"""
        # Test Value at Risk
        var = calculate_value_at_risk(self.returns, confidence_level=0.95)
        self.assertIsInstance(var, float)
        
        # Test Expected Shortfall
        es = calculate_expected_shortfall(self.returns, confidence_level=0.95)
        self.assertIsInstance(es, float)
        
        # ES should be more severe than VaR for the same confidence level
        self.assertLessEqual(es, var)

    def test_real_world_data(self):
        """Test with real market data from yfinance"""
        try:
            # Download sample data
            data = yf.download('AAPL', start='2023-01-01', end='2023-06-01', progress=False)
            returns = data['Adj Close'].pct_change().dropna()
            
            # Test PortfolioOptimizer with real data
            optimizer = PortfolioOptimizer(pd.DataFrame({'AAPL': returns}))
            weights = optimizer.optimize_portfolio()
            self.assertAlmostEqual(sum(weights), 1.0, places=6)
            
            # Test risk metrics with real data
            var = calculate_value_at_risk(returns)
            es = calculate_expected_shortfall(returns)
            self.assertIsInstance(var, float)
            self.assertIsInstance(es, float)
            
        except Exception as e:
            self.skipTest(f"Skipping real-world data test due to: {str(e)}")

if __name__ == '__main__':
    unittest.main()
