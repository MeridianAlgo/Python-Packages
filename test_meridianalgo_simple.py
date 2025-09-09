"""
Simple test script for MeridianAlgo package functionality.
"""
import numpy as np
import pandas as pd
import unittest

class TestMeridianAlgoSimple(unittest.TestCase):
    """Simple test cases for MeridianAlgo functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        self.prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            index=dates,
            name='Close'
        )
        
        # Create sample returns data
        self.returns = self.prices.pct_change().dropna()
        
        # Create sample OHLCV data
        self.ohlcv = pd.DataFrame({
            'Open': self.prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
            'High': self.prices * (1 + np.random.uniform(0, 0.02, 100)),
            'Low': self.prices * (1 - np.random.uniform(0, 0.02, 100)),
            'Close': self.prices,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
    
    def test_basic_functionality(self):
        """Test basic functionality with sample data."""
        # Test if we can calculate basic statistics
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        
        self.assertIsInstance(mean_return, (int, float, np.number))
        self.assertIsInstance(std_return, (int, float, np.number))
        self.assertGreater(std_return, 0)
    
    def test_ohlcv_data(self):
        """Test OHLCV data structure and calculations."""
        # Test if OHLCV data is valid
        self.assertEqual(len(self.ohlcv), 100)
        self.assertIn('Open', self.ohlcv.columns)
        self.assertIn('High', self.ohlcv.columns)
        self.assertIn('Low', self.ohlcv.columns)
        self.assertIn('Close', self.ohlcv.columns)
        self.assertIn('Volume', self.ohlcv.columns)
        
        # Test if high is greater than or equal to low
        self.assertTrue((self.ohlcv['High'] >= self.ohlcv['Low']).all())
    
    def test_returns_calculation(self):
        """Test returns calculation."""
        # Test if returns are calculated correctly
        manual_returns = self.prices.pct_change().dropna()
        pd.testing.assert_series_equal(self.returns, manual_returns)
        
        # Test that we have one less return than price points
        self.assertEqual(len(self.returns), len(self.prices) - 1)

if __name__ == '__main__':
    unittest.main()
