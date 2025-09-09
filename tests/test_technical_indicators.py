"""
Unit tests for technical indicators module.
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for technical indicators functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create sample OHLCV data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = base_price * np.cumprod(1 + returns)
        
        self.data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, 100)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=self.dates)
        
        # Ensure High >= Low and High >= Close >= Low
        self.data['High'] = np.maximum(self.data['High'], self.data['Close'])
        self.data['Low'] = np.minimum(self.data['Low'], self.data['Close'])
        self.data['High'] = np.maximum(self.data['High'], self.data['Open'])
        self.data['Low'] = np.minimum(self.data['Low'], self.data['Open'])
    
    def test_momentum_indicators(self):
        """Test momentum indicators."""
        from meridianalgo import RSI, Stochastic, WilliamsR, ROC, Momentum
        
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        
        # Test RSI
        rsi = RSI(close, period=14)
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(close))
        self.assertTrue((rsi.dropna() >= 0).all())
        self.assertTrue((rsi.dropna() <= 100).all())
        
        # Test Stochastic
        stoch_k, stoch_d = Stochastic(high, low, close, k_period=14, d_period=3)
        self.assertIsInstance(stoch_k, pd.Series)
        self.assertIsInstance(stoch_d, pd.Series)
        self.assertEqual(len(stoch_k), len(close))
        self.assertTrue((stoch_k >= 0).all())
        self.assertTrue((stoch_k <= 100).all())
        
        # Test Williams %R
        wr = WilliamsR(high, low, close, period=14)
        self.assertIsInstance(wr, pd.Series)
        self.assertEqual(len(wr), len(close))
        self.assertTrue((wr <= 0).all())
        self.assertTrue((wr >= -100).all())
        
        # Test ROC
        roc = ROC(close, period=12)
        self.assertIsInstance(roc, pd.Series)
        self.assertEqual(len(roc), len(close))
        
        # Test Momentum
        momentum = Momentum(close, period=10)
        self.assertIsInstance(momentum, pd.Series)
        self.assertEqual(len(momentum), len(close))
    
    def test_trend_indicators(self):
        """Test trend indicators."""
        from meridianalgo import SMA, EMA, MACD, ADX, Aroon, ParabolicSAR, Ichimoku
        
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        
        # Test SMA
        sma = SMA(close, period=20)
        self.assertIsInstance(sma, pd.Series)
        self.assertEqual(len(sma), len(close))
        
        # Test EMA
        ema = EMA(close, period=20)
        self.assertIsInstance(ema, pd.Series)
        self.assertEqual(len(ema), len(close))
        
        # Test MACD
        macd_line, signal_line, histogram = MACD(close, fast=12, slow=26, signal=9)
        self.assertIsInstance(macd_line, pd.Series)
        self.assertIsInstance(signal_line, pd.Series)
        self.assertIsInstance(histogram, pd.Series)
        self.assertEqual(len(macd_line), len(close))
        
        # Test ADX
        adx = ADX(high, low, close, period=14)
        self.assertIsInstance(adx, pd.Series)
        self.assertEqual(len(adx), len(close))
        self.assertTrue((adx.dropna() >= 0).all())
        self.assertTrue((adx.dropna() <= 100).all())
        
        # Test Aroon
        aroon_up, aroon_down = Aroon(high, low, period=25)
        self.assertIsInstance(aroon_up, pd.Series)
        self.assertIsInstance(aroon_down, pd.Series)
        self.assertEqual(len(aroon_up), len(close))
        self.assertTrue((aroon_up >= 0).all())
        self.assertTrue((aroon_up <= 100).all())
        
        # Test Parabolic SAR
        psar = ParabolicSAR(high, low, close)
        self.assertIsInstance(psar, pd.Series)
        self.assertEqual(len(psar), len(close))
        
        # Test Ichimoku
        ichimoku = Ichimoku(high, low, close)
        self.assertIsInstance(ichimoku, dict)
        self.assertIn('tenkan_sen', ichimoku)
        self.assertIn('kijun_sen', ichimoku)
        self.assertIn('senkou_span_a', ichimoku)
        self.assertIn('senkou_span_b', ichimoku)
        self.assertIn('chikou_span', ichimoku)
    
    def test_volatility_indicators(self):
        """Test volatility indicators."""
        from meridianalgo import BollingerBands, ATR, KeltnerChannels, DonchianChannels
        
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        
        # Test Bollinger Bands
        bb_upper, bb_middle, bb_lower = BollingerBands(close, period=20, std_dev=2)
        self.assertIsInstance(bb_upper, pd.Series)
        self.assertIsInstance(bb_middle, pd.Series)
        self.assertIsInstance(bb_lower, pd.Series)
        self.assertEqual(len(bb_upper), len(close))
        self.assertTrue((bb_upper.dropna() >= bb_middle.dropna()).all())
        self.assertTrue((bb_middle.dropna() >= bb_lower.dropna()).all())
        
        # Test ATR
        atr = ATR(high, low, close, period=14)
        self.assertIsInstance(atr, pd.Series)
        self.assertEqual(len(atr), len(close))
        self.assertTrue((atr >= 0).all())
        
        # Test Keltner Channels
        kc_upper, kc_middle, kc_lower = KeltnerChannels(high, low, close, period=20, multiplier=2)
        self.assertIsInstance(kc_upper, pd.Series)
        self.assertIsInstance(kc_middle, pd.Series)
        self.assertIsInstance(kc_lower, pd.Series)
        self.assertEqual(len(kc_upper), len(close))
        
        # Test Donchian Channels
        dc_upper, dc_middle, dc_lower = DonchianChannels(high, low, period=20)
        self.assertIsInstance(dc_upper, pd.Series)
        self.assertIsInstance(dc_middle, pd.Series)
        self.assertIsInstance(dc_lower, pd.Series)
        self.assertEqual(len(dc_upper), len(close))
    
    def test_volume_indicators(self):
        """Test volume indicators."""
        from meridianalgo import OBV, ADLine, ChaikinOscillator, MoneyFlowIndex, EaseOfMovement
        
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        volume = self.data['Volume']
        
        # Test OBV
        obv = OBV(close, volume)
        self.assertIsInstance(obv, pd.Series)
        self.assertEqual(len(obv), len(close))
        
        # Test AD Line
        ad_line = ADLine(high, low, close, volume)
        self.assertIsInstance(ad_line, pd.Series)
        self.assertEqual(len(ad_line), len(close))
        
        # Test Chaikin Oscillator
        chaikin = ChaikinOscillator(high, low, close, volume, fast=3, slow=10)
        self.assertIsInstance(chaikin, pd.Series)
        self.assertEqual(len(chaikin), len(close))
        
        # Test Money Flow Index
        mfi = MoneyFlowIndex(high, low, close, volume, period=14)
        self.assertIsInstance(mfi, pd.Series)
        self.assertEqual(len(mfi), len(close))
        self.assertTrue((mfi.dropna() >= 0).all())
        self.assertTrue((mfi.dropna() <= 100).all())
        
        # Test Ease of Movement
        eom = EaseOfMovement(high, low, volume, period=14)
        self.assertIsInstance(eom, pd.Series)
        self.assertEqual(len(eom), len(close))
    
    def test_overlay_indicators(self):
        """Test overlay indicators."""
        from meridianalgo import PivotPoints, FibonacciRetracement, SupportResistance
        
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        
        # Test Pivot Points
        pivot_data = PivotPoints(high, low, close)
        self.assertIsInstance(pivot_data, dict)
        self.assertIn('pivot', pivot_data)
        self.assertIn('r1', pivot_data)
        self.assertIn('s1', pivot_data)
        
        # Test Fibonacci Retracement
        fib_data = FibonacciRetracement(high, low)
        self.assertIsInstance(fib_data, dict)
        self.assertIn('fib_0.236', fib_data)
        self.assertIn('fib_0.618', fib_data)
        
        # Test Support and Resistance
        sr_data = SupportResistance(close, window=20, min_touches=2)
        self.assertIsInstance(sr_data, dict)
        self.assertIn('resistance', sr_data)
        self.assertIn('support', sr_data)


if __name__ == '__main__':
    unittest.main()
