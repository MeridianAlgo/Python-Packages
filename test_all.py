"""Comprehensive test script for MeridianAlgo package"""
import sys
import numpy as np
import pandas as pd

def print_header(text):
    """Print section header."""
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50)

def test_core():
    """Test core module functionality."""
    print_header("TESTING CORE FUNCTIONALITY")
    
    try:
        from meridianalgo import TimeSeriesAnalyzer, PortfolioOptimizer
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            index=dates,
            name='Close'
        )
        
        # Test TimeSeriesAnalyzer
        print("Testing TimeSeriesAnalyzer...")
        analyzer = TimeSeriesAnalyzer(prices)
        returns = analyzer.calculate_returns()
        print(f"✓ Calculated returns: {len(returns)} data points")
        
        # Test PortfolioOptimizer
        print("\nTesting PortfolioOptimizer...")
        returns_df = pd.DataFrame({
            'AAPL': returns,
            'MSFT': returns * 0.8 + np.random.normal(0, 0.01, len(returns))
        })
        optimizer = PortfolioOptimizer(returns_df)
        print(f"✓ Created PortfolioOptimizer with {len(optimizer.returns.columns)} assets")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in core functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_statistics():
    """Test statistics module functionality."""
    print_header("TESTING STATISTICS MODULE")
    
    try:
        from meridianalgo.statistics import StatisticalArbitrage
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        data = pd.DataFrame({
            'AAPL': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            'MSFT': np.cumprod(1 + np.random.normal(0.001, 0.018, 100))
        }, index=dates)
        
        # Test StatisticalArbitrage
        print("Testing StatisticalArbitrage...")
        arb = StatisticalArbitrage(data)
        corr = arb.calculate_rolling_correlation(window=10)
        print(f"✓ Calculated rolling correlation: {corr.shape} shape")
        
        # Test cointegration
        print("\nTesting cointegration...")
        x = data['AAPL']
        y = data['MSFT']
        result = arb.test_cointegration(x, y)
        print(f"✓ Cointegration test results: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in statistics module: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml():
    """Test ML module functionality."""
    print_header("TESTING ML MODULE")
    
    try:
        from meridianalgo.ml import FeatureEngineer, LSTMPredictor, prepare_data_for_lstm
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200)
        prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 200)),
            index=dates,
            name='Close'
        )
        
        # Test FeatureEngineer
        print("Testing FeatureEngineer...")
        engineer = FeatureEngineer(lookback=10)
        features = engineer.create_features(prices)
        print(f"✓ Created {len(features.columns)} features")
        
        # Test prepare_data_for_lstm
        print("\nTesting prepare_data_for_lstm...")
        target = prices.pct_change().shift(-1).dropna()
        common_idx = features.index.intersection(target.index)
        X_train, X_test, y_train, y_test = prepare_data_for_lstm(
            features.loc[common_idx], 
            target.loc[common_idx],
            sequence_length=10,
            test_size=0.2
        )
        print(f"✓ Prepared data: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in ML module: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MERIDIANALGO COMPREHENSIVE TEST")
    print("="*70)
    
    # Run tests
    core_ok = test_core()
    stats_ok = test_statistics()
    ml_ok = test_ml()
    
    # Print summary
    print_header("TEST SUMMARY")
    print(f"Core Module:    {'✓' if core_ok else '✗'}")
    print(f"Statistics Module: {'✓' if stats_ok else '✗'}")
    print(f"ML Module:      {'✓' if ml_ok else '✗'}")
    
    if all([core_ok, stats_ok, ml_ok]):
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    print("\n" + "="*70)
