"""Debug script to test MeridianAlgo package"""

def test_imports():
    """Test basic imports"""
    print("\n=== Testing Imports ===")
    try:
        import meridianalgo
        print("✓ meridianalgo imported successfully")
        print(f"Version: {meridianalgo.__version__}")
        return True
    except Exception as e:
        print(f"✗ Error importing meridianalgo: {str(e)}")
        return False

def test_core():
    """Test core functionality"""
    print("\n=== Testing Core Functionality ===")
    try:
        import numpy as np
        import pandas as pd
        from meridianalgo import PortfolioOptimizer, TimeSeriesAnalyzer
        
        # Create test data
        np.random.seed(42)
        prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
        returns = prices.pct_change().dropna()
        
        # Test PortfolioOptimizer
        print("\nTesting PortfolioOptimizer...")
        returns_df = pd.DataFrame({
            'AAPL': returns,
            'MSFT': returns * 0.8 + np.random.normal(0, 0.01, len(returns))
        })
        optimizer = PortfolioOptimizer(returns_df)
        print(f"✓ PortfolioOptimizer created with {len(optimizer.returns.columns)} assets")
        
        # Test TimeSeriesAnalyzer
        print("\nTesting TimeSeriesAnalyzer...")
        analyzer = TimeSeriesAnalyzer(prices)
        vol = analyzer.calculate_volatility(window=21)
        print(f"✓ TimeSeriesAnalyzer calculated volatility for {len(vol)} periods")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in core functionality: {str(e)}")
        return False

def test_ml():
    """Test ML module"""
    print("\n=== Testing ML Module ===")
    try:
        from meridianalgo.ml import FeatureEngineer
        import numpy as np
        import pandas as pd
        
        # Create test data
        np.random.seed(42)
        prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
        
        # Test FeatureEngineer
        print("\nTesting FeatureEngineer...")
        engineer = FeatureEngineer(lookback=10)
        features = engineer.create_features(prices)
        print(f"✓ FeatureEngineer created {len(features.columns)} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in ML module: {str(e)}")
        return False

def test_statistics():
    """Test statistics module"""
    print("\n=== Testing Statistics Module ===")
    try:
        from meridianalgo.statistics import StatisticalArbitrage
        import numpy as np
        import pandas as pd
        
        # Create test data
        np.random.seed(42)
        x = np.cumsum(np.random.normal(0, 1, 100))
        y = x + np.random.normal(0, 0.5, 100)
        df = pd.DataFrame({'X': x, 'Y': y})
        
        # Test StatisticalArbitrage
        print("\nTesting StatisticalArbitrage...")
        arb = StatisticalArbitrage(df)
        corr = arb.calculate_rolling_correlation(window=10)
        print(f"✓ Calculated rolling correlation for {len(corr)} periods")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in statistics module: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  MERIDIANALGO DEBUG TEST")
    print("="*50)
    
    # Run tests
    results = {
        'imports': test_imports(),
        'core': test_core(),
        'ml': test_ml(),
        'statistics': test_statistics()
    }
    
    # Print summary
    print("\n" + "="*50)
    print("  TEST SUMMARY")
    print("="*50)
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test.upper():<15} [{'✓' if passed else '✗'}] {status}")
    
    if all(results.values()):
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. Please check the output above for details.")
