"""
Diagnostic test for MeridianAlgo package
This script helps identify and fix issues in the package.
"""

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50)

def test_environment():
    """Test Python environment and basic imports"""
    print_header("TESTING PYTHON ENVIRONMENT")
    
    # Check Python version
    import sys
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Check if we can import numpy and pandas
    try:
        import numpy as np
        import pandas as pd
        print("\n✓ Successfully imported numpy and pandas")
        print(f"numpy version: {np.__version__}")
        print(f"pandas version: {pd.__version__}")
        return True
    except ImportError as e:
        print(f"\n✗ Error importing required packages: {e}")
        return False

def test_package_import():
    """Test if the package can be imported"""
    print_header("TESTING PACKAGE IMPORT")
    
    try:
        import meridianalgo
        print(f"✓ Successfully imported meridianalgo version {meridianalgo.__version__}")
        
        # List available attributes
        print("\nAvailable attributes:")
        for attr in dir(meridianalgo):
            if not attr.startswith('_'):
                print(f"- {attr}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error importing meridianalgo: {e}")
        return False

def test_core_functionality():
    """Test core functionality"""
    print_header("TESTING CORE FUNCTIONALITY")
    
    try:
        from meridianalgo import PortfolioOptimizer, TimeSeriesAnalyzer
        import numpy as np
        import pandas as pd
        
        print("\n✓ Successfully imported core components")
        
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
        print(f"✓ Created PortfolioOptimizer with {len(optimizer.returns.columns)} assets")
        
        # Test TimeSeriesAnalyzer
        print("\nTesting TimeSeriesAnalyzer...")
        analyzer = TimeSeriesAnalyzer(prices)
        vol = analyzer.calculate_volatility(window=21)
        print(f"✓ Calculated volatility for {len(vol)} periods")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error in core functionality: {e}")
        return False

def test_ml_module():
    """Test ML module"""
    print_header("TESTING ML MODULE")
    
    try:
        from meridianalgo.ml import FeatureEngineer
        import numpy as np
        import pandas as pd
        
        print("\n✓ Successfully imported ML module")
        
        # Create test data
        np.random.seed(42)
        prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
        
        # Test FeatureEngineer
        print("\nTesting FeatureEngineer...")
        engineer = FeatureEngineer(lookback=10)
        features = engineer.create_features(prices)
        print(f"✓ Created {len(features.columns)} features")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error in ML module: {e}")
        return False

def test_statistics_module():
    """Test statistics module"""
    print_header("TESTING STATISTICS MODULE")
    
    try:
        from meridianalgo.statistics import StatisticalArbitrage
        import numpy as np
        import pandas as pd
        
        print("\n✓ Successfully imported statistics module")
        
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
        print(f"\n✗ Error in statistics module: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  MERIDIANALGO DIAGNOSTIC TOOL")
    print("="*50)
    
    # Run tests
    results = {
        'Environment': test_environment(),
        'Package Import': test_package_import(),
        'Core Functionality': test_core_functionality(),
        'ML Module': test_ml_module(),
        'Statistics Module': test_statistics_module()
    }
    
    # Print summary
    print_header("TEST SUMMARY")
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test.upper():<20} [{'✓' if passed else '✗'}] {status}")
    
    if all(results.values()):
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. Please check the output above for details.")
