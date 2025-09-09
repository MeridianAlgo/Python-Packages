"""Quick test script to verify core functionality"""
import sys
import numpy as np
import pandas as pd

print("=== MeridianAlgo Quick Test ===\n")

# Test 1: Basic imports
try:
    from meridianalgo import __version__
    print(f"✓ Package version: {__version__}")
except Exception as e:
    print(f"✗ Error importing package: {e}")
    sys.exit(1)

# Test 2: Core functionality
try:
    from meridianalgo import TimeSeriesAnalyzer
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100)
    prices = pd.Series(
        np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
        index=dates,
        name='Close'
    )
    
    # Test TimeSeriesAnalyzer
    print("\nTesting TimeSeriesAnalyzer...")
    analyzer = TimeSeriesAnalyzer(prices)
    returns = analyzer.calculate_returns()
    print(f"✓ Calculated returns: {len(returns)} data points")
    
    # Test basic statistics
    print(f"✓ Mean return: {returns.mean():.4f}")
    print(f"✓ Volatility: {returns.std():.4f}")
    
except Exception as e:
    print(f"\n✗ Error in core functionality: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests completed successfully!")
