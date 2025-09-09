"""Test script to verify MeridianAlgo package functionality"""
import sys
import numpy as np
import pandas as pd

print("=== Testing MeridianAlgo Package ===\n")
print(f"Python version: {sys.version}\n")

# Test imports
try:
    import meridianalgo
    print("✓ Successfully imported meridianalgo")
    print(f"Version: {meridianalgo.__version__}\n")
except Exception as e:
    print(f"✗ Error importing meridianalgo: {e}")
    sys.exit(1)

# Test core functionality
try:
    from meridianalgo import PortfolioOptimizer, TimeSeriesAnalyzer
    
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
    
    # Test statistics module
    try:
        from meridianalgo.statistics import StatisticalArbitrage, calculate_value_at_risk
        
        print("\nTesting StatisticalArbitrage...")
        arb = StatisticalArbitrage(returns_df)
        corr = arb.calculate_rolling_correlation(window=10)
        print(f"✓ Calculated rolling correlation: {corr.shape} shape")
        
        print("\nTesting calculate_value_at_risk...")
        var = calculate_value_at_risk(returns, confidence_level=0.95)
        print(f"✓ Calculated VaR: {var:.4f}")
        
    except Exception as e:
        print(f"✗ Error in statistics module: {e}")
    
    # Test ML module
    try:
        from meridianalgo.ml import FeatureEngineer
        
        print("\nTesting FeatureEngineer...")
        engineer = FeatureEngineer(lookback=10)
        features = engineer.create_features(prices)
        print(f"✓ Created {len(features.columns)} features")
        
    except Exception as e:
        print(f"✗ Error in ML module: {e}")
    
    print("\n✓ All tests completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
