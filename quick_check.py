"""Quick test for MeridianAlgo package"""
import numpy as np
import pandas as pd

print("Testing MeridianAlgo package...")

try:
    # Test core functionality
    from meridianalgo import TimeSeriesAnalyzer
    
    # Create test data
    np.random.seed(42)
    prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
    
    # Basic analysis
    analyzer = TimeSeriesAnalyzer(prices)
    returns = analyzer.calculate_returns()
    print(f"✓ Basic analysis: Calculated {len(returns)} returns")
    
    # Test feature engineering
    from meridianalgo import FeatureEngineer
    engineer = FeatureEngineer()
    features = engineer.create_features(prices)
    print(f"✓ Feature engineering: Created {len(features.columns)} features")
    
    print("\n✅ Basic functionality tests passed!")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
