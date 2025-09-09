print("Testing basic imports...")

try:
    from meridianalgo import TimeSeriesAnalyzer, FeatureEngineer
    import numpy as np
    
    print("✓ Basic imports successful!")
    
    # Test TimeSeriesAnalyzer
    prices = np.random.rand(100).cumsum() + 100
    analyzer = TimeSeriesAnalyzer(prices)
    returns = analyzer.calculate_returns()
    print(f"✓ TimeSeriesAnalyzer: Calculated {len(returns)} returns")
    
    # Test FeatureEngineer
    engineer = FeatureEngineer()
    features = engineer.create_features(prices)
    print(f"✓ FeatureEngineer: Created {len(features.columns)} features")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
