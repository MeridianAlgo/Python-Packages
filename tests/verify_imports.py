print("Testing MeridianAlgo imports...")

# Test core imports
try:
    from meridianalgo import TimeSeriesAnalyzer, PortfolioOptimizer
    print("✓ Core modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing core modules: {e}")

# Test statistics imports
try:
    from meridianalgo import StatisticalArbitrage, calculate_value_at_risk
    print("✓ Statistics modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing statistics modules: {e}")

# Test ML imports
try:
    from meridianalgo import FeatureEngineer, LSTMPredictor
    print("✓ ML modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing ML modules: {e}")

print("\nTesting basic functionality...")
try:
    import numpy as np
    import pandas as pd
    
    # Test TimeSeriesAnalyzer
    prices = pd.Series(np.random.randn(100).cumsum() + 100)
    analyzer = TimeSeriesAnalyzer(prices)
    returns = analyzer.calculate_returns()
    print(f"✓ TimeSeriesAnalyzer: Calculated {len(returns)} returns")
    
    # Test FeatureEngineer
    engineer = FeatureEngineer()
    features = engineer.create_features(prices)
    print(f"✓ FeatureEngineer: Created {len(features.columns)} features")
    
except Exception as e:
    print(f"✗ Error during functionality test: {e}")
