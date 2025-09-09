"""Test script to verify all imports in MeridianAlgo package"""

def test_imports():
    """Test importing all modules and components."""
    print("Testing imports...")
    
    # Test core imports
    try:
        from meridianalgo import (
            PortfolioOptimizer, 
            TimeSeriesAnalyzer,
            get_market_data,
            calculate_metrics,
            calculate_max_drawdown
        )
        print("✓ Core components imported successfully")
    except ImportError as e:
        print(f"✗ Error importing core components: {e}")
        return False
    
    # Test statistics imports
    try:
        from meridianalgo.statistics import StatisticalArbitrage
        print("✓ Statistics module imported successfully")
    except ImportError as e:
        print(f"✗ Error importing statistics module: {e}")
        return False
    
    # Test ML imports
    try:
        from meridianalgo.ml import FeatureEngineer, LSTMPredictor, prepare_data_for_lstm
        print("✓ ML module imported successfully")
    except ImportError as e:
        print(f"✗ Error importing ML module: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== MeridianAlgo Import Test ===\n")
    success = test_imports()
    
    if success:
        print("\n✓ All imports successful!")
    else:
        print("\n✗ Some imports failed. Please check the error messages above.")
