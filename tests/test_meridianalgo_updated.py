"""
Updated test suite for MeridianAlgo package
"""
import numpy as np
import pandas as pd
from datetime import datetime

def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_core():
    """Test core functionality"""
    print_section("TESTING CORE FUNCTIONALITY")
    
    try:
        from meridianalgo import TimeSeriesAnalyzer, PortfolioOptimizer
        
        # Test TimeSeriesAnalyzer
        print("Testing TimeSeriesAnalyzer...")
        dates = pd.date_range(end=datetime.now(), periods=100)
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        analyzer = TimeSeriesAnalyzer(prices)
        returns = analyzer.calculate_returns()
        print(f"✓ Calculated {len(returns)} returns")
        
        # Test PortfolioOptimizer with calculate_efficient_frontier
        print("\nTesting PortfolioOptimizer...")
        returns_df = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 1000),
            'MSFT': np.random.normal(0.0008, 0.018, 1000),
            'GOOG': np.random.normal(0.0012, 0.022, 1000)
        })
        optimizer = PortfolioOptimizer(returns_df)
        frontier = optimizer.calculate_efficient_frontier()
        
        # Get the portfolio with maximum Sharpe ratio
        max_sharpe_idx = np.argmax(frontier['sharpe'])
        optimal_weights = frontier['weights'][max_sharpe_idx]
        
        print("✓ Calculated efficient frontier")
        print(f"  Optimal portfolio Sharpe ratio: {frontier['sharpe'][max_sharpe_idx]:.2f}")
        for i, ticker in enumerate(returns_df.columns):
            print(f"  {ticker}: {optimal_weights[i]:.2%}")
            
        return True
    except Exception as e:
        print(f"❌ Core test failed: {str(e)}")
        return False

def test_ml():
    """Test machine learning functionality"""
    print_section("TESTING MACHINE LEARNING")
    
    try:
        from meridianalgo import FeatureEngineer
        
        # Generate test data
        np.random.seed(42)
        prices = pd.Series(np.random.randn(1000).cumsum() + 100)
        
        # Test FeatureEngineer
        print("Testing FeatureEngineer...")
        engineer = FeatureEngineer()
        features = engineer.create_features(prices)
        print(f"✓ Created {len(features.columns)} features")
        
        # Skip LSTM test if torch is not available
        try:
            import torch
            from meridianalgo import LSTMPredictor
            
            # Prepare data for LSTM
            print("\nTesting LSTMPredictor...")
            target = prices.pct_change().shift(-1).dropna()
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]
            
            if len(X) < 100:
                print("⚠️ Not enough data for LSTM testing")
                return False
                
            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize and train LSTM
            predictor = LSTMPredictor(
                sequence_length=10,
                hidden_size=50,
                num_layers=2,
                epochs=5,  # Reduced for testing
                batch_size=32,
                learning_rate=0.001,
                dropout=0.2
            )
            
            # Train model
            predictor.fit(X_scaled, y.values)
            print("✓ LSTM model trained successfully")
            
            # Make predictions
            predictions = predictor.predict(X_scaled[:10])  # Predict on first 10 samples
            print(f"✓ Made {len(predictions)} predictions")
            
        except ImportError:
            print("⚠️ PyTorch not available, skipping LSTM tests")
        
        return True
    except Exception as e:
        print(f"❌ ML test failed: {str(e)}")
        return False

def run_tests():
    """Run all tests and print summary"""
    print("\n" + "="*80)
    print("  MERIDIANALGO TEST SUITE")
    print("="*80)
    
    tests = [
        ("Core Functionality", test_core),
        ("Machine Learning", test_ml)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n🔄 Running {name} tests...")
        results[name] = test_func()
    
    # Print summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASSED ✅" if passed else "FAILED ❌"
        print(f"{name}: {status}")
    
    print("\n" + "="*80)
    if all_passed:
        print("  ALL TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("  SOME TESTS FAILED. PLEASE CHECK THE OUTPUT ABOVE. ⚠️")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    run_tests()
