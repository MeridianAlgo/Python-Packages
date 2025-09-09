"""
Fixed test suite for MeridianAlgo package
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
        
        # Test PortfolioOptimizer with correct method
        print("\nTesting PortfolioOptimizer...")
        returns_df = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 1000),
            'MSFT': np.random.normal(0.0008, 0.018, 1000),
            'GOOG': np.random.normal(0.0012, 0.022, 1000)
        })
        optimizer = PortfolioOptimizer(returns_df)
        frontier = optimizer.calculate_efficient_frontier()
        
        if isinstance(frontier, dict) and 'sharpe' in frontier:
            max_sharpe_idx = np.argmax(frontier['sharpe'])
            print(f"✓ Calculated efficient frontier with max Sharpe ratio: {frontier['sharpe'][max_sharpe_idx]:.2f}")
            return True
        else:
            print("❌ Failed to calculate efficient frontier")
            return False
            
    except Exception as e:
        print(f"❌ Core test failed: {str(e)}")
        return False

def test_statistics():
    """Test statistics functionality"""
    print_section("TESTING STATISTICS")
    
    try:
        from meridianalgo import StatisticalArbitrage, calculate_value_at_risk
        
        # Generate test data
        np.random.seed(42)
        data = pd.DataFrame({
            'AAPL': np.cumprod(1 + np.random.normal(0.001, 0.02, 1000)),
            'MSFT': np.cumprod(1 + np.random.normal(0.0008, 0.018, 1000))
        })
        
        # Test StatisticalArbitrage
        print("Testing StatisticalArbitrage...")
        arb = StatisticalArbitrage(data)
        corr = arb.calculate_rolling_correlation(window=20)
        print(f"✓ Calculated rolling correlation (shape: {corr.shape})")
        
        # Test Value at Risk
        print("\nTesting Value at Risk...")
        returns = data['AAPL'].pct_change().dropna()
        var = calculate_value_at_risk(returns, confidence_level=0.95)
        print(f"✓ 95% Value at Risk: {var:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Statistics test failed: {str(e)}")
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
        
        # Test LSTMPredictor with PyTorch
        print("\nTesting LSTMPredictor...")
        try:
            import torch
            from meridianalgo import LSTMPredictor
            
            # Prepare data for LSTM
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
                epochs=2,  # Reduced for testing
                batch_size=32,
                learning_rate=0.001,
                dropout=0.2
            )
            
            # Train model (using first 100 samples for speed)
            X_train = X_scaled[:100]
            y_train = y.iloc[:100].values
            
            # Verify input dimensions
            if X_train.ndim != 2:
                print(f"⚠️ Expected 2D input, got {X_train.ndim}D")
                return False
                
            predictor.fit(X_train, y_train)
            print("✓ LSTM model trained successfully")
            
            # Make predictions
            predictions = predictor.predict(X_train[:5])  # Predict on first 5 samples
            print(f"✓ Made {len(predictions)} predictions")
            
        except ImportError as e:
            print(f"❌ Required package missing: {str(e)}")
            print("Please install PyTorch with: pip install torch")
            return False
        except Exception as e:
            print(f"❌ LSTM test failed: {str(e)}")
            return False
        
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
        ("Statistics", test_statistics),
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
