"""
Comprehensive test suite for MeridianAlgo package
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

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
        
        # Test PortfolioOptimizer
        print("\nTesting PortfolioOptimizer...")
        returns_df = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 1000),
            'MSFT': np.random.normal(0.0008, 0.018, 1000),
            'GOOG': np.random.normal(0.0012, 0.022, 1000)
        })
        optimizer = PortfolioOptimizer(returns_df)
        weights = optimizer.optimize_portfolio()
        print("✓ Optimized portfolio weights:")
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.2%}")
            
        return True
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
        from sklearn.preprocessing import StandardScaler
        
        # Generate test data
        np.random.seed(42)
        prices = pd.Series(np.random.randn(1000).cumsum() + 100)
        
        # Test FeatureEngineer
        print("Testing FeatureEngineer...")
        engineer = FeatureEngineer()
        features = engineer.create_features(prices)
        print(f"✓ Created {len(features.columns)} features")
        
        # Check if PyTorch is available for LSTM tests
        try:
            import torch
            from meridianalgo import LSTMPredictor
            
            # Test LSTMPredictor
            print("\nTesting LSTMPredictor...")
            target = prices.pct_change().shift(-1).dropna()
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]
            
            if len(X) < 100:
                print("⚠️ Not enough data for LSTM testing")
                return False
                
            # Split and scale data
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            predictor = LSTMPredictor(sequence_length=10, input_size=X_train_scaled.shape[1])
            predictor.train(X_train_scaled, y_train.values, epochs=5, batch_size=32, verbose=0)
            print("✓ LSTM model trained successfully")
            
            # Make predictions
            predictions = predictor.predict(X_test_scaled)
            print(f"✓ Made {len(predictions)} predictions")
            
        except ImportError:
            print("⚠️ PyTorch not available, skipping LSTM tests")
        
        return True
    except Exception as e:
        print(f"❌ ML test failed: {str(e)}")
        return False

def test_yfinance():
    """Test yfinance integration"""
    print_section("TESTING YAHOO FINANCE INTEGRATION")
    
    try:
        print("Downloading AAPL data...")
        data = yf.download('AAPL', start='2023-01-01', end='2023-12-31', progress=False)
        
        if data.empty:
            print("⚠️ No data downloaded from Yahoo Finance")
            return False
            
        print(f"✓ Downloaded {len(data)} days of AAPL data")
        
        # Test with TimeSeriesAnalyzer
        from meridianalgo import TimeSeriesAnalyzer
        analyzer = TimeSeriesAnalyzer(data['Close'])
        returns = analyzer.calculate_returns()
        print(f"✓ Calculated {len(returns)} daily returns")
        
        return True
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and print summary"""
    print("\n" + "="*80)
    print("  MERIDIANALGO COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Core Functionality", test_core),
        ("Statistics", test_statistics),
        ("Machine Learning", test_ml),
        ("Yahoo Finance", test_yfinance)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n🔄 Running {name} tests...")
        results[name] = test_func()
    
    # Print summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASSED ✅" if passed else "FAILED ❌"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("  ALL TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("  SOME TESTS FAILED. PLEASE CHECK THE OUTPUT ABOVE. ⚠️")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
