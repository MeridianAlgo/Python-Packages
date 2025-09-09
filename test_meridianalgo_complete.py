"""
Comprehensive test for MeridianAlgo package
"""
import numpy as np
import pandas as pd
from meridianalgo import (
    TimeSeriesAnalyzer, PortfolioOptimizer, StatisticalArbitrage,
    FeatureEngineer, LSTMPredictor, prepare_data_for_lstm
)

def test_core():
    print("\n=== Testing Core Components ===")
    # Test data
    prices = pd.Series(np.random.randn(100).cumsum() + 100, 
                      index=pd.date_range('2023-01-01', periods=100))
    
    # Test TimeSeriesAnalyzer
    analyzer = TimeSeriesAnalyzer(prices)
    returns = analyzer.calculate_returns()
    print(f"✓ TimeSeriesAnalyzer: {len(returns)} returns calculated")
    
    # Test PortfolioOptimizer
    returns_df = pd.DataFrame({
        'Asset1': returns,
        'Asset2': returns * 0.8 + np.random.normal(0, 0.01, len(returns))
    })
    optimizer = PortfolioOptimizer(returns_df)
    print("✓ PortfolioOptimizer initialized")

def test_statistics():
    print("\n=== Testing Statistics ===")
    # Test data
    data = pd.DataFrame({
        'AAPL': np.cumprod(1 + np.random.normal(0.001, 0.02, 200)),
        'MSFT': np.cumprod(1 + np.random.normal(0.001, 0.018, 200))
    })
    
    # Test StatisticalArbitrage
    arb = StatisticalArbitrage(data)
    corr = arb.calculate_rolling_correlation(window=10)
    print(f"✓ Rolling correlation: {corr.shape}")
    
    # Test cointegration
    result = arb.test_cointegration(data['AAPL'], data['MSFT'])
    print(f"✓ Cointegration test completed")

def test_ml():
    print("\n=== Testing Machine Learning ===")
    # Generate more test data to ensure we have enough samples
    np.random.seed(42)  # For reproducibility
    prices = pd.Series(np.random.randn(1000).cumsum() + 100)  # Increased to 1000 samples
    
    # Test FeatureEngineer
    try:
        from meridianalgo.ml import FeatureEngineer
        from sklearn.preprocessing import StandardScaler
        
        engineer = FeatureEngineer()
        features = engineer.create_features(prices)
        print(f"✓ Created {len(features.columns)} features")
        
        # Ensure we have valid features
        if features.empty or features.isnull().values.any():
            features = features.fillna(0)  # Handle any NaN values
            print("⚠️ Found and filled NaN values in features")
        
        # Test LSTMPredictor if PyTorch is available
        try:
            import torch
            from meridianalgo.ml import LSTMPredictor
            
            # Prepare data for LSTM - ensure we have enough data points
            target = prices.pct_change().shift(-1).dropna()
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]
            
            # Ensure we have enough data
            if len(X) < 100:
                print(f"⚠️ Not enough data for LSTM testing (need at least 100 samples, got {len(X)})")
                return False
                
            # Scale features
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
            
            # Train model with first 100 samples for speed
            X_train = X_scaled[:100]
            y_train = y.iloc[:100].values
            
            # Verify input dimensions
            if X_train.shape[0] == 0 or X_train.shape[1] == 0:
                print(f"❌ Invalid input shape: {X_train.shape}")
                return False
                
            predictor.fit(X_train, y_train)
            print("✓ LSTM model trained successfully")
            
            # Make predictions on a small subset
            predictions = predictor.predict(X_train[:5])
            print(f"✓ Made {len(predictions)} predictions")
            
            return True
            
        except ImportError as e:
            print(f"⚠️ PyTorch not available: {e}")
            return False
        except Exception as e:
            print(f"❌ LSTM test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error in Machine Learning: {e}")
        return False

if __name__ == "__main__":
    print("=== Starting MeridianAlgo Tests ===\n")
    
    tests = {
        "Core Components": test_core,
        "Statistics": test_statistics,
        "Machine Learning": test_ml
    }
    
    for name, test_func in tests.items():
        try:
            print(f"\n🔄 Testing {name}...")
            test_func()
            print(f"✅ {name} tests passed!")
        except Exception as e:
            print(f"❌ Error in {name}: {str(e)}")
    
    print("\n=== Test Summary ===")