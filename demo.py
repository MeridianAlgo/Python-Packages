#!/usr/bin/env python3
"""
MeridianAlgo Demo Script

This script demonstrates the key features of the Ultimate Quantitative Development Platform.
Run this script to see the comprehensive library capabilities in action.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def demo_basic_functionality():
    """Demonstrate basic package functionality."""
    print("=" * 80)
    print("MeridianAlgo - Ultimate Quantitative Development Platform Demo")
    print("=" * 80)
    
    try:
        import meridianalgo as ma
        print(f"✅ MeridianAlgo v{ma.__version__} imported successfully!")
        
        # Test basic imports
        print("\n📦 Available modules:")
        print("  - Data: Multi-source data providers, real-time streaming, processing")
        print("  - Technical Analysis: 200+ indicators, pattern recognition")
        print("  - Portfolio: Advanced optimization, risk management, attribution")
        print("  - Backtesting: Event-driven engine, realistic market simulation")
        print("  - Machine Learning: Financial ML models, feature engineering")
        print("  - Fixed Income: Bond pricing, derivatives valuation")
        print("  - Risk Analysis: VaR, stress testing, regulatory compliance")
        print("  - ML: FeatureEngineer, LSTMPredictor")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import MeridianAlgo: {e}")
        return False

def demo_portfolio_optimization():
    """Demonstrate portfolio optimization."""
    print("\n" + "=" * 60)
    print("Portfolio Optimization Demo")
    print("=" * 60)
    
    try:
        import meridianalgo as ma
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'MSFT': np.random.normal(0.0008, 0.018, 252),
            'GOOG': np.random.normal(0.0012, 0.022, 252)
        }, index=dates)
        
        # Create portfolio optimizer
        optimizer = ma.PortfolioOptimizer(returns_data)
        print("✅ Portfolio optimizer created")
        
        # Calculate efficient frontier
        frontier = optimizer.calculate_efficient_frontier(num_portfolios=100)
        print(f"✅ Efficient frontier calculated with {len(frontier['volatility'])} portfolios")
        
        # Find optimal portfolio
        max_sharpe_idx = np.argmax(frontier['sharpe'])
        optimal_weights = frontier['weights'][max_sharpe_idx]
        
        print("\n📊 Optimal Portfolio Weights:")
        for i, ticker in enumerate(['AAPL', 'MSFT', 'GOOG']):
            print(f"  {ticker}: {optimal_weights[i]:.2%}")
        
        print(f"\n📈 Portfolio Metrics:")
        print(f"  Expected Return: {frontier['returns'][max_sharpe_idx]:.2%}")
        print(f"  Volatility: {frontier['volatility'][max_sharpe_idx]:.2%}")
        print(f"  Sharpe Ratio: {frontier['sharpe'][max_sharpe_idx]:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Portfolio optimization failed: {e}")
        return False

def demo_risk_metrics():
    """Demonstrate risk metrics calculation."""
    print("\n" + "=" * 60)
    print("Risk Metrics Demo")
    print("=" * 60)
    
    try:
        import meridianalgo as ma
        
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
        # Calculate risk metrics
        var_95 = ma.calculate_value_at_risk(returns, confidence_level=0.95)
        var_99 = ma.calculate_value_at_risk(returns, confidence_level=0.99)
        es_95 = ma.calculate_expected_shortfall(returns, confidence_level=0.95)
        
        print("✅ Risk metrics calculated")
        
        print(f"\n📊 Risk Metrics:")
        print(f"  95% Value at Risk: {var_95:.2%}")
        print(f"  99% Value at Risk: {var_99:.2%}")
        print(f"  95% Expected Shortfall: {es_95:.2%}")
        
        # Calculate performance metrics
        metrics = ma.calculate_metrics(returns)
        print(f"\n📈 Performance Metrics:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Risk metrics calculation failed: {e}")
        return False

def demo_time_series_analysis():
    """Demonstrate time series analysis."""
    print("\n" + "=" * 60)
    print("Time Series Analysis Demo")
    print("=" * 60)
    
    try:
        import meridianalgo as ma
        
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 252)), index=dates)
        
        # Create time series analyzer
        analyzer = ma.TimeSeriesAnalyzer(prices)
        print("✅ Time series analyzer created")
        
        # Calculate returns and volatility
        returns = analyzer.calculate_returns()
        volatility = analyzer.calculate_volatility(window=21, annualized=True)
        
        print(f"✅ Calculated {len(returns)} returns and {volatility.notna().sum()} volatility values")
        
        # Calculate Hurst exponent
        hurst = ma.hurst_exponent(returns)
        print(f"\n📊 Time Series Properties:")
        print(f"  Hurst Exponent: {hurst:.3f}")
        
        if hurst > 0.5:
            print("  → Series shows trending behavior")
        elif hurst < 0.5:
            print("  → Series shows mean-reverting behavior")
        else:
            print("  → Series shows random walk behavior")
        
        return True
    except Exception as e:
        print(f"❌ Time series analysis failed: {e}")
        return False

def demo_machine_learning():
    """Demonstrate machine learning features."""
    print("\n" + "=" * 60)
    print("Machine Learning Demo")
    print("=" * 60)
    
    try:
        import meridianalgo as ma
        
        # Create sample data
        np.random.seed(42)
        prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 500)))
        
        # Feature engineering
        engineer = ma.FeatureEngineer()
        features = engineer.create_features(prices)
        
        print(f"✅ Created {len(features.columns)} features:")
        for col in features.columns[:5]:  # Show first 5 features
            print(f"  - {col}")
        if len(features.columns) > 5:
            print(f"  ... and {len(features.columns) - 5} more")
        
        # Test LSTM predictor (if PyTorch is available)
        try:
            import torch
            print("\n✅ PyTorch available - testing LSTM predictor...")
            
            # Create small dataset for testing
            target = prices.pct_change().shift(-1).dropna()
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]
            
            if len(X) > 100:
                # Scale features
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train LSTM model
                predictor = ma.LSTMPredictor(sequence_length=10, epochs=2)
                predictor.fit(X_scaled, y.values)
                
                # Make predictions
                predictions = predictor.predict(X_scaled[-50:])
                print(f"✅ LSTM model trained and made {len(predictions)} predictions")
            else:
                print("⚠️ Not enough data for LSTM training")
                
        except ImportError:
            print("⚠️ PyTorch not available - skipping LSTM demo")
        
        return True
    except Exception as e:
        print(f"❌ Machine learning demo failed: {e}")
        return False

def demo_market_data():
    """Demonstrate market data fetching."""
    print("\n" + "=" * 60)
    print("Market Data Demo")
    print("=" * 60)
    
    try:
        import meridianalgo as ma
        
        print("📡 Fetching market data from Yahoo Finance...")
        data = ma.get_market_data(['AAPL'], start_date='2023-01-01', end_date='2023-01-10')
        
        if not data.empty:
            print(f"✅ Retrieved {len(data)} days of AAPL data")
            print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
            if 'AAPL' in data.columns:
                print(f"  Price range: ${data['AAPL'].min():.2f} - ${data['AAPL'].max():.2f}")
            else:
                print(f"  Columns: {list(data.columns)}")
        else:
            print("⚠️ No data retrieved (check internet connection)")
        
        return True
    except Exception as e:
        print(f"❌ Market data demo failed: {e}")
        return False

def main():
    """Run all demos."""
    print("🚀 MeridianAlgo Package Demo")
    print("This demo showcases the key features of the MeridianAlgo library.")
    
    demos = [
        ("Basic Functionality", demo_basic_functionality),
        ("Portfolio Optimization", demo_portfolio_optimization),
        ("Risk Metrics", demo_risk_metrics),
        ("Time Series Analysis", demo_time_series_analysis),
        ("Machine Learning", demo_machine_learning),
        ("Market Data", demo_market_data)
    ]
    
    results = {}
    for name, demo_func in demos:
        print(f"\n🔄 Running {name} demo...")
        results[name] = demo_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("Demo Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("\n🎉 All demos completed successfully!")
        print("MeridianAlgo is working correctly and ready to use!")
    else:
        print(f"\n⚠️ {total - passed} demos failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
