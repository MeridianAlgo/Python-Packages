"""
Basic test script for MeridianAlgo package
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
        
        # Test PortfolioOptimizer
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

def run_tests():
    """Run all tests and print summary"""
    print("\n" + "="*80)
    print("  MERIDIANALGO BASIC TEST SUITE")
    print("="*80)
    
    tests = [
        ("Core Functionality", test_core),
        ("Statistics", test_statistics)
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
    run_tests()
