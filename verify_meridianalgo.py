"""
Simple verification script for MeridianAlgo package
"""
import sys
import numpy as np
import pandas as pd

def print_section(title):
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

def test_import():
    print_section("TESTING MERIDIANALGO IMPORT")
    try:
        import meridianalgo
        print(f"✓ Successfully imported meridianalgo version {meridianalgo.__version__}")
        return True
    except Exception as e:
        print(f"✗ Error importing meridianalgo: {e}")
        return False

def test_core_functionality():
    print_section("TESTING CORE FUNCTIONALITY")
    try:
        from meridianalgo import PortfolioOptimizer, TimeSeriesAnalyzer
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            index=dates,
            name='Close'
        )
        
        # Test TimeSeriesAnalyzer
        print("Testing TimeSeriesAnalyzer...")
        analyzer = TimeSeriesAnalyzer(prices)
        returns = analyzer.calculate_returns()
        print(f"✓ Calculated returns: {len(returns)} data points")
        
        # Test PortfolioOptimizer
        print("\nTesting PortfolioOptimizer...")
        returns_df = pd.DataFrame({
            'AAPL': returns,
            'MSFT': returns * 0.8 + np.random.normal(0, 0.01, len(returns))
        })
        optimizer = PortfolioOptimizer(returns_df)
        print(f"✓ Created PortfolioOptimizer with {len(optimizer.returns.columns)} assets")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in core functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_statistics_module():
    print_section("TESTING STATISTICS MODULE")
    try:
        from meridianalgo.statistics import StatisticalArbitrage, calculate_value_at_risk
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        data = pd.DataFrame({
            'AAPL': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            'MSFT': np.cumprod(1 + np.random.normal(0.001, 0.018, 100))
        }, index=dates)
        
        # Test StatisticalArbitrage
        print("Testing StatisticalArbitrage...")
        arb = StatisticalArbitrage(data)
        corr = arb.calculate_rolling_correlation(window=10)
        print(f"✓ Calculated rolling correlation: {corr.shape} shape")
        
        # Test cointegration
        print("\nTesting cointegration...")
        x = data['AAPL']
        y = data['MSFT']
        result = arb.test_cointegration(x, y)
        print(f"✓ Cointegration test results: {result}")
        
        # Test VaR
        print("\nTesting Value at Risk...")
        returns = data.pct_change().dropna()
        var = calculate_value_at_risk(returns['AAPL'], confidence_level=0.95)
        print(f"✓ Calculated VaR: {var:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in statistics module: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MERIDIANALGO PACKAGE VERIFICATION")
    print("="*70)
    
    # Test imports
    if not test_import():
        print("\n✗ Failed to import meridianalgo. Please check the installation.")
        sys.exit(1)
    
    # Test core functionality
    if not test_core_functionality():
        print("\n✗ Core functionality tests failed.")
    
    # Test statistics module
    if not test_statistics_module():
        print("\n✗ Statistics module tests failed.")
    
    print("\nVerification complete!")
    print("="*70)
