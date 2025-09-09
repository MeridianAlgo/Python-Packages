# Quick verification of MeridianAlgo package
import sys

def main():
    print("Testing MeridianAlgo package...")
    
    # Test basic imports
    try:
        import numpy as np
        import pandas as pd
        from meridianalgo import TimeSeriesAnalyzer, PortfolioOptimizer
        print("✓ Imports successful")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return
    
    # Test TimeSeriesAnalyzer
    try:
        prices = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 101])
        analyzer = TimeSeriesAnalyzer(prices)
        returns = analyzer.calculate_returns()
        print(f"✓ TimeSeriesAnalyzer test passed ({len(returns)} returns calculated)")
    except Exception as e:
        print(f"✗ TimeSeriesAnalyzer test failed: {e}")
    
    # Test PortfolioOptimizer
    try:
        returns_df = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01, 0.03, -0.02],
            'MSFT': [0.015, 0.01, 0.005, -0.01, 0.02]
        })
        optimizer = PortfolioOptimizer(returns_df)
        frontier = optimizer.calculate_efficient_frontier()
        print(f"✓ PortfolioOptimizer test passed (Frontier points: {len(frontier['returns'])})")
    except Exception as e:
        print(f"✗ PortfolioOptimizer test failed: {e}")

if __name__ == "__main__":
    main()
