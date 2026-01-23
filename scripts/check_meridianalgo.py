"""Check MeridianAlgo package functionality"""


def main():
    print("=== MeridianAlgo Package Check ===\n")

    # Test basic Python
    print("1. Testing Python environment...")
    try:
        import sys

        print(f"  Python version: {sys.version.split()[0]}")
        print(f"  Executable: {sys.executable}")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # Test imports
    print("\n2. Testing imports...")
    try:
        import numpy as np
        import pandas as pd

        print(f"  numpy: {np.__version__}")
        print(f"  pandas: {pd.__version__}")
    except ImportError as e:
        print(f"  Error: {e}")
        return

    # Test MeridianAlgo import
    print("\n3. Testing MeridianAlgo import...")
    try:
        import meridianalgo

        print(f"  meridianalgo version: {meridianalgo.__version__}")
    except Exception as e:
        print(f"  Error importing meridianalgo: {e}")
        return

    # Test core functionality
    print("\n4. Testing core functionality...")
    try:
        from meridianalgo import PortfolioOptimizer, TimeSeriesAnalyzer

        # Create test data
        np.random.seed(42)
        prices = pd.Series(np.random.randn(100).cumsum() + 100)

        # Test TimeSeriesAnalyzer
        analyzer = TimeSeriesAnalyzer(prices)
        returns = analyzer.calculate_returns()
        print(f"   TimeSeriesAnalyzer: Calculated {len(returns)} returns")

        # Test PortfolioOptimizer
        returns_df = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 1000),
                "MSFT": np.random.normal(0.0008, 0.018, 1000),
            }
        )
        optimizer = PortfolioOptimizer(returns_df)
        frontier = optimizer.calculate_efficient_frontier()
        print(
            f"   PortfolioOptimizer: Calculated efficient frontier with {len(frontier['returns'])} points"
        )

    except Exception as e:
        print(f"  Error testing core functionality: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n MeridianAlgo package is working correctly!")


if __name__ == "__main__":
    main()
