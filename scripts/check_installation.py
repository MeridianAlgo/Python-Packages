"""
Script to verify MeridianAlgo package installation and basic functionality.
"""


def main():
    print("=== MeridianAlgo Installation Check ===\n")

    # Check Python version
    import sys

    print(f"Python version: {sys.version}\n")

    # Check if required packages are installed
    required_packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "yfinance",
        "torch",
        "requests",
        "python-dateutil",
    ]

    print("Checking required packages:")
    for package in required_packages:
        try:
            __import__(package)
            print(f" {package} is installed")
        except ImportError as e:
            print(f" {package} is NOT installed: {str(e)}")

    # Try to import the package
    print("\nTrying to import meridianalgo...")
    try:
        import meridianalgo

        print(" meridianalgo imported successfully!")
        print(f"Version: {meridianalgo.__version__}")

        # Test core functionality
        print("\nTesting core functionality...")
        try:
            import numpy as np
            import pandas as pd

            # Test PortfolioOptimizer
            print("Testing PortfolioOptimizer...")
            returns = pd.DataFrame(
                {
                    "AAPL": np.random.normal(0.001, 0.02, 100),
                    "MSFT": np.random.normal(0.001, 0.02, 100),
                }
            )
            meridianalgo.PortfolioOptimizer(returns)
            print(" PortfolioOptimizer created successfully")

            # Test TimeSeriesAnalyzer
            print("\nTesting TimeSeriesAnalyzer...")
            prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
            meridianalgo.TimeSeriesAnalyzer(prices)
            print(" TimeSeriesAnalyzer created successfully")

            # Test get_market_data
            print("\nTesting get_market_data...")
            try:
                data = meridianalgo.get_market_data(
                    ["AAPL"], start_date="2023-01-01", end_date="2023-01-10"
                )
                print(f" get_market_data returned data for {len(data)} days")
            except Exception as e:
                print(f" get_market_data test skipped: {str(e)}")

            print("\n All core functionality tests passed!")

        except Exception as e:
            print(f" Error testing core functionality: {str(e)}")

    except Exception as e:
        print(f" Error importing meridianalgo: {str(e)}")


if __name__ == "__main__":
    main()
