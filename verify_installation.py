"""
Script to verify MeridianAlgo package installation and basic functionality.
"""

def main():
    print("=== MeridianAlgo Installation Check ===\n")
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}\n")
    
    # Check if required packages are installed
    required_packages = ['numpy', 'pandas', 'yfinance']
    
    print("Checking required packages:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
    
    # Try to import the package
    print("\nTrying to import meridianalgo...")
    try:
        import meridianalgo
        print("✓ meridianalgo imported successfully!")
        print(f"Version: {meridianalgo.__version__}")
        
        # Try to access some core functionality
        print("\nTesting core functionality...")
        try:
            import numpy as np
            import pandas as pd
            
            # Create sample data
            dates = pd.date_range('2023-01-01', periods=100)
            prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
                             index=dates, name='Close')
            returns = prices.pct_change().dropna()
            
            # Test if we can create a PortfolioOptimizer
            from meridianalgo.core import PortfolioOptimizer
            optimizer = PortfolioOptimizer(pd.DataFrame({'Test': returns}))
            print("✓ PortfolioOptimizer created successfully")
            
            # Test calculating covariance matrix
            cov_matrix = optimizer._calculate_covariance_matrix()
            print("✓ Covariance matrix calculated successfully")
            
        except Exception as e:
            print(f"✗ Error testing core functionality: {str(e)}")
            
    except Exception as e:
        print(f"✗ Error importing meridianalgo: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you've installed the package in development mode")
        print("   Run: pip install -e . from the package root directory")
        print("2. Check for any syntax errors in the package files")
        print("3. Make sure all required dependencies are installed")

if __name__ == "__main__":
    main()
