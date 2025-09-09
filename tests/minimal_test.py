"""Minimal test to identify the issue"""

def main():
    print("=== MINIMAL TEST ===\n")
    
    # Test basic Python functionality
    print("1. Testing basic Python functionality...")
    try:
        print("  - Basic print statement works")
        x = 1 + 1
        print(f"  - Basic math works: 1 + 1 = {x}")
        
        # Test imports
        print("\n2. Testing imports...")
        import numpy as np
        import pandas as pd
        print("  - Successfully imported numpy and pandas")
        
        # Test package import
        print("\n3. Testing meridianalgo import...")
        try:
            import meridianalgo
            print(f"  - Successfully imported meridianalgo version {meridianalgo.__version__}")
            
            # Test if core components exist
            print("\n4. Testing core components...")
            from meridianalgo import PortfolioOptimizer, TimeSeriesAnalyzer
            print("  - Successfully imported PortfolioOptimizer and TimeSeriesAnalyzer")
            
            # Create test data
            np.random.seed(42)
            returns = pd.DataFrame({
                'AAPL': np.random.normal(0.001, 0.02, 100),
                'MSFT': np.random.normal(0.001, 0.02, 100)
            })
            
            # Test PortfolioOptimizer
            print("\n5. Testing PortfolioOptimizer...")
            optimizer = PortfolioOptimizer(returns)
            print(f"  - Created PortfolioOptimizer with {len(optimizer.returns.columns)} assets")
            
            # Test TimeSeriesAnalyzer
            print("\n6. Testing TimeSeriesAnalyzer...")
            prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
            analyzer = TimeSeriesAnalyzer(prices)
            print(f"  - Created TimeSeriesAnalyzer with {len(analyzer.data)} data points")
            
            print("\n✓ ALL TESTS PASSED!")
            
        except Exception as e:
            print(f"  - Error: {str(e)}")
            print("\n✗ TEST FAILED")
            
    except Exception as e:
        print(f"  - Error during imports: {str(e)}")
        print("\n✗ TEST FAILED")

if __name__ == "__main__":
    main()
