#!/usr/bin/env python3
"""
Simple test script to verify MeridianAlgo imports correctly.
"""

def test_basic_import():
    """Test basic package import."""
    try:
        import meridianalgo as ma
        print(f"✅ MeridianAlgo {ma.__version__} imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_api_functionality():
    """Test basic API functionality."""
    try:
        import meridianalgo as ma
        
        # Test API creation
        api = ma.get_api()
        print(f"✅ API created successfully")
        
        # Test system info
        info = api.get_system_info()
        print(f"✅ System info: {info['package_version']}")
        
        # Test available modules
        modules = api.get_available_modules()
        available_count = sum(modules.values())
        total_count = len(modules)
        print(f"✅ Available modules: {available_count}/{total_count}")
        
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_statistics():
    """Test statistics functionality."""
    try:
        import meridianalgo as ma
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Test VaR calculation
        var_95 = ma.calculate_value_at_risk(returns, 0.95)
        print(f"✅ VaR calculation: {var_95:.4f}")
        
        # Test metrics calculation
        metrics = ma.stats_calculate_metrics(returns)
        print(f"✅ Metrics calculation: {len(metrics)} metrics")
        
        return True
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicators."""
    try:
        import meridianalgo as ma
        import pandas as pd
        import numpy as np
        
        # Create sample price data
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 100)))
        
        # Test RSI
        rsi = ma.RSI(prices, period=14)
        print(f"✅ RSI calculation: {len(rsi.dropna())} values")
        
        # Test SMA
        sma = ma.SMA(prices, period=20)
        print(f"✅ SMA calculation: {len(sma.dropna())} values")
        
        return True
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing MeridianAlgo Package...")
    print("=" * 50)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("API Functionality", test_api_functionality),
        ("Statistics", test_statistics),
        ("Technical Indicators", test_technical_indicators)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready for deployment.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Package needs fixes.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)