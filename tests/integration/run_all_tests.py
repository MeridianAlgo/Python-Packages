#!/usr/bin/env python3
"""
Comprehensive integration test runner for MeridianAlgo.

This script runs all integration tests and provides a summary report.
"""

import sys
import os
import importlib.util
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def run_test_module(module_path: Path) -> tuple[bool, str]:
    """Run a test module and return success status and output."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        
        # Capture output
        from io import StringIO
        import contextlib
        
        output = StringIO()
        with contextlib.redirect_stdout(output):
            spec.loader.exec_module(module)
            if hasattr(module, 'main'):
                success = module.main()
            else:
                success = True
        
        return success, output.getvalue()
        
    except Exception as e:
        return False, f"Error running {module_path.name}: {str(e)}"

def main():
    """Run all integration tests."""
    print("MeridianAlgo Integration Test Suite")
    print("=" * 60)
    
    # Find all test files
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*_integration.py"))
    
    # Also include main test files if they exist
    main_test_dir = test_dir.parent
    main_test_files = [
        main_test_dir / "test_data_infrastructure.py",
        main_test_dir / "test_technical_analysis.py", 
        main_test_dir / "test_portfolio_management.py"
    ]
    
    # Add existing main test files
    for test_file in main_test_files:
        if test_file.exists():
            test_files.append(test_file)
    
    if not test_files:
        print("No integration test files found!")
        return False
    
    results = []
    total_passed = 0
    total_tests = len(test_files)
    
    for test_file in sorted(test_files):
        print(f"\nRunning {test_file.name}...")
        print("-" * 40)
        
        success, output = run_test_module(test_file)
        results.append((test_file.name, success, output))
        
        if success:
            total_passed += 1
            print("✓ PASSED")
        else:
            print("✗ FAILED")
        
        # Print test output
        if output.strip():
            print(output)
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success, _ in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<40} {status}")
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("❌ Some integration tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)