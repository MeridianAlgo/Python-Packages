"""
Comprehensive test suite runner for MeridianAlgo package.
"""

import os
import sys
import unittest
import warnings

# Add the parent directory to the path so we can import meridianalgo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


def run_all_tests():
    """Run all test modules and return the result."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_specific_module(module_name):
    """Run tests for a specific module."""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern=f"test_{module_name}.py")

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        module = sys.argv[1]
        print(f"Running tests for {module} module...")
        success = run_specific_module(module)
    else:
        print("Running all tests...")
        success = run_all_tests()

    if success:
        print("\n All tests passed!")
        sys.exit(0)
    else:
        print("\n Some tests failed!")
        sys.exit(1)
