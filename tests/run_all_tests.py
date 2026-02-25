#!/usr/bin/env python3
"""
Comprehensive test runner for MeridianAlgo v6.2.2.

This script runs all tests including unit tests, integration tests, and performance benchmarks.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_test_suite(test_name, test_file):
    """Run a specific test suite and return results."""
    print(f"\n{'=' * 60}")
    print(f"Running {test_name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Run pytest on the specific test file
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f" {test_name} PASSED ({duration:.2f}s)")
            return True, duration, result.stdout
        else:
            print(f" {test_name} FAILED ({duration:.2f}s)")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False, duration, result.stderr

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f" {test_name} ERROR ({duration:.2f}s): {e}")
        return False, duration, str(e)


def run_integration_tests():
    """Run integration tests."""
    print(f"\n{'=' * 60}")
    print("Running Integration Tests")
    print(f"{'=' * 60}")

    integration_dir = Path(__file__).parent / "integration"
    if integration_dir.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(integration_dir / "run_all_tests.py")],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(" Integration Tests PASSED")
                return True
            else:
                print(" Integration Tests FAILED")
                print("Output:", result.stdout)
                return False
        except Exception as e:
            print(f" Integration Tests ERROR: {e}")
            return False
    else:
        print("  Integration tests directory not found")
        return True


def main():
    """Run all tests and generate summary report."""
    print(" MeridianAlgo v6.2.2 - Comprehensive Test Suite")
    print("=" * 80)

    # Define test suites
    test_suites = [
        ("Data Infrastructure Tests", "test_data_infrastructure.py"),
        ("Technical Analysis Tests", "test_technical_analysis.py"),
        ("Portfolio Management Tests", "test_portfolio_management.py"),
        ("Backtesting Engine Tests", "test_backtesting_engine.py"),
        ("Machine Learning Tests", "test_ml_framework.py"),
        ("Fixed Income Tests", "test_fixed_income.py"),
        ("Risk Management Tests", "test_risk_management.py"),
        ("HPC Architecture Tests", "test_hpc_architecture.py"),
        ("API Integration Tests", "test_api_integration.py"),
    ]

    results = []
    total_duration = 0

    # Run each test suite
    for test_name, test_file in test_suites:
        test_path = Path(__file__).parent / test_file

        if test_path.exists():
            success, duration, output = run_test_suite(test_name, test_path)
            results.append((test_name, success, duration, output))
            total_duration += duration
        else:
            print(f"  {test_name} file not found: {test_file}")
            results.append((test_name, None, 0, f"File not found: {test_file}"))

    # Run integration tests
    integration_success = run_integration_tests()

    # Generate summary report
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY REPORT")
    print(f"{'=' * 80}")

    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0

    for test_name, success, duration, _output in results:
        if success is True:
            status = " PASSED"
            passed_tests += 1
        elif success is False:
            status = " FAILED"
            failed_tests += 1
        else:
            status = "  SKIPPED"
            skipped_tests += 1

        print(f"{test_name:<40} {status:<12} ({duration:.2f}s)")

    # Integration tests summary
    if integration_success:
        print(f"{'Integration Tests':<40} {' PASSED':<12}")
        passed_tests += 1
    else:
        print(f"{'Integration Tests':<40} {' FAILED':<12}")
        failed_tests += 1

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS:")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"    Skipped: {skipped_tests}")
    print(f"    Total Duration: {total_duration:.2f}s")
    print(f"{'=' * 80}")

    # Overall result
    if failed_tests == 0:
        print(" ALL TESTS PASSED! MeridianAlgo v4.0.0 is ready for production!")
        return 0
    else:
        print(f" {failed_tests} TEST(S) FAILED! Please review and fix issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
