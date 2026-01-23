"""Script to install required dependencies and test the package"""

import importlib
import subprocess
import sys


def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_import(package_name):
    try:
        importlib.import_module(package_name)
        print(f" {package_name} is installed")
        return True
    except ImportError:
        print(f" {package_name} is not installed")
        return False


# List of required packages
required_packages = ["numpy", "pandas", "scipy", "scikit-learn", "yfinance", "torch"]

print("=== Setting up dependencies ===\n")

# Install missing packages
for package in required_packages:
    if not check_import(package):
        install_package(package)

# Install the package in development mode
print("\nInstalling meridianalgo in development mode...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])

print("\n=== Dependencies setup complete ===\n")
print("You can now run the test script with: python -m tests.test_package")
