"""Check Python environment and installed packages"""

import sys


def check_python_version():
    """Check Python version"""
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")


def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        print(f" {package_name} is installed")
        return True
    except ImportError:
        print(f" {package_name} is NOT installed")
        return False


def main():
    """Main function"""
    print("\n" + "=" * 50)
    print("  PYTHON ENVIRONMENT CHECK")
    print("=" * 50)

    # Check Python version
    print("\n1. Python Environment:")
    check_python_version()

    # Check required packages
    print("\n2. Checking required packages:")
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "yfinance",
        "torch",
        "requests",
        "python-dateutil",
    ]

    results = {pkg: check_package(pkg) for pkg in packages}

    if all(results.values()):
        print("\n All required packages are installed")
    else:
        missing = [pkg for pkg, installed in results.items() if not installed]
        print(f"\n Missing packages: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing)}")


if __name__ == "__main__":
    main()
