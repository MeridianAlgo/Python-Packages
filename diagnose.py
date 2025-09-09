"""Diagnostic script to check Python environment and package structure"""
import sys
import os
import importlib
import subprocess

def print_header(text):
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50)

def check_python():
    print_header("PYTHON ENVIRONMENT")
    print(f"Python version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Python path: {sys.path}")

def check_packages():
    print_header("REQUIRED PACKAGES")
    packages = [
        'numpy', 'pandas', 'scipy', 'scikit-learn', 'yfinance', 'torch',
        'meridianalgo'
    ]
    
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            print(f"✓ {pkg} ({getattr(mod, '__version__', 'unknown version')})")
        except ImportError as e:
            print(f"✗ {pkg}: {str(e)}")

def check_package_structure():
    print_header("PACKAGE STRUCTURE")
    package_path = os.path.join(os.getcwd(), 'meridianalgo')
    if os.path.exists(package_path):
        print(f"Package directory exists at: {package_path}")
        print("\nContents:")
        for root, dirs, files in os.walk(package_path):
            level = root.replace(package_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")
    else:
        print(f"Package directory not found at: {package_path}")

def run_tests():
    print_header("RUNNING TESTS")
    test_cmd = [sys.executable, "-m", "pytest", "-v"]
    print(f"Running: {' '.join(test_cmd)}")
    try:
        subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with error code: {e.returncode}")

if __name__ == "__main__":
    check_python()
    check_packages()
    check_package_structure()
    run_tests()
