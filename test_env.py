import sys
import os

print("=== Python Environment Test ===")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Working Directory: {os.getcwd()}")
print("\n=== Environment Variables ===")
for key, value in os.environ.items():
    if 'python' in key.lower() or 'path' in key.lower():
        print(f"{key}: {value}")

print("\n=== Basic Import Test ===")
try:
    import numpy as np
    print("✓ NumPy imported successfully")
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ Failed to import NumPy: {e}")
