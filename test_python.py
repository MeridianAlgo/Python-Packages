print("Python is working!")
print(f"Python version: {__import__('sys').version}")
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy is not installed")
