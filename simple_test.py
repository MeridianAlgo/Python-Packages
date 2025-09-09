# Simple test script to check Python environment

def main():
    print("Simple test script running...")
    
    # Test basic Python
    print("\n1. Basic Python test:")
    try:
        x = 1 + 1
        print(f"  1 + 1 = {x} (Python is working)")
    except Exception as e:
        print(f"  Error in basic Python: {e}")
    
    # Test file operations
    print("\n2. File operations test:")
    try:
        with open("test_file.txt", "w") as f:
            f.write("Test successful!")
        print("  Successfully wrote to test file")
        
        import os
        if os.path.exists("test_file.txt"):
            print("  Test file exists")
            os.remove("test_file.txt")
            print("  Test file cleaned up")
        else:
            print("  Error: Test file not found")
    except Exception as e:
        print(f"  Error in file operations: {e}")
    
    # Test imports
    print("\n3. Testing imports:")
    try:
        import sys
        print(f"  Python version: {sys.version.split(' ')[0]}")
        
        import numpy as np
        print(f"  numpy version: {np.__version__}")
        
        import pandas as pd
        print(f"  pandas version: {pd.__version__}")
        
        print("  All imports successful!")
    except ImportError as e:
        print(f"  Import error: {e}")

if __name__ == "__main__":
    main()
