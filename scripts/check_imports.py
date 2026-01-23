"""Check if required imports are working"""


def main():
    print("Testing imports...")

    # Test basic Python
    print("\n1. Testing basic Python...")
    try:
        import sys

        print(f"  Python version: {sys.version}")
        print(f"  Python executable: {sys.executable}")
        print("   Basic Python OK")
    except Exception as e:
        print(f"   Basic Python error: {e}")

    # Test numpy
    print("\n2. Testing numpy...")
    try:
        import numpy as np

        print(f"  numpy version: {np.__version__}")
        print("   numpy OK")
    except Exception as e:
        print(f"   numpy error: {e}")

    # Test pandas
    print("\n3. Testing pandas...")
    try:
        import pandas as pd

        print(f"  pandas version: {pd.__version__}")
        print("   pandas OK")
    except Exception as e:
        print(f"   pandas error: {e}")

    # Test meridianalgo
    print("\n4. Testing meridianalgo...")
    try:
        import meridianalgo

        print(f"  meridianalgo version: {meridianalgo.__version__}")

        # List available attributes
        print("  Available attributes:")
        for attr in dir(meridianalgo):
            if not attr.startswith("_"):
                print(f"    - {attr}")

        print("   meridianalgo OK")
    except Exception as e:
        print(f"   meridianalgo error: {e}")


if __name__ == "__main__":
    main()
