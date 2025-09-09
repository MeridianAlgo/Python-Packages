import sys
print(f"Python version: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import numpy as np
    print(f"numpy version: {np.__version__}")
except ImportError as e:
    print(f"numpy import error: {e}")

try:
    import pandas as pd
    print(f"pandas version: {pd.__version__}")
except ImportError as e:
    print(f"pandas import error: {e}")

try:
    import meridianalgo
    print(f"meridianalgo version: {meridianalgo.__version__}")
except ImportError as e:
    print(f"meridianalgo import error: {e}")
except Exception as e:
    print(f"Error with meridianalgo: {e}")
