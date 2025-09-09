print("Testing meridianalgo import...")
try:
    import meridianalgo
    print(f"Successfully imported meridianalgo version {meridianalgo.__version__}")
    print("\nAvailable attributes:")
    for attr in dir(meridianalgo):
        if not attr.startswith('_'):
            print(f"- {attr}")
except Exception as e:
    print(f"Error importing meridianalgo: {e}")
