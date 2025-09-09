import sys
import subprocess

def run_test():
    try:
        # Run the test script and capture output
        result = subprocess.run(
            [sys.executable, "test_meridianalgo_simple.py", "-v"],
            capture_output=True,
            text=True
        )
        
        # Print the output
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        print("=== RETURN CODE ===")
        print(result.returncode)
        
    except Exception as e:
        print(f"Error running test: {str(e)}")

if __name__ == "__main__":
    run_test()
