"""Test PyTorch installation and basic functionality"""

def test_pytorch():
    print("Testing PyTorch installation...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        print(f"Error testing PyTorch: {e}")
        return False

if __name__ == "__main__":
    if test_pytorch():
        print("\nPyTorch is working correctly!")
    else:
        print("\nThere was an issue with PyTorch installation.")
        print("Try reinstalling with: pip install torch torchvision torchaudio")
