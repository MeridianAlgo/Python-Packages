# Installation Guide

Complete installation guide for MeridianAlgo.

##  System Requirements

### Python Version
- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Python 3.11+** recommended for best performance

### Operating Systems
- **Windows** 10/11
- **macOS** 10.14+
- **Linux** (Ubuntu 18.04+, CentOS 7+, etc.)

### Hardware Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 4GB+ recommended (8GB+ for large datasets)
- **Storage**: 1GB+ free space
- **GPU**: Optional but recommended for ML features (NVIDIA CUDA, AMD ROCm, Apple MPS)

##  Quick Installation

### Basic Installation

```bash
# Install MeridianAlgo
pip install meridianalgo

# Verify installation
python -c "import meridianalgo; print(meridianalgo.__version__)"
```

### Installation with Optional Dependencies

```bash
# Install with all optional dependencies
pip install "meridianalgo[all]"

# Install with specific groups
pip install "meridianalgo[ml]"            # scikit-learn, torch, statsmodels, hmmlearn
pip install "meridianalgo[optimization]"  # cvxpy, cvxopt
pip install "meridianalgo[volatility]"    # arch (GARCH)
pip install "meridianalgo[data]"          # lxml, beautifulsoup4, polygon-api-client
pip install "meridianalgo[distributed]"   # ray, dask
pip install "meridianalgo[dev]"           # test and lint tooling
```

The core install imports cleanly without any of these extras; modules that need
an optional dependency report as unavailable via `meridianalgo.ModuleRegistry`
until the matching group is installed.

##  Detailed Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv meridianalgo_env

# Activate virtual environment
# Windows
meridianalgo_env\Scripts\activate
# macOS/Linux
source meridianalgo_env/bin/activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install "numpy>=2.0.0" "pandas>=2.2.0" "scipy>=1.12.0"
pip install "yfinance>=0.2.40" "matplotlib>=3.8.0" "seaborn>=0.13.0" "ta>=0.11.0"

# Optional ML dependencies
pip install "scikit-learn>=1.4.0" "torch>=2.2.0" "statsmodels>=0.14.1"
```

### 3. Install MeridianAlgo

```bash
# Install from PyPI
pip install meridianalgo

# Or install from source
git clone https://github.com/MeridianAlgo/Python-Packages.git
cd Python-Packages
pip install -e .
```

##  Development Installation

For developers who want to contribute or modify the source code:

```bash
# Clone repository
git clone https://github.com/MeridianAlgo/Python-Packages.git
cd Python-Packages

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r dev-requirements.txt

# Run tests
pytest tests/
```

##  Conda Installation

If you prefer using Conda:

```bash
# Create conda environment
conda create -n meridianalgo python=3.12

# Activate environment
conda activate meridianalgo

# Install dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn

# Install MeridianAlgo
pip install meridianalgo
```

##  Docker Installation

For containerized deployment:

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy pandas scipy scikit-learn \
    yfinance matplotlib seaborn \
    meridianalgo

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Run application
CMD ["python", "your_script.py"]
```

##  Verification

After installation, verify everything is working:

```python
# Test basic import
import meridianalgo as ma
print(f"MeridianAlgo version: {ma.__version__}")

# Test core functionality
data = ma.get_market_data(['AAPL'], start_date='2023-01-01', end_date='2023-01-31')
print(f"Data shape: {data.shape}")

# Test technical indicators
from meridianalgo.signals.indicators import RSI
rsi = RSI(data['AAPL'], period=14)
print(f"RSI calculated: {len(rsi.dropna())} values")

# Test portfolio optimization
returns = data.pct_change().dropna()
optimizer = ma.PortfolioOptimizer(returns)
result = optimizer.optimize_portfolio(method='sharpe')
print(f"Portfolio optimization successful: {result['sharpe_ratio']:.2f}")

print("All checks passed! MeridianAlgo is working correctly.")
```

##  Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# If you get import errors, try reinstalling
pip uninstall meridianalgo
pip install meridianalgo

# Or install with --force-reinstall
pip install --force-reinstall meridianalgo
```

#### 2. PyTorch Installation Issues

```bash
# For CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. yfinance Issues

```bash
# Update yfinance
pip install --upgrade yfinance

# If still having issues, try:
pip install yfinance==0.2.18
```

#### 4. Memory Issues

```python
# For large datasets, use chunking
import pandas as pd

# Process data in chunks
chunk_size = 1000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # Process chunk
    pass
```

### Platform-Specific Issues

#### Windows

```bash
# If you get Microsoft Visual C++ errors
# Install Microsoft C++ Build Tools
# Or use conda instead of pip
conda install -c conda-forge meridianalgo
```

#### macOS

```bash
# If you get compilation errors
# Install Xcode command line tools
xcode-select --install

# Or use conda
conda install -c conda-forge meridianalgo
```

#### Linux

```bash
# Install build essentials
sudo apt-get update
sudo apt-get install build-essential

# For Ubuntu/Debian
sudo apt-get install python3-dev

# For CentOS/RHEL
sudo yum install python3-devel
```

##  Performance Optimization

### 1. Enable Multi-threading

```python
import numpy as np
import pandas as pd

# Set number of threads
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
```

### 2. Use Efficient Data Types

```python
# Use appropriate data types
df = df.astype({
    'price': 'float32',
    'volume': 'int32',
    'date': 'datetime64[ns]'
})
```

### 3. GPU Acceleration (Optional)

```python
# Check if CUDA is available
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

##  Updating

```bash
# Update to latest version
pip install --upgrade meridianalgo

# Check current version
python -c "import meridianalgo; print(meridianalgo.__version__)"
```

##  Uninstallation

```bash
# Uninstall MeridianAlgo
pip uninstall meridianalgo

# Remove all dependencies (be careful!)
pip uninstall numpy pandas scipy scikit-learn yfinance matplotlib seaborn torch
```

##  Support

If you encounter issues:

1. **Check the [FAQ](faq.md)** for common solutions
2. **Search [GitHub Issues](https://github.com/MeridianAlgo/Python-Packages/issues)** for similar problems
3. **Create a new issue** with detailed error information
4. **Join [GitHub Discussions](https://github.com/MeridianAlgo/Python-Packages/discussions)** for community help

##  Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quickstart.md)** to get started
2. **Explore the [API Reference](api/)** for detailed documentation
3. **Try the [Examples](examples/)** for practical use cases
4. **Check out [Performance Benchmarks](benchmarks.md)** for optimization tips
