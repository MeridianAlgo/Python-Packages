"""
MeridianAlgo Setup Configuration

The Complete Quantitative Finance Platform
"""

import os

from setuptools import find_packages, setup

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements


setup(
    name="meridianalgo",
    version="6.2.1",
    author="Meridian Algorithmic Research Team",
    author_email="support@meridianalgo.com",
    description="MeridianAlgo - The Complete Quantitative Finance Platform for Professional Developers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeridianAlgo/Python-Packages",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core scientific stack
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        # Data acquisition
        "yfinance>=0.2.30",
        "requests>=2.31.0",
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        # Technical analysis
        "ta>=0.11.0",
        # Utilities
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
        "python-dateutil>=2.8.2",
        "pytz>=2023.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "ruff>=0.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "torch>=2.1.0",
            "statsmodels>=0.14.0",
            "hmmlearn>=0.3.0",
        ],
        "optimization": [
            "cvxpy>=1.4.0",
            "cvxopt>=1.3.0",
        ],
        "volatility": [
            "arch>=6.2.0",
        ],
        "data": [
            "lxml>=4.9.0",
            "beautifulsoup4>=4.12.0",
            "polygon-api-client>=1.12.0",
        ],
        "distributed": [
            "ray>=2.7.0",
            "dask>=2023.10.0",
        ],
        "full": [
            # ML
            "scikit-learn>=1.3.0",
            "torch>=2.1.0",
            "statsmodels>=0.14.0",
            "hmmlearn>=0.3.0",
            # Optimization
            "cvxpy>=1.4.0",
            # Volatility
            "arch>=6.2.0",
            # Data
            "lxml>=4.9.0",
            "beautifulsoup4>=4.12.0",
        ],
        "all": [
            # Everything
            "scikit-learn>=1.3.0",
            "torch>=2.1.0",
            "statsmodels>=0.14.0",
            "hmmlearn>=0.3.0",
            "cvxpy>=1.4.0",
            "cvxopt>=1.3.0",
            "arch>=6.2.0",
            "lxml>=4.9.0",
            "beautifulsoup4>=4.12.0",
            "polygon-api-client>=1.12.0",
            "ray>=2.7.0",
        ],
    },
    keywords=[
        # Core
        "quantitative-finance",
        "algorithmic-trading",
        "trading",
        "finance",
        # Portfolio & Risk
        "portfolio-optimization",
        "risk-management",
        "portfolio-analytics",
        "pyfolio",
        # Execution
        "execution-algorithms",
        "vwap",
        "twap",
        "market-impact",
        # Market Microstructure
        "market-microstructure",
        "liquidity",
        "order-book",
        "vpin",
        # Strategies
        "statistical-arbitrage",
        "pairs-trading",
        "mean-reversion",
        "factor-models",
        # Derivatives
        "options-pricing",
        "black-scholes",
        "greeks",
        "volatility-surface",
        # Advanced
        "high-frequency-trading",
        "regime-detection",
        "machine-learning",
        "quantlib",
        "backtrader",
        "zipline",
    ],
    project_urls={
        "Bug Reports": "https://github.com/MeridianAlgo/Python-Packages/issues",
        "Source": "https://github.com/MeridianAlgo/Python-Packages",
        "Documentation": "https://meridianalgo.readthedocs.io",
        "Changelog": "https://github.com/MeridianAlgo/Python-Packages/blob/main/CHANGELOG.md",
    },
    entry_points={
        "console_scripts": [
            "meridianalgo=meridianalgo.cli:main",
        ],
    },
    include_package_data=True,
    license_files = [],   # ← this disables the automatic license-file metadata
    zip_safe=False,
    license="MIT",
)
