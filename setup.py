from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meridianalgo",
    version="3.0.0",
    author="Meridian Algorithmic Research Team",
    author_email="support@meridianalgo.com",
    description="Advanced Algorithmic Trading and Statistical Analysis Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeridianAlgo/Python-Packages",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "yfinance>=0.2.0",
        "requests>=2.28.0",
        "torch>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "ta>=0.10.0",
        "pandas-ta>=0.3.14",
        "statsmodels>=0.13.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "python-dateutil>=2.8.2",
        "pytz>=2021.3",
        "lxml>=4.6.0",
        "beautifulsoup4>=4.10.0",
        "pytest>=6.2.5",
        "python-dateutil>=2.8.2",
        "torch>=1.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ]
    },
    keywords=[
        "finance", 
        "trading", 
        "algorithmic-trading", 
        "quantitative-finance", 
        "statistical-arbitrage",
        "portfolio-optimization",
        "machine-learning"
    ],
    project_urls={
        "Bug Reports": "https://github.com/MeridianAlgo/Python-Packages/issues",
        "Source": "https://github.com/MeridianAlgo/Python-Packages",
    },
)
