# MeridianAlgo Documentation

This directory contains comprehensive documentation for the MeridianAlgo package.

## Documentation Files

### index.md
Main documentation homepage with overview and quick links.

### installation.md
Installation instructions for different platforms and use cases.

### quickstart.md
Quick start guide to get up and running in minutes.

### API_REFERENCE.md
Complete API reference for all modules and functions.

### benchmarks.md
Performance benchmarks and optimization guidelines.

## API Documentation

### api/portfolio_management.md
Documentation for portfolio optimization and management:
- Portfolio optimization algorithms
- Risk parity
- Black-Litterman model
- Hierarchical Risk Parity (HRP)
- Rebalancing strategies

### api/technical_indicators.md
Documentation for technical analysis:
- Moving averages
- Momentum indicators
- Volatility indicators
- Volume indicators
- Custom indicator creation

## Building Documentation

If you have Sphinx installed, you can build HTML documentation:

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build
```

## Documentation Structure

```
docs/
├── README.md                    # This file
├── index.md                     # Documentation homepage
├── installation.md              # Installation guide
├── quickstart.md               # Quick start tutorial
├── API_REFERENCE.md            # Complete API reference
├── benchmarks.md               # Performance benchmarks
├── api/
│   ├── portfolio_management.md # Portfolio API docs
│   └── technical_indicators.md # Technical analysis API docs
├── examples/                   # Example code snippets
└── requirements.txt            # Documentation dependencies
```

## Contributing to Documentation

When adding new features:
1. Update relevant API documentation
2. Add examples to demonstrate usage
3. Update the quickstart if it's a core feature
4. Add benchmarks for performance-critical code

## Documentation Standards

- Use clear, concise language
- Include code examples for all features
- Explain the "why" not just the "how"
- Link to related functions and concepts
- Keep examples self-contained and runnable
