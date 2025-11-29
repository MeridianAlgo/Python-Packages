.. _contributing:

Contributing to MeridianAlgo
============================

We welcome contributions to MeridianAlgo! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a virtual environment
4. Install the package in development mode

.. code-block:: bash

    git clone https://github.com/yourusername/Python-Packages.git
    cd Python-Packages/meridianalgo
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -e .[dev]

Development Workflow
--------------------

1. Create a new branch for your feature or bugfix
2. Make your changes
3. Add tests for new functionality
4. Run the test suite
5. Update documentation if needed
6. Submit a pull request

.. code-block:: bash

    git checkout -b feature/your-feature-name
    # Make your changes
    pytest tests/
    # Update docs if needed
    git add .
    git commit -m "Add your feature"
    git push origin feature/your-feature-name

Code Style
-----------

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Before submitting a PR, run:

.. code-block:: bash

    black meridianalgo tests
    isort meridianalgo tests
    flake8 meridianalgo tests
    mypy meridianalgo

Testing
-------

All new features must include tests. We use pytest for testing.

Running Tests
~~~~~~~~~~~~~~

Run all tests:

.. code-block:: bash

    pytest

Run tests with coverage:

.. code-block:: bash

    pytest --cov=meridianalgo --cov-report=html

Run specific test file:

.. code-block:: bash

    pytest tests/test_core.py

Writing Tests
~~~~~~~~~~~~~

Tests should be placed in the ``tests/`` directory. Follow these guidelines:

1. Use descriptive test names
2. Test both success and failure cases
3. Use fixtures for test data
4. Mock external dependencies

Example test:

.. code-block:: python

    import pytest
    import numpy as np
    import pandas as pd
    from meridianalgo.core import PortfolioOptimizer
    
    @pytest.fixture
    def sample_returns():
        np.random.seed(42)
        return pd.DataFrame(
            np.random.normal(0.001, 0.02, (252, 3)),
            columns=['AAPL', 'MSFT', 'GOOGL']
        )
    
    def test_portfolio_optimization(sample_returns):
        optimizer = PortfolioOptimizer(sample_returns)
        result = optimizer.optimize_portfolio()
        
        assert 'weights' in result
        assert np.isclose(sum(result['weights']), 1.0)

Documentation
-------------

Documentation is built with Sphinx. To build locally:

.. code-block:: bash

    cd docs
    make html

The documentation will be available at ``_build/html/index.html``.

When adding new features, update:

1. Docstrings (use Google or NumPy style)
2. API reference (autodoc will pick up new functions)
3. User guide (if it's a major feature)
4. Examples (if helpful)

Pull Requests
-------------

When submitting a PR:

1. Use a clear title and description
2. Link to relevant issues
3. Include tests
4. Update documentation
5. Ensure CI passes

PR Template
~~~~~~~~~~~

Use this template for your PR description:

```
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

Release Process
----------------

1. Update version number in ``__init__.py``
2. Update CHANGELOG.md
3. Create a release tag
4. Build and upload to PyPI

.. code-block:: bash

    # Update version
    # Update changelog
    
    git tag -a v4.1.0 -m "Version 4.1.0"
    git push origin v4.1.0
    
    # Build and upload
    python -m build
    twine upload dist/*

Community
----------

- GitHub Discussions: Ask questions and share ideas
- Issues: Report bugs and request features
- Wiki: Community-maintained documentation

Code of Conduct
---------------

Be respectful, inclusive, and professional. We follow the Python Community Code of Conduct.

Getting Help
------------

If you need help:

1. Check the documentation
2. Search existing issues
3. Create a new issue with details
4. Join our Discord/Slack community

Thank you for contributing to MeridianAlgo! 🚀
