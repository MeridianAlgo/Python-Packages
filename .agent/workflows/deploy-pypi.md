---
description: How to deploy MeridianAlgo to PyPI
---

# Deploy to PyPI Workflow

This workflow describes the process of publishing a new version of MeridianAlgo to PyPI.

## Prerequisites

1.  **PyPI Account**: Ensure you have an account on [PyPI](https://pypi.org/).
2.  **API Token**: Create an API token on PyPI and add it as a GitHub secret named `PYPI_API_TOKEN`.
3.  **Permissions**: Ensure you have write access to the repository.

## Steps

### 1. Update Version

Ensure the version in `meridianalgo/__init__.py` and `setup.py` (if applicable) is bumped.

### 2. Lint and Test

Run the full suite of tests and linting to ensure quality.

```bash
# Linting
ruff check .

# Testing
pytest tests/
```

### 3. Tag the Release

Create and push a new git tag. The GitHub Action is triggered by tags starting with `v`.

```bash
git tag v6.0.1
git push origin v6.0.1
```

### 4. Monitor Deployment

Go to the **Actions** tab on GitHub and monitor the "Publish to PyPI" workflow.

### 5. Verify

Once completed, verify the new version is available on [PyPI](https://pypi.org/project/meridianalgo/).
