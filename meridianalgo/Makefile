.PHONY: help install install-dev test test-cov lint format type-check docs clean build upload

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  type-check   Run type checking"
	@echo "  docs         Build documentation"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  upload       Upload to PyPI"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

# Testing
test:
	pytest tests/

test-cov:
	pytest tests/ --cov=meridianalgo --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -x -v --disable-warnings

# Code quality
lint:
	flake8 meridianalgo tests
	mypy meridianalgo

format:
	black meridianalgo tests
	isort meridianalgo tests

type-check:
	mypy meridianalgo

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Building and distribution
build:
	python -m build

upload: build
	twine upload dist/*

upload-test: build
	twine upload --repository testpypi dist/

# Development helpers
pre-commit:
	pre-commit install
	pre-commit run --all-files

benchmark:
	pytest tests/ --benchmark-only

profile:
	python -m cProfile -o profile.stats -m pytest tests/
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
