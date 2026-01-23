# Deployment Guide for MeridianAlgo

##  PyPI Deployment

### Prerequisites

1. **Install build tools**:
```bash
pip install build twine
```

2. **Create PyPI account** and get API token from [pypi.org](https://pypi.org)

### Build and Upload Process

#### 1. Build the Package
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build source and wheel distributions
python -m build
```

#### 2. Upload to Test PyPI (Recommended)
```bash
# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ meridianalgo
```

#### 3. Upload to Production PyPI
```bash
# Upload to production PyPI
twine upload dist/*
```

### Authentication

Use API token authentication:
- **Username**: `__token__`
- **Password**: Your PyPI API token

##  Docker Deployment

### Build Docker Image
```bash
# Build the image
docker build -t meridianalgo:latest .

# Run container
docker run -it meridianalgo:latest python -c "import meridianalgo; print('Success!')"
```

### Docker Compose for Development
```yaml
version: '3.8'
services:
  meridianalgo:
    build: .
    volumes:
      - .:/app
    ports:
      - "8888:8888"  # Jupyter
    environment:
      - PYTHONPATH=/app
```

##  Cloud Deployment

### AWS Lambda
```python
# lambda_function.py
import meridianalgo as ma

def lambda_handler(event, context):
    # Your quantitative analysis code
    data = ma.get_market_data(['AAPL'], start_date='2023-01-01')
    result = ma.calculate_metrics(data)
    return {'statusCode': 200, 'body': result}
```

### Google Cloud Functions
```python
# main.py
import functions_framework
import meridianalgo as ma

@functions_framework.http
def analyze_portfolio(request):
    # Portfolio analysis endpoint
    symbols = request.json.get('symbols', ['AAPL', 'GOOGL'])
    analysis = ma.PortfolioOptimizer().analyze(symbols)
    return analysis
```

### Azure Functions
```python
# __init__.py
import azure.functions as func
import meridianalgo as ma

def main(req: func.HttpRequest) -> func.HttpResponse:
    # Risk analysis service
    portfolio_data = req.get_json()
    risk_metrics = ma.calculate_risk_metrics(portfolio_data)
    return func.HttpResponse(risk_metrics)
```

##  CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

##  Monitoring and Analytics

### Package Usage Analytics
- Monitor download statistics on PyPI
- Track GitHub repository metrics
- Set up error reporting with Sentry

### Performance Monitoring
```python
# Add to your application
import meridianalgo as ma

# Enable performance monitoring
ma.enable_monitoring(
    service='your-service',
    environment='production'
)
```

##  Security Considerations

### API Key Management
```python
# Use environment variables
import os
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

# Or use cloud secret managers
from azure.keyvault.secrets import SecretClient
secret_client = SecretClient(vault_url, credential)
api_key = secret_client.get_secret('alpha-vantage-key').value
```

### Data Encryption
```python
# Encrypt sensitive data
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher_suite = Fernet(key)
encrypted_data = cipher_suite.encrypt(sensitive_data.encode())
```

##  Scaling Considerations

### Horizontal Scaling
- Use Redis for caching market data
- Implement database connection pooling
- Deploy multiple instances behind load balancer

### Performance Optimization
- Enable GPU acceleration where available
- Use distributed computing with Dask/Ray
- Implement intelligent caching strategies

##  Troubleshooting

### Common Deployment Issues

#### PyPI Upload Failures
```bash
# Check package structure
python setup.py check

# Validate metadata
twine check dist/*

# Test upload to TestPyPI first
twine upload --repository testpypi dist/*
```

#### Docker Build Issues
```bash
# Check Dockerfile syntax
docker build --no-cache -t meridianalgo:debug .

# Debug container
docker run -it meridianalgo:debug /bin/bash
```

#### Cloud Function Timeouts
- Optimize cold start performance
- Use connection pooling
- Implement caching strategies
- Consider using container-based deployments

##  Support

For deployment issues:
1. Check the [troubleshooting guide](TROUBLESHOOTING.md)
2. Review deployment logs
3. Contact support at support@meridianalgo.com
4. Open GitHub issue with deployment details

---

**MeridianAlgo** - Deploy anywhere, analyze everywhere! 