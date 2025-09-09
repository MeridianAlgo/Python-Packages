@echo off
REM Script to set up a clean Python environment for MeridianAlgo

echo Creating virtual environment...
python -m venv venv

if exist "venv\Scripts\activate" (
    echo Activating virtual environment...
    call venv\Scripts\activate
    
    echo Upgrading pip...
    python -m pip install --upgrade pip
    
    echo Installing dependencies...
    pip install numpy pandas scipy scikit-learn yfinance torch
    
    echo Installing meridianalgo in development mode...
    pip install -e .
    
    echo.
    echo Setup complete! Run 'venv\Scripts\activate' to activate the environment.
) else (
    echo Failed to create virtual environment.
    exit /b 1
)
