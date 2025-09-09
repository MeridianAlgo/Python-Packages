@echo off
REM Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Upgrade pip and install packages
echo Installing dependencies...
python -m pip install --upgrade pip
pip install numpy pandas scipy scikit-learn tensorflow yfinance

REM Install package in development mode
echo Installing MeridianAlgo in development mode...
pip install -e .

REM Run tests
echo Running tests...
python test_all.py

pause
