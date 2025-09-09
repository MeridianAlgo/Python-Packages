@echo off
REM Script to set up a clean Python environment for MeridianAlgo

echo Creating virtual environment...
python -m venv venv

if exist "venv\Scripts\activate" (
    @echo off
echo Setting up Python environment...

REM Create virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Install the package in development mode
echo Installing MeridianAlgo in development mode...
pip install -e .

echo Environment setup complete!
pause
echo Run 'venv\Scripts\activate' to activate the environment.
) else (
    echo Failed to create virtual environment.
    exit /b 1
)
