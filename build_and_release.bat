@echo off
echo Building MeridianAlgo v3.0.0...

:: Clean up previous builds
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist meridianalgo.egg-info rmdir /s /q meridianalgo.egg-info

:: Create source distribution and wheel
python setup.py sdist bdist_wheel

echo.
echo Build complete! Files created in the 'dist' directory:
dir /b dist

echo.
echo To upload to PyPI, run:
echo twine upload dist/*

pause
