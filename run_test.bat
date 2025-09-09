@echo off
echo Running Python environment test...
python test_env.py > test_output.txt 2>&1
type test_output.txt
pause
