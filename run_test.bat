@echo off
echo Running Python environment test...
python -c "from tests.verify_environment import main; main()" > test_output.txt 2>&1
type test_output.txt
pause
