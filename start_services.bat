@echo on
REM ----------------------------------------
REM GLOBAL: unbuffered output for Python
REM ----------------------------------------
set PYTHONUNBUFFERED=1

REM ----------------------------------------
REM Start Waitress apps
REM ----------------------------------------
start cmd /k "python -m server.run_waitress --port=5001"
start cmd /k "python -m server.run_waitress --port=5002"

REM ----------------------------------------
REM Start RF model (logs + live console)
REM ----------------------------------------
start cmd /k "python -m model.LGBM_MODEL"

REM ----------------------------------------
REM Start Nginx
REM ----------------------------------------
cd server\nginx-1.28.0
start cmd /k "nginx.exe"
