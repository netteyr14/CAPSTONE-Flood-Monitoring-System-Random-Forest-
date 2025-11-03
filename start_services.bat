@echo on
REM Start Waitress app on port 5001
start cmd /k "python -m server.run_waitress --port=5001"

REM Start Waitress app on port 5002
start cmd /k "python -m server.run_waitress --port=5002"

REM Start RF MODEL on port 5002
start cmd /k "python -m model.model_server"

REM Start Nginx
cd server\nginx-1.28.0
start cmd /k "nginx.exe"
