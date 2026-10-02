@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please follow LOCAL_APP_SETUP.md first to create the Python environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
