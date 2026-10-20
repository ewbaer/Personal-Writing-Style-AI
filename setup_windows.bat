@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.11 --version
if errorlevel 1 goto missing_python
py -3.11 -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)"
if errorlevel 1 goto wrong_python
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\check_gpu.py
if errorlevel 1 goto failed
echo Setup finished. Double-click start_app.bat to open the app.
echo Copy your laptop's models\ai_detector.joblib here to restore detection.
pause
exit /b 0
:missing_python
echo Install Python 3.11 64-bit, then run this file again.
echo PowerShell command: winget install -e --id Python.Python.3.11
pause
exit /b 1
:wrong_python
echo The existing .venv uses a different Python version.
echo Rename .venv to .venv-old, then run setup_windows.bat again.
pause
exit /b 1
:failed
echo Setup stopped. Read the error above before trying again.
pause
exit /b 1
