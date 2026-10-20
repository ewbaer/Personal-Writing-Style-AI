@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
if exist "models\ai_detector.joblib" (
    echo A trained detector already exists. Back it up and rename it before retraining.
    pause
    exit /b 1
)
if exist "data\detector_labeled.csv" goto train
if not exist "data\AI_Human.csv" (
    echo Copy AI_Human.csv into data, or copy models\ai_detector.joblib from your laptop.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\prepare_detector_data.py
if errorlevel 1 goto failed
:train
".venv\Scripts\python.exe" scripts\train_detector.py --csv data\detector_labeled.csv
if errorlevel 1 goto failed
echo Detector saved. Restart the app to load it.
pause
exit /b 0
:failed
echo Training stopped. Read the error above.
pause
exit /b 1
