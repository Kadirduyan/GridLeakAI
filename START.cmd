@echo off
setlocal
cd /d "%~dp0"
if exist "backend\.venv\Scripts\python.exe" goto install
set "GRIDLEAK_PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not exist "%GRIDLEAK_PYTHON%" set "GRIDLEAK_PYTHON=python"
"%GRIDLEAK_PYTHON%" -m venv backend\.venv
if errorlevel 1 goto failed
:install
if not exist "backend\.venv\installed-requirements.txt" goto packages
fc /b backend\requirements.txt backend\.venv\installed-requirements.txt >nul 2>&1
if not errorlevel 1 goto run
:packages
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if errorlevel 1 goto failed
copy /y backend\requirements.txt backend\.venv\installed-requirements.txt >nul
:run
echo GridLeak: http://127.0.0.1:8000
echo API docs: http://127.0.0.1:8000/docs
echo Stop: Ctrl+C. Keep this window open while using GridLeak.
backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto failed
exit /b 0
:failed
echo Startup failed. Check the error above. Port 8000 may already be in use.
pause
exit /b 1
