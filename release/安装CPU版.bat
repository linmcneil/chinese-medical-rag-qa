@echo off
cd /d "%~dp0.."
python -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -e ".[api,ui]"
if errorlevel 1 exit /b 1
if not exist artifacts\demo-v2 .venv\Scripts\python.exe -m ragqa build --data examples\knowledge.json --index artifacts\demo-v2
pause
