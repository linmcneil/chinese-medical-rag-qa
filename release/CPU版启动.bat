@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe -m ragqa ui --index artifacts\demo-v2
pause
