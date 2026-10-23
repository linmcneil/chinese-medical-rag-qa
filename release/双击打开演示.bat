@echo off
cd /d "%~dp0.."
start "" http://127.0.0.1:7861
.venv\Scripts\python.exe -m ragqa ui --index artifacts\demo-v2
pause
