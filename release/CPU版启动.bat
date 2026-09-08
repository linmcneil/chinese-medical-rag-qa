@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if not exist ".venv-cpu\Scripts\python.exe" (
  echo 未找到 .venv-cpu，请先运行 “安装CPU版.bat”
  pause
  exit /b 1
)
echo 正在启动本地 CPU 版网页（http://127.0.0.1:7861）...
start "" http://127.0.0.1:7861
".venv-cpu\Scripts\python.exe" app_local.py --port 7861
pause