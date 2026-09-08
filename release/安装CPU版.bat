@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo ============================================
echo  中文医疗 RAG 问答 - 本地 CPU 版 安装向导
echo ============================================
echo [1/4] 检查 Python...
python --version >nul 2>&1 || (echo 未找到 Python，请先安装 Python 3.10+ 并勾选 Add to PATH & pause & exit /b 1)
if not exist ".venv-cpu\Scripts\python.exe" (
  echo [2/4] 创建虚拟环境 .venv-cpu ...
  python -m venv .venv-cpu || (echo 创建虚拟环境失败 & pause & exit /b 1)
)
echo [3/4] 安装依赖（首次约 5-15 分钟，视网速）...
".venv-cpu\Scripts\python.exe" -m pip install --upgrade pip
".venv-cpu\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
".venv-cpu\Scripts\python.exe" -m pip install -r requirements-cpu.txt
echo [4/4] 构建本地向量索引（首次约几分钟，生成后可跳过）...
".venv-cpu\Scripts\python.exe" document_processer.py --device cpu
echo.
echo 安装完成！请双击 “CPU版启动.bat” 使用。
pause