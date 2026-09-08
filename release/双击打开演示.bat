@echo off
chcp 65001 >nul
echo 正在建立到云端 GPU 发行版的 SSH 隧道（本机 7860 -^> 云端 7860）...
echo 前提：云端实例上已运行 bash start_release.sh（监听 0.0.0.0:7860）
echo.
echo 使用前请先编辑下面这行 ssh 命令，把 3 处占位符换成你自己的信息：
echo   1. PUT_YOUR_PRIVATE_KEY_PATH - 你的 SSH 私钥路径（Windows 如 C:\Users\你的用户名\.ssh\id_ed25519）
echo   2. root@YOUR_SERVER_HOST     - 你云端实例的用户名与地址
echo   3. 22                        - 你云端实例的 SSH 端口
echo.
start "" http://localhost:7860
ssh -N -L 7860:127.0.0.1:7860 -i "PUT_YOUR_PRIVATE_KEY_PATH" -p 22 root@YOUR_SERVER_HOST
pause