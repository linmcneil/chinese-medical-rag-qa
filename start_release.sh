#!/usr/bin/env bash
# AutoDL 上启动发行版（前台运行；保持窗口即可）
set -e
cd "$(dirname "$0")"
echo "启动医疗问答网页版 -> http://127.0.0.1:7860"
echo "远程访问：本机执行 release/start_demo_local.bat（SSH 隧道）或 AutoDL 自定义服务映射 7860"
if [ ! -d models/CareBot_Medical ]; then echo "缺少基座模型 models/CareBot_Medical"; exit 1; fi
/root/miniconda3/bin/python app_demo.py --port 7860