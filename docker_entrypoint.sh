#!/usr/bin/env bash
# Docker 容器入口：首次启动用内置样例建索引，再打开本地 CPU 版网页。
set -e
cd "$(dirname "$0")"

if [ ! -d chroma_data ] || [ -z "$(ls -A chroma_data 2>/dev/null)" ]; then
  echo "[docker] 首次启动：用内置样例构建向量索引（CPU，约几分钟）..."
  python document_processer.py --device cpu --data data/train.json
fi

echo "[docker] 启动网页 http://0.0.0.0:7861 （生成模型：${RAGQA_CPU_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}）"
exec python app_local.py --port 7861 --server-name 0.0.0.0 \\
  --cpu-model "${RAGQA_CPU_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"