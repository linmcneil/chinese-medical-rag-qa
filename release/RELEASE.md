# 医疗问答发行版使用说明（2026-09-08）

本项目提供两条发行链路，代码都在本仓库：

## A. 云端 GPU 完整版（8B CareBot，基座/LoRA + RAG 开关）
- 基座：BAAI/CareBot_Medical_multi-llama3-8b-instruct（4bit 加载）
- LoRA：5000 条 Toyhom 医疗问答（内科等）指令微调 2 epoch（实验对比用）
- 检索：bge-small-zh + Chroma（8524 块），脏检索自动降级
- 界面：Gradio（app_demo.py），支持「微调版/基座」「RAG 开/关」切换

云端一键启动（AutoDL 等 GPU 实例上）：
    cd /root/ragflow
    bash start_release.sh           # 监听 0.0.0.0:7860
本机访问：双击 `release/双击打开演示.bat`（SSH 隧道自动开浏览器）。

## B. 本地 CPU 版（无需显卡，适合演示与复现）
- 两种模式（app_local.py）：
  1. 轻量生成：Qwen/Qwen2.5-1.5B-Instruct 在 CPU 上跑（首次下载约 3GB，建议 8G 内存）；
  2. 仅检索摘录：不加载大模型，直接给出知识库最相关片段的解答（几秒出结果）。
- 端口：http://127.0.0.1:7861

Windows 一键：
    双击 `release/安装CPU版.bat`   （建 .venv-cpu、装依赖、建本地索引）
    双击 `release/CPU版启动.bat`   （启动网页）
命令行（Linux/macOS 同思路）见根目录 README「本地 CPU」小节。

## 关键文件
- 训练脚本：scripts/train_lora.py；生成对比：scripts/eval_generation.py
- 检索评测：python -m scripts.eval_retrieval --limit 1000 --device cuda
- LoRA 权重与训练日志在 GPU 实例 outputs/ 下（不入库，可用脚本复现）

## 免责声明
仅供学习研究，不构成医疗建议。