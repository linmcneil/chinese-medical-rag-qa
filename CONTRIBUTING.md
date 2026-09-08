# 贡献指南（CONTRIBUTING）

欢迎提 Issue、PR 或在 Discussions 交流。本仓库为学习研究项目，请遵守 MIT LICENSE 与 README 中的免责声明。

## 代码结构
- `ragqa/`：核心库（data / chunking / retriever / generator / inference / localgen / modelstore / prompts / config）
- 根目录脚本：`qa_system.py`（CLI）、`app_demo.py`（GPU 网页）、`app_local.py`（CPU 网页）、`document_processer.py`（建索引）
- `scripts/`：prepare_data / eval_retrieval / train_lora / eval_generation
- `tests/`：纯 CPU 单元测试（无需 torch/transformers 即可运行）

## 本地开发
```bash
python -m venv .venv-dev
# Windows: .venv-dev\\Scripts\\pip install -e ".[cpu]"
# Linux/macOS: .venv-dev/bin/pip install -e ".[cpu]"
# PyTorch 请按 CPU/CUDA 源先安装；重依赖仍以 requirements-*.txt 为准
python tests/test_chunking.py
python tests/test_data.py
python tests/test_prompts.py
python tests/test_retriever.py
python tests/test_inference_clean.py
python tests/test_rerank.py
```

## 改动约定
1. 尽量保持纯逻辑可单测、重型依赖延迟导入（`--help` / 纯函数路径不需要 torch）。
2. 新功能附可复现脚本；涉及评测口径时同步更新 README 的数字与说明。
3. 不改动样例数据与历史评测结论，除非重跑并有新报告（写入 docs/）。
4. 提交前跑一遍上面 5 个测试文件与 `python -m compileall -q ragqa scripts *.py`。

## PR 检查清单
- [ ] 本地测试全绿
- [ ] 文档（README/CHANGELOG）已同步
- [ ] 未提交模型权重、向量库、大体积产物
- [ ] 未包含任何个人隐私或机器相关信息
- [ ] 代码风格与现有模块保持一致（函数级 import、中文注释、配置走 CLI 参数）