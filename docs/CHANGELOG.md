# CHANGELOG

## 2026-09-08（升级第 6 轮：可选 reranker 重排 + 开源社区化）
- 新增 `ragqa/rerank.py`：`rerank_hits` 纯函数 + `CrossEncoderReranker`（bge-reranker-base，模型延迟加载）
- `qa_system.py` 新增 `--rerank / --rerank-model / --rerank-candidates`：向量 top-N 候选经 cross-encoder 精排到最终 top-k
- `scripts/eval_retrieval.py --rerank`：一次评测同时输出「纯向量 Top-k」与「向量+重排」两套 Hit@k 与延迟，结果 JSON 含 rerank 段
- 单元测试扩至 49 项（新增 rerank 排序/截断/稳定序 6 条）；README/ROADMAP 同步
- 新增 CONTRIBUTING.md、GitHub Issue/PR 模板

## 2026-09-08（升级第 5 轮：可安装 + 容器化 + 测试扩展）
- 新增 `pyproject.toml`：`pip install -e .` 可安装；CLI 入口 `ragqa-qa / ragqa-build-index / ragqa-app / ragqa-app-gpu`；
  重型依赖仍以 requirements-*.txt 为准，另提供 cpu/gpu 可选 extras
- 新增 `Dockerfile` + `docker-compose.yml`：CPU 版一键容器化，首次启动自动建索引（端口 7861）
- `app_local.py` 新增 `--server-name`（Docker 内可绑定 0.0.0.0）
- 单元测试扩至 43 项：新增 prompts、retriever（假 collection）、生成后处理与检索保护纯逻辑测试；
  `ragqa/inference.py` 改为 torch 函数内延迟导入，纯函数可在无 torch 环境测试
- `start_release.sh`：修正隧道脚本名，Python 路径支持回退到 python3


## 2026-09-08（升级第 4 轮：生成质量收口 + 发行版调优）
- 生成默认改为「基座 CareBot + RAG」：网页默认关 LoRA（LoRA 把语料里的营销式客套尾巴学歪了，仅保留作对比实验）
- ragqa/inference.py 生成后处理：去提示词复读、去“答案是/参考文献/医生询问”等转场截断、整句去重、
  删除“谢谢/祝您/仅供参考/AI身份声明/再见”等客套与跑题句、按句号边界限长（≤300 字）
- 检索注入保护：clean_context_docs 清洗片段（只留解答主体）+ 关键词相关性门；命中不可信时自动降级为
  模型直接回答，不再把脏语料喂给模型（修复“检索到孕妇语料→回答跑题”类问题）
- 参数：temperature 0.1、repetition_penalty 1.3、no_repeat_ngram_size 5、默认 max_new≈250
- 实测 5 例：回答 95~280 字、无复读无客套尾巴；Q1 柚子/降压药、Q2 小儿咳嗽等开头正常收尾
- 云端已重启发行版：gradio 0.0.0.0:7860（AutoDL 实例）
## 2026-09-08（升级第 3 轮：AutoDL 实机出数 + LoRA 实验 + 网页发行版）
- 检索评测（RTX 5090，1000 问）：Hit@1 86.5% / Hit@3 90.3% / Hit@5 91.1%，平均 3.5ms/查询
- LoRA 微调实验（QLoRA，5000 条，2 epoch，24.9 分钟，train_loss≈0.84）：
  - 留出 40 条生成对比：bge 语义相似度 0.789 → 0.851（+6.3%）；Rouge-1 1.25 → 5.46；回答长度 390 → 262 字
  - 复现脚本 scripts/train_lora.py；报告 docs/EXPERIMENT.md
- 发行版：app_demo.py（Gradio：基座/LoRA 切换 + RAG 开关）、start_release.sh、release/双击打开演示.bat（SSH 隧道）
- 新增 ragqa/inference.py（统一 4bit 推理与 LoRA 挂载）、scripts/eval_generation.py（生成对比）
## 2026-09-08（升级第 2 轮：数据层 + 生成/检索重构 + 评测）
- 数据层
  - 新增 `ragqa/data.py`：CSV 编码自动探测(GBK/UTF-8)、字段解析、清洗去重、蓄水池抽样、确定性划分、RAG/SFT 两种格式
  - 新增 `scripts/prepare_data.py`：`--source {sample,toyhom,local}`，支持指定科室与规模
    `--dept 内科 --limit 30000 --eval-size 2000`；评测集从知识库内抽样并带 rid，可回溯
  - 用 3.4MB 样例端到端验证：train=5000 / eval=1000 / sft=5000（评测记录 100% 在知识库内）
- 分块
  - `chunking.py` 支持可选的【科室】【主题】渲染；新增 `chunk_records_with_ids`（分块保留记录 id，供评测与溯源）
- 建索引
  - `document_processer.py` 默认读 `data/train.json`；写入每块的 `rid` 元数据；改用共享的 `ragqa/modelstore.py`
- 检索与生成
  - 新增 `ragqa/prompts.py`（提示词+后处理）、`ragqa/retriever.py`（Chroma 封装+rid 返回）、`ragqa/generator.py`
    （CareBot 加载：4bit/8bit 量化需 bitsandbytes，缺失时明确回退，不再“假 8bit”）
  - 新增 `ragqa/modelstore.py`：模型本地化下载统一入口，重型依赖全部延迟导入
  - 重写根目录 `qa_system.py`：REPL/单问/`--retrieve-only` 三模式，路径可配置，`--help` 不需装 torch/transformers
- 评测
  - 新增 `scripts/eval_retrieval.py`：Hit@1/3/5、平均查询延迟、分科室命中、结果 JSON 输出
- 工程与文档
  - 单元测试：chunking 14/14、data 9/9（共 23 项）
  - 重写 `README.md`；更新 `ROADMAP.md`、`requirements.txt`、`.gitignore`
- 待办（真机）
  - 建索引跑出块数/耗时 → 跑 Hit@k 出数字 → reranker/混合检索对比 → 生成质量评估

## 2026-09-07（升级第 1 轮，A 阶段）
- 新增 `ragqa/` 包：`config.py`、`chunking.py`
- 重写 `document_processer.py`：CLI 参数化、sha1 断点续传、批量写入、--device auto
- 新增 `.gitignore`、`requirements.txt`、`docs/{ROADMAP,CHANGELOG}.md`
- 重构前版本已本地归档，不随仓库分发