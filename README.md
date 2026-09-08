# Chinese Medical RAG QA · 中文医疗检索增强问答
[![CI](https://github.com/linmcneil/chinese-medical-rag-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/linmcneil/chinese-medical-rag-qa/actions) · 开源仓库：https://github.com/linmcneil/chinese-medical-rag-qa

基于 **RAG（检索增强生成）** 的中文医疗问答系统：清洗真实医疗问答语料 → 字段感知分块 →
向量化建索引 → 回答时先检索相关知识片段 → 再交给医疗大模型生成“有出处、可解释”的回答。

> 本项目仅供学习研究，输出不构成医疗建议，涉及健康问题请及时就医。

---

## 1. 结果速览（2026-09-08 · AutoDL RTX 5090）

| 实验 | 关键数字 | 出处 |
| --- | --- | --- |
| 检索 Hit@1 / Hit@3 / Hit@5 | **86.5% / 90.3% / 91.1%**（1000 问留出样本，8524 块知识库） | `docs/EXPERIMENT.md`、`data/eval_result.json` |
| 检索延迟 | 平均 **5.1 ms/查询**（GPU，单次 Top-k，复测） | `data/eval_result_rerank.json` |
| 重排对比（Top-20 + bge-reranker-base） | Hit@1 **86.5% → 91.4%**，Hit@3 90.3% → 92.3%，Hit@5 91.1% → 92.3% | 同上 |
| 重排延迟 | 向量 5.1ms + 重排 25.1ms ≈ **30.3 ms/问**（GPU，端到端） | 同上 |
| 生成对比（基座 vs QLoRA） | bge 语义相似度 **0.789 → 0.851**；回答平均长度 390 → 262 字 | `outputs/semantic_summary.json` |
| 微调成本 | 5000 条 × 2 epoch，RTX 5090 上约 **25 分钟**（QLoRA 4bit，adapter 168MB） | `docs/EXPERIMENT.md` |
| 工程测试 | 核心纯逻辑单元测试 **49/49 通过**（分块 14 + 数据 9 + 逻辑 26） | `tests/`、GitHub Actions CI |

一句话定位：**用向量检索把“大模型幻觉 + 缺领域知识”的问题，替换成“先查资料、再组织回答”。**

---

## 2. 架构

```
 数据层                    索引层                       问答层
─────────────           ──────────────             ──────────────
Toyhom 医疗语料 ──> 清洗/抽样 ──> bge-small-zh ──> Chroma 8524 块
(GBK/UTF-8 识别)   字段感知分块      ~100MB          sha1 断点续传
                    【科室】【主题】                   批量写入
                        │                                ▲
                        │            eval.json(1000 问) ──┘  Hit@k 评测
                        │
  用户问题 ────────────> Top-k 检索
                          │  ① 片段清洗（只留解答主体）
                          │  ② 关键词相关性门（脏检索自动降级）
                          ▼
                   ┌──────────────┐
                   │ 8B CareBot（GPU，4bit）  ← 可选挂 LoRA
                   │ Qwen1.5B（CPU 发行版）    ← 可选“仅检索摘录”
                   └──────────────┘
                          │  去复读 / 去客套 / 转场截断 / 限长
                          ▼
                    回答 + 命中片段
```

工程链路一句话：`数据 → 分块 → 向量化 → 检索 → 生成 → 清理 → 评测`，每段都有脚本和口径说明。

---

## 3. 主要设计点

1. **可复现的数据管线**：蓄水池抽样 + 固定 seed；一份 CSV 同时产出知识库、评测集、SFT 格式，
   评测记录与知识库同源、带 `rid` 可回溯。
2. **字段感知分块**：按【科室】【主题】【问题】【解答】边界切分，长文不跨主题；
   分块器 14 个单元测试覆盖字段边界/句号/硬限/去重等。
3. **索引可续跑**：内容 sha1 哈希去重入库，中断重跑即续传；批量 256 条写入；`--rebuild` 可重建。
4. **模型加载可降级**：4bit/8bit 量化依赖 bitsandbytes，缺失时明确回退普通精度/CPU，
   不“假量化”；重型依赖全部延迟导入（`--help`/只检索不需要 torch）。
5. **脏检索自动降级（新增）**：注入上下文前先清洗片段，再做关键词相关性门；
   顶配命中与问题不相关时自动切换为“模型直接回答”，避免脏语料带偏答案。
6. **回答后处理（新增）**：去提示词复读、去“答案是/参考文献/医生询问”等转场截断、
   整句去重、删除“祝您康复/仅供参考/AI 身份声明”等语料客套尾巴、按句号限长。
7. **可选 reranker 精排（新增）**：`--rerank` 用 bge-reranker 对向量 top-N 候选二次精排，
   1000 问实测：Hit@1 86.5% → 91.4%（+4.9pp），端到端约 30ms/问。

8. **有对照的实验**：基座 vs LoRA 微调在留出集上比语义相似度与 Rouge；
   LoRA 更贴标准答案但把语料营销式客套学歪了 → 发行版默认“基座 + RAG”，LoRA 保留作对比开关。

---

## 4. 快速开始

### 4.1 本地 CPU（无需显卡，推荐先试）

前置：Python 3.10+；Windows 可双击 `release/安装CPU版.bat`，Linux/macOS 用命令行。

```bash
# 1. 安装 CPU 版 PyTorch 与其余依赖（torch 约 200MB，其余约几分钟）
python -m venv .venv-cpu
.venv-cpu/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu   # Windows
.venv-cpu/Scripts/pip install -r requirements-cpu.txt

# 2. 用仓库内置样例构建本地索引（首次约几分钟）
.venv-cpu/Scripts/python document_processer.py --device cpu

# 3. 启动网页（http://127.0.0.1:7861）
.venv-cpu/Scripts/python app_local.py --cpu-model Qwen/Qwen2.5-1.5B-Instruct
```

- `app_local.py` 默认用 **Qwen2.5-1.5B-Instruct 在 CPU 生成**（首次自动下载约 3GB，建议 8G 内存）；
  也可选 **“仅检索摘录”** 模式：不下载大模型，直接把最相关知识片段的解答展示出来。
- 想更轻量可换 `--cpu-model Qwen/Qwen2.5-0.5B-Instruct`（约 1GB）。

### 4.2 GPU / 云端（完整 8B 版）

```bash
pip install -r requirements.txt        # GPU 另装匹配 CUDA 的 PyTorch
python document_processer.py --device cuda
python qa_system.py --question "高血压患者能吃柚子吗？"
```

网页发行版：`bash start_release.sh`（AutoDL 上监听 7860）；
本机访问用 SSH 隧道 `release/双击打开演示.bat`（对应云端 GPU 发行版）。

### 4.3 评测与测试

```bash
python tests/test_chunking.py          # 14/14
python tests/test_data.py              # 9/9
python tests/test_prompts.py           # 5/5
python tests/test_retriever.py         # 6/6
python tests/test_inference_clean.py   # 9/9
python tests/test_rerank.py            # 6/6
python -m scripts.eval_retrieval --limit 1000 --device cuda --out data/eval_result.json
python scripts/eval_generation.py --model-dir models/CareBot_Medical \
      --lora-path outputs/lora-med-v1 --val outputs/lora-med-v1/val_split.json --limit 60
```

### 4.4 Docker 一键运行（可选，CPU 版）

```bash
docker compose up --build        # 打开 http://127.0.0.1:7861
# 或手动：
docker build -t ragqa-cpu .
docker run -p 7861:7861 -v "$PWD/models:/app/models" \
       -v "$PWD/chroma_data:/app/chroma_data" ragqa-cpu
```

> 首次启动会自动用内置样例建索引并下载 CPU 小模型（约 1–3GB）；
> 不想下载大模型可在网页里选「仅检索摘录」模式。

### 4.5 作为 Python 包安装（开发者可选）

```bash
pip install -e .            # 只装库本体与 CLI（重型依赖另行安装）
pip install -e ".[cpu]"     # 或补 CPU 版依赖（PyTorch 请先按 CPU 源安装）
ragqa-qa --help
```

---

## 5. 评测口径（诚实说明）

- **Hit@k**：评测记录从知识库内抽样，gold=`rid`，衡量“分块 → 嵌入 → 检索”整条管线的
  **端到端一致性**；命中偏高是正常现象，它不等同于“回答质量”。
- **回答质量**：目前用留出集做 基座/LoRA 语义相似度与 Rouge 对比 + 人工抽样；
  LLM-as-judge 与临床校验仍属后续计划（见 `docs/ROADMAP.md`）。
- 生成的回答会在端上做一层“去语料味”清理，数字对比如上表。

---

## 6. 项目结构

```
├── ragqa/                  # 核心库
│   ├── data.py             # CSV 编码识别 / 清洗 / 蓄水池抽样 / 三种格式
│   ├── chunking.py         # 字段感知分块 + 去重 + 记录级溯源
│   ├── retriever.py        # Chroma Top-k（返回 distance / rid）
│   ├── inference.py        # 8B 推理 + 生成后处理 + 检索清洗/相关性门
│   ├── localgen.py         # CPU 小模型（Qwen）聊天模板生成
│   ├── modelstore.py       # 模型本地化下载（镜像优先、延迟触发）
│   └── config.py / prompts.py / generator.py
├── app_demo.py             # GPU 网页发行版（基座/LoRA + RAG 开关）
├── app_local.py            # 本地 CPU 网页版（轻量生成 / 仅检索摘录）
├── qa_system.py            # CLI：REPL / 单问 / 只检索
├── document_processer.py   # 建索引（断点续传 / --rebuild / --device）
├── pyproject.toml          # 可安装包 + CLI 入口（ragqa-*）
├── Dockerfile              # CPU 版容器镜像（可选）
├── docker-compose.yml      # 一键起容器（端口 7861）
├── scripts/                # prepare_data / eval_retrieval / train_lora / eval_generation
├── tests/                  # 纯 CPU 单元测试（49 项）
├── release/                # 双击 .bat：云端隧道版 / 本地 CPU 安装与启动
├── data/                   # 仓库内置样例数据与评测结果（小文件）
└── docs/                   # EXPERIMENT / ROADMAP / CHANGELOG
```

---

## 7. 数据与授权

- 数据整理自 [Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)
  （内科等样例，仓库内置约 3.3MB；全量按科室约 45–105MB/份），以该仓库 LICENSE 为准，研究用途。
- 模型：`BAAI/CareBot_Medical_multi-llama3-8b-instruct`（GPU 主发行版）、
  `Qwen/Qwen2.5-1.5B-Instruct`（CPU 发行版，Apache-2.0）。
- 本项目代码以 MIT 协议开源（见 LICENSE）。

---

## 8. 边界与后续

- 目前知识库为 Toyhom 问答语料，不等于临床指南；回答未经临床审核。
- 后续计划（按性价比）：更大留出集的 LLM-as-judge 生成评测、端到端延迟基准、临床数据接入。
- 更新日志见 `docs/CHANGELOG.md`；英语摘要见 `README.en.md`。
