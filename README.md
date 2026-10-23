# 中文医疗 RAG QA · v2 重构

面向学习研究的中文医疗资料检索系统。统一链路：**读取资料 → 答案分块 → 检索 → 可选重排 → 逐条证据筛选 → 摘录/生成 → 来源编号检查**。CLI、FastAPI、Gradio 使用同一个 `RAGPipeline`。

本项目不是医疗产品：历史问答未经临床审核，引用编号和相似度不能证明医学正确性。资料不足、检索故障或来源编号校验失败时，系统不会自动切换成无资料的模型回答。

## 本次重构解决什么

- SFT 前缀只出现一次，只监督答案及 EOS；按序列长度生成 attention mask。
- 按规范化问题分组，一次划分并保存训练/验证集、ID 和哈希；训练时不重新合并划分。
- 每个块都包含答案正文，保留原问题、来源和字符偏移；重复答案不会丢失其他记录的来源。
- 检索统一使用 cosine **similarity**（越大越相关），逐条筛选；距离为零的结果不再被误判。
- 回答清理仅整理首尾空白，保留注意事项、否定、拒答和不确定性。
- 相同入口共享模型实例、提示词、筛选策略和结构化返回；单进程限制一个在途请求。
- 中文 ROUGE 使用显式汉字 token；评测区分原问题回查、人工整理评测和不可回答样本。
- 索引和切分产物写入新目录，记录语料指纹、参数及模型版本，拒绝原地覆盖。

**旧实验结果仍保存在 `data/` 和 `docs/EXPERIMENT.md`，不代表 v2 效果。** 旧 LoRA 训练和评测存在已定位的问题，必须修复后重训重测。迁移细节见 [docs/MIGRATION.md](docs/MIGRATION.md)，设计与限制见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 先在 Windows 跑通：不下载模型

Python 3.10–3.12。下面在 PowerShell 中执行，项目示例位置为 `D:\code\chinese-medical-rag-qa`：

```powershell
cd D:\code\chinese-medical-rag-qa
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[api,dev]"
.\.venv\Scripts\python.exe -m ragqa build --data examples/knowledge.json --index artifacts/demo-v2
.\.venv\Scripts\python.exe -m ragqa ask --index artifacts/demo-v2 --question "演示甲的资料编号是什么？"
.\.venv\Scripts\python.exe -m ragqa serve --index artifacts/demo-v2
```

打开 http://127.0.0.1:8000/docs 。在 `POST /v1/answers` 提交：

```json
{"question": "演示甲的资料编号是什么？"}
```

默认使用**汉字二元组/英文词 cosine 检索 + 原文摘录**。这是离线可运行的基线，不能与旧 BGE 向量检索成绩直接比较。`examples/knowledge.json` 是明确标记的合成数据，不含医学结论。

Linux/macOS 使用 `.venv/bin/python`。安装后也可用 `ragqa` 命令；所有功能都能通过 `python -m ragqa --help` 查看。

已有原始数据可直接转换为新索引，不覆盖旧数据：

```powershell
.\.venv\Scripts\python.exe -m ragqa build --data data/train.json --index artifacts/medical-lexical-v2
```

JSON 支持新格式 `question/answer`、旧 RAG 格式 `instruction/output` 和旧 SFT 格式 `input/output`。CSV 支持 Toyhom 的四列及中英文表头，GBK 数据须明确传 `--encoding gb18030`。遇到空问题、空答案或损坏记录会报告错误，不静默跳过。

## 网页

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\python.exe -m ragqa ui --index artifacts/demo-v2
```

打开 http://127.0.0.1:7861 。网页显示回答、实际使用的来源、状态和分阶段耗时。默认仅监听本机，不建立公开分享链接。

## BGE 向量检索与重排

先按 CPU 或 CUDA 环境安装 PyTorch，再安装 `.[dense]`。构建向量索引时必须固定模型的完整 Hugging Face commit SHA；查询端读取索引 manifest，使用同一模型和版本。这样更换模型不会悄悄污染旧索引。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dense]"
# 查询模型的当前 commit，把打印出的完整 SHA 填入下一条命令。
.\.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; print(HfApi().model_info('BAAI/bge-small-zh-v1.5').sha)"
.\.venv\Scripts\python.exe -m ragqa build --data data/train.json --index artifacts/medical-bge-v2 --backend chroma --embedding-revision MODEL_COMMIT_SHA
.\.venv\Scripts\python.exe -m ragqa ask --index artifacts/medical-bge-v2 --question "你的问题" --rerank
```

有 CUDA 时可加 `--device cuda`。重排器在同一进程中复用；可通过 `--rerank-model`、`--rerank-revision` 固定版本。

`--min-score` 默认 0.25，**只是尚未校准的起点**。它不是置信度，不应在不同检索后端之间直接迁移。应在独立开发集上选择阈值，再在未参与调参的测试集报告效果。重排阈值使用独立的 `--min-rerank-score`。

## 接入生成模型

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[llm]"
.\.venv\Scripts\python.exe -m ragqa ui --index artifacts/medical-bge-v2 --generator transformers --model Qwen/Qwen2.5-1.5B-Instruct
```

生成是可选项；第一次有效请求才加载模型。默认贪心解码。GPU 可用 `--device cuda`；4bit 还需安装匹配环境的 bitsandbytes，再加 `--quantize 4bit`。不会静默切换设备或量化方案。

模型 tokenizer 必须提供 chat template 与 EOS。超出上下文 token 预算时返回错误，不截掉用户问题或证据。`--adapter` 仅用于重新训练的 v2 实验 adapter，不能继续沿用旧结果。CPU 模型速度和内存取决于模型与机器，本次不宣称真实模型的延迟或质量提升。

## 评测

```powershell
.\.venv\Scripts\python.exe -m ragqa make-eval --data examples/knowledge.json --out artifacts/self-check.json
.\.venv\Scripts\python.exe -m ragqa eval --index artifacts/demo-v2 --dataset artifacts/self-check.json --out artifacts/self-check-result.json
```

自动生成的评测文件始终标为 `self_retrieval`：它检查索引一致性，不是留出集泛化成绩。报告含 document-level Hit@1/3/5、Recall@k、候选范围内 MRR、逐条结果及端到端 p50/p95。文档排名会合并同一文档的多个块；最多考察 `--candidates` 个检索块中的文档。

人工整理评测格式与分组原则见 [docs/EVALUATION.md](docs/EVALUATION.md)。不可回答样本参与拒答率统计，不混入可回答样本的检索分母。请求错误单独统计，不能充当成功拒答。`--retrieval-only` 不调用生成器。

## 可选 QLoRA 实验

```powershell
.\.venv\Scripts\python.exe -m ragqa split --data data/train.json --out artifacts/split-v2 --seed 42
# 仅 CUDA 环境，先安装匹配的 PyTorch。
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
.\.venv\Scripts\python.exe -m ragqa train --split artifacts/split-v2 --model MODEL_ID --revision MODEL_COMMIT_SHA --out artifacts/lora-v2
```

训练是无检索上下文的 answer-only SFT 对照实验，不能单独证明 RAG 的依据忠实性。超过长度的样本不截断训练，会记录在 `tokenized_manifest.json` 的 `skipped_ids`。保存的 `validation.json` 始终与训练组互斥。近义改写、相同病例的其他提问仍须在数据整理阶段做语义分组，本工具只保证规范化相同问题不跨集。

## 测试与部署

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check ragqa tests
.\.venv\Scripts\python.exe -m build
```

核心/API 测试不下载模型。可选 Chroma 测试使用真实数据库和合成向量；可选模型适配测试使用本地随机小模型，不代表医学效果。CI 在 Linux/Windows 上跑核心/API 测试，另有 Chroma 集成任务。

```bash
docker compose up --build
```

Docker 默认提供合成样例的离线 API，监听本机 8000，不自动下载大模型。尚未实现账号、鉴权、租户隔离、流式生成、多进程模型服务或临床审核；这是本地研究服务。每个进程最多一个在途请求，忙时 API 返回 503 和 `Retry-After`。

## 数据与许可

旧样例来源：[Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)。代码 MIT；数据、模型的授权分别遵循各自来源，不能由本仓库代码许可推断数据或模型的使用权限。
