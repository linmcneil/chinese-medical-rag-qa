# LoRA 微调实验报告（2026-09-08，AutoDL RTX 5090）

## 目标
验证“在 5000 条医疗问答上做 LoRA 指令微调”能否让 CareBot-8B 更贴合该数据域的答法。
这是本仓库的第二个量化实验；第一个是检索 Hit@k（见 README）。

## 实验设置
- 基座：BAAI/CareBot_Medical_multi-llama3-8b-instruct（4bit NF4 量化）
- 数据：5000 条 Toyhom 内科等医疗问答（与向量库同源），留出 200 条不参与训练
- 方法：QLoRA（r=16, alpha=32, dropout=0.05，全线性层）；lr=2e-4，batch=2×grad16
- 训练：2 epoch，约 300 步，RTX 5090 上 24.9 分钟；final train_loss ≈ 0.84
- 产出：outputs/lora-med-v1/（adapter 168MB）
- 复现：python scripts/train_lora.py --model-dir models/CareBot_Medical \
      --data data/sft_train.json --output outputs/lora-med-v1 --epochs 2

## 结果（40 条留出样本，temperature=0.3）
| 指标 | 基座 CareBot | +LoRA 微调 | 说明 |
| --- | --- | --- | --- |
| bge 语义相似度（与标准答案） | 0.789 | **0.851** | +6.3%，主指标 |
| Rouge-1（字符级） | 1.25 | 5.46 | 参考指标，绝对量低属正常 |
| 平均回答长度 | 390 字 | 262 字 | 标准答案约 145 字，微调后更贴近 |

对比脚本：python scripts/eval_generation.py（需 models 与 lora 权重）
语义相似度用 bge-small-zh 编码“预测/标准答案”算余弦，见 outputs/semantic_summary.json。

## 结论与边界
- 小数据量 LoRA（2 epoch、24.9min）在该域明显提升了“语义贴近标准答案”，并顺带更简洁；
- 局限：只有 40 条生成样本、单次采样；Rouge 对自由生成的中文参考意义有限；
  医疗安全性评估与更大规模留出集仍未做，仅限学习研究，不构成诊疗建议。
## 结论修订（第 4 轮，上线前）
网页演示默认改用「基座 CareBot + RAG」：LoRA 虽然 bge 语义更贴近标准答案，
但它同时把语料里“祝您早日康复/以上意见仅供参考/积极面对疾病”等营销式客套尾巴学成了说话习惯，
观感差；推理端已加去重、客套清理与检索相关性降级保护。LoRA 在 UI 中保留为“实验对比”开关。

## 补充实验：reranker 重排对照（2026-09-08，AutoDL RTX 5090）

验证「Chroma 向量 Top-20 候选 → bge-reranker-base 精排到最终 top-5」能否提升原文命中。

### 设置
- 评测：`data/eval.json` 按 seed=42 抽 1000 问；gold = 记录 rid；命中 = 最终 top-k 内包含 gold
- 首筛：bge-small-zh + Chroma（8524 块知识库），召回 Top-20
- 精排：BAAI/bge-reranker-base（cross-encoder，GPU），截断到 top-5

### 结果（1000 问）
| 方案 | Hit@1 | Hit@3 | Hit@5 | 延迟/问 |
| --- | --- | --- | --- | --- |
| 纯向量 Top-k | 86.5% | 90.3% | 91.1% | 5.1 ms |
| Top-20 + bge-reranker-base | **91.4%** | **92.3%** | **92.3%** | +25.1 ms（合计约 30.3 ms） |

Hit@1 提升 **+4.9pp**；大科室提升最明显，如消化科 Hit@1 86.6% → 94.5%。

### 复现
```bash
python -m scripts.eval_retrieval --limit 1000 --device cuda --rerank --out data/eval_result_rerank.json
```

### 结论
- cross-encoder 精排以约 25ms/问的代价换来 +4.9pp 的 Hit@1，适合对准确率敏感的场景；
- 向量首筛仍是吞吐主力（Top-20 只要 5.1ms），rerank 只在最终少量候选上做二次打分，端到端约 30.3ms/问。