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
