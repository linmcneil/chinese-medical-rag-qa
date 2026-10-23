# 从 v0.1 迁移至 v0.2

基线提交：`64404c1d58a6952c9169b95cb70d31a225815d2c`。本次是破坏性接口重构，旧版本仍可通过该提交检出。不要将新代码直接指向旧 `chroma_data/`。

| 旧功能 | 新入口 |
| --- | --- |
| `document_processer.py` | `python -m ragqa build --data data/train.json --index artifacts/medical-v2` |
| `qa_system.py` | `python -m ragqa ask --index artifacts/medical-v2 --question "问题"` |
| `app_local.py` / `app_demo.py` | `python -m ragqa ui --index ...`，模型由统一参数选择 |
| 检索评测脚本 | `make-eval` 创建明确标注的自检文件，`eval` 执行 |
| FastAPI | `python -m ragqa serve --index ...` |
| 数据准备 / SFT 划分 | 显式本地文件输入；`split --data ... --out ...` |
| QLoRA | `train --split ... --model ... --out ...` |

保留的根目录及 scripts 包装脚本仅转发到新 CLI；旧参数不保证兼容，请使用 `--help`。旧模型下载镜像逻辑、独立的 CPU/GPU 提示词和语义删句逻辑已移除。

## 历史结果

`data/eval_result*.json`、`data/outputs/semantic_summary.json` 和原实验说明原样保留。它们是 v0.1 的历史材料，不是 v0.2 的有效基准。

已确认的问题包括：SFT 输入重复拼接前缀并将其纳入监督、保存的验证分区与实际训练分区不一致、中文 ROUGE 的默认 tokenizer 过滤汉字、回答清理误删注意事项、网页零距离值被 `or 1.0` 替换。

原 Hit@k 是知识库原问题回查，不能重新命名为真实新问题准确率。新报告单独写入 `artifacts/`，拒绝覆盖已有文件。

## 操作顺序

1. 在独立分支/目录安装新环境，先运行合成数据演示和测试。
2. 使用保留的原数据建立新索引；审核被拒绝的坏记录，不靠静默过滤“修好”数据。
3. 生成原问题回查自检，再建立人工改写、负例和不可回答问题集。
4. 再安装真实模型后端，固定模型版本、阈值和硬件记录后开展实验。
5. 重新划分 SFT 数据并重训；旧 adapter 不作为新实现的已验证模型。
