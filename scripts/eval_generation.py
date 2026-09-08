"""生成评测：基座 vs LoRA 微调，在留出验证集上对比 Rouge-1/2/L。

用法（GPU 机器）：
    python scripts/eval_generation.py \
        --model-dir models/CareBot_Medical \
        --lora-path outputs/lora-med-v1 \
        --val outputs/lora-med-v1/val_split.json \
        --limit 60 --out outputs/gen_eval_result.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ragqa.inference import generate_answer, load_generation_model  # noqa: E402
from rouge_score import rouge_scorer  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="基座 vs LoRA 生成对比评测")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--lora-path", default=None, help="提供则加载 LoRA 评测")
    p.add_argument("--val", required=True, help="val_split.json")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--out", default=None)
    return p


def main():
    args = build_parser().parse_args()
    with open(args.val, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if args.limit:
        rng = random.Random(args.seed)
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), min(args.limit, len(rows))))]
    print(f"eval n={len(rows)}")

    def zh_tokens(t: str) -> str:
        # 中文没有天然空格分词：退化为字符级 token，保证 Rouge 有意义
        import re as _re
        t = _re.sub(r"(?<=[\u4e00-\u9fff])(?=[\u4e00-\u9fff])", " ", t)
        return t

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=False)
    model, tokenizer = load_generation_model(args.model_dir,
                                             lora_path=args.lora_path)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    records = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        ask = (row.get("input") or "").strip()
        gold = (row.get("output") or "").strip()
        pred = generate_answer(model, tokenizer, ask,
                               max_new_tokens=args.max_new_tokens)
        m = scorer.score(zh_tokens(gold), zh_tokens(pred))
        for k in scores:
            scores[k].append(m[k].fmeasure)
        records.append({"ask": ask[:80], "gold": gold, "pred": pred})
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}  elapsed {(time.time()-t0)/60:.1f}min")
    dt = (time.time() - t0) / 60.0
    summary = {k: round(100.0 * sum(v) / len(v), 2) for k, v in scores.items()}
    print("Rouge(f1%):", summary, f"elapsed={dt:.1f}min")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "n": len(rows), "records": records},
                      fh, ensure_ascii=False, indent=1)
        print("saved to", args.out)


if __name__ == "__main__":
    main()