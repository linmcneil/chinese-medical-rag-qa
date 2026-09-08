"""检索评测：对 eval.json 的 question 做 top-k 检索，统计 Hit@k 与延迟。

评测前提：eval.json 由 scripts/prepare_data.py 生成，其记录同时存在于
向量库（train.json）中，gold = eval 记录的 id(rid)。命中判定：top-k 内
任意一块的 metadata.rid == gold rid。

用法（项目根目录）：
    python -m scripts.eval_retrieval                       # 默认 200 条快测
    python -m scripts.eval_retrieval --limit 1000 --out data/eval_result.json
    python -m scripts.eval_retrieval --device cuda

说明：
    - 首次运行会自动下载 bge-small-zh 嵌入模型（约 100MB）
    - 该指标衡量“分块-嵌入-检索”整条管线的端到端一致性，
      对原文检索命中偏高属正常；语义改写评测见 README 说明。
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="检索 Hit@k 评测")
    p.add_argument("--eval", default=str(BASE / "data" / "eval.json"))
    p.add_argument("--chroma-dir", default=str(BASE / "chroma_data"))
    p.add_argument("--model-dir", default=str(BASE / "models"))
    p.add_argument("--collection", default="knowledges")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--limit", type=int, default=200, help="评测条数（0=全部）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5, help="检索上限，需 >= 最大 k")
    p.add_argument("--out", default=None, help="结果 JSON 输出路径")
    return p


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    args = build_parser().parse_args()

    with open(args.eval, "r", encoding="utf-8") as fh:
        eval_rows = json.load(fh)
    if args.limit and args.limit < len(eval_rows):
        rng = random.Random(args.seed)
        eval_rows = [eval_rows[i] for i in sorted(rng.sample(range(len(eval_rows)), args.limit))]

    from ragqa.modelstore import set_hf_mirror
    from ragqa.retriever import ChromaRetriever
    set_hf_mirror()
    retriever = ChromaRetriever(
        chroma_dir=args.chroma_dir,
        model_dir=args.model_dir,
        collection=args.collection,
        device=_resolve_device(args.device),
        top_k=args.top_k,
    ).connect()
    corpus = retriever.count()
    print(f"collection={args.collection} corpus_chunks={corpus} eval_n={len(eval_rows)}")

    # 预热：先检索一次，把嵌入模型加载/首次查询开销排除在计时之外
    try:
        retriever.query("预热", top_k=1)
    except Exception:
        pass

    ks = [1, 3, 5]
    hit_count = {k: 0 for k in ks}
    dept_hit = {}   # department -> [hit@1, hit@5, n]
    latencies = []
    errors = 0

    t0 = time.time()
    for row in eval_rows:
        gold = row.get("id", "")
        q = (row.get("question") or "").strip()
        if not q:
            continue
        ts = time.time()
        hits = retriever.query(q, top_k=args.top_k)
        latencies.append(time.time() - ts)
        rids = [h.get("rid", "") for h in hits]

        for k in ks:
            if gold and gold in rids[:k]:
                hit_count[k] += 1

        dept = row.get("department") or "未知"
        d = dept_hit.setdefault(dept, [0, 0, 0])
        d[2] += 1
        if gold and gold in rids[:1]:
            d[0] += 1
        if gold and gold in rids[:5]:
            d[1] += 1

    elapsed = time.time() - t0
    n = len(eval_rows)

    def pct(v):
        return f"{100.0 * v / n:.2f}%"

    print("\n===== Hit@k（原文检索命中）=====")
    for k in ks:
        print(f"Hit@{k:<2} {hit_count[k]:>5}/{n}  {pct(hit_count[k])}")
    print(f"\n平均单次查询延迟: {1000.0 * sum(latencies) / max(1, len(latencies)):.1f} ms "
          f"({len(latencies)} 次)")
    if dept_hit:
        print("\n===== 分科室 Hit@1 / Hit@5（占本科室比例）=====")
        for dept, (h1, h5, cnt) in sorted(dept_hit.items()):
            if cnt == 0:
                continue
            print(f"{dept:<8} Hit@1 {100.0 * h1 / cnt:6.2f}%  Hit@5 {100.0 * h5 / cnt:6.2f}%  n={cnt}")

    if args.out:
        out = {
            "eval_file": args.eval,
            "n": n,
            "corpus_chunks": corpus,
            "hit": {str(k): hit_count[k] for k in ks},
            "mean_query_sec": (sum(latencies) / max(1, len(latencies))) if latencies else None,
            "elapsed_sec": round(elapsed, 3),
            "by_department": dept_hit,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()