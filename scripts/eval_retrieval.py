"""检索评测：对 eval.json 的 question 做 top-k 检索，统计 Hit@k 与延迟。

评测前提：eval.json 由 scripts/prepare_data.py 生成，其记录同时存在于
向量库（train.json）中，gold = eval 记录的 id(rid)。命中判定：top-k 内
任意一块的 metadata.rid == gold rid。

用法（项目根目录）：
    python -m scripts.eval_retrieval                          # 默认 200 条快测
    python -m scripts.eval_retrieval --limit 1000 --out data/eval_result.json
    python -m scripts.eval_retrieval --device cuda
    python -m scripts.eval_retrieval --rerank --device cuda   # 对比 向量 Top-k vs +bge-reranker

说明：
    - 首次运行会自动下载 bge-small-zh 嵌入模型（约 100MB）
    - --rerank 会额外下载 bge-reranker-base（约 1GB）并对向量候选做二次精排，
      输出里会同时给出“纯向量 Top-k”与“向量+重排”两套 Hit@k 便于对照
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

KS = [1, 3, 5]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="检索 Hit@k 评测（可选 reranker 对照）")
    p.add_argument("--eval", default=str(BASE / "data" / "eval.json"))
    p.add_argument("--chroma-dir", default=str(BASE / "chroma_data"))
    p.add_argument("--model-dir", default=str(BASE / "models"))
    p.add_argument("--collection", default="knowledges")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--limit", type=int, default=200, help="评测条数（0=全部）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5, help="最终保留条数（需 <= --rerank-candidates）")
    p.add_argument("--rerank", action="store_true", help="开启 bge-reranker 二次精排对照")
    p.add_argument("--rerank-model", default="BAAI/bge-reranker-base",
                   help="cross-encoder 重排模型 id（默认 bge-reranker-base）")
    p.add_argument("--rerank-candidates", type=int, default=20,
                   help="重排前从向量库取的候选条数（需 >= --top-k）")
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


def _new_stats() -> dict:
    return {"hit": {k: 0 for k in KS}, "latencies": [], "dept_hit": {}}


def _accumulate(stats: dict, rids, gold: str, dept: str) -> None:
    """rids 为已按最终顺序排列的命中 rid 列表（截断到 top-k 之后）。"""
    if not gold:
        return
    for k in KS:
        if gold in rids[:k]:
            stats["hit"][k] += 1
    d = stats["dept_hit"].setdefault(dept, [0, 0, 0])
    d[2] += 1
    if gold in rids[:1]:
        d[0] += 1
    if gold in rids[:KS[2]]:
        d[1] += 1
    return


def _print_stats(title: str, stats: dict, n: int,
                 latency_label: str = "平均单次查询延迟", extra: str = "") -> None:
    print(f"\n===== {title} =====")
    for k in KS:
        v = stats["hit"][k]
        print(f"Hit@{k:<2} {v:>5}/{n}  {100.0 * v / n:.2f}%")
    lat = stats["latencies"]
    if lat:
        print(f"{latency_label}: {1000.0 * sum(lat) / len(lat):.1f} ms ({len(lat)} 次){extra}")
    dh = stats["dept_hit"]
    if dh:
        print("分科室 Hit@1 / Hit@5（占本科室比例）:")
        for dept, (h1, h5, cnt) in sorted(dh.items()):
            if cnt == 0:
                continue
            print(f"  {dept:<8} Hit@1 {100.0 * h1 / cnt:6.2f}%  Hit@5 {100.0 * h5 / cnt:6.2f}%  n={cnt}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.top_k < KS[-1]:
        parser.error(f"--top-k 至少应为 {KS[-1]}（当前 {args.top_k}）")
    device = _resolve_device(args.device)
    use_rerank = args.rerank
    if use_rerank and args.rerank_candidates < args.top_k:
        args.rerank_candidates = args.top_k
    retrieve_k = max(args.rerank_candidates, args.top_k) if use_rerank else args.top_k

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
        device=device,
        top_k=retrieve_k,
    ).connect()
    corpus = retriever.count()
    print(f"collection={args.collection} corpus_chunks={corpus} eval_n={len(eval_rows)}"
          f" device={device} rerank={use_rerank}")

    # 预热：先检索一次，把嵌入模型加载/首次查询开销排除在计时之外
    try:
        retriever.query("预热", top_k=1)
    except Exception:
        pass

    reranker = None
    if use_rerank:
        from ragqa.rerank import CrossEncoderReranker
        print(f"加载 reranker: {args.rerank_model} (device={device}) ...")
        reranker = CrossEncoderReranker(model_name=args.rerank_model,
                                        device=device, top_k=args.top_k)
        reranker.rerank("预热", [{"document": "预热文本"}])

    baseline = _new_stats()
    rerank_stats = _new_stats() if use_rerank else None
    t0 = time.time()

    for row in eval_rows:
        gold = row.get("id", "")
        q = (row.get("question") or "").strip()
        if not q:
            continue
        dept = row.get("department") or "未知"
        ts = time.time()
        hits = retriever.query(q, top_k=retrieve_k)
        baseline["latencies"].append(time.time() - ts)
        _accumulate(baseline, [h.get("rid", "") for h in hits[:args.top_k]], gold, dept)

        if use_rerank:
            ts2 = time.time()
            reranked = reranker.rerank(q, hits)
            rerank_stats["latencies"].append(time.time() - ts2)
            _accumulate(rerank_stats, [h.get("rid", "") for h in reranked], gold, dept)
    elapsed = time.time() - t0
    n = max(1, len(eval_rows))

    _print_stats("Hit@k（纯向量 Top-k）", baseline, n)
    if rerank_stats is not None:
        vec_lat = baseline["latencies"]
        vec_ms = (1000.0 * sum(vec_lat) / len(vec_lat)) if vec_lat else 0.0
        rr_lat = rerank_stats["latencies"]
        rr_ms = (1000.0 * sum(rr_lat) / len(rr_lat)) if rr_lat else 0.0
        _print_stats("Hit@k（向量 Top-%d + %s 重排）" % (args.rerank_candidates,
                                                        args.rerank_model.rsplit("/", 1)[-1]),
                     rerank_stats, n,
                     latency_label="平均重排耗时(不含向量检索)",
                     extra=f"；向量检索平均 {vec_ms:.1f} ms/次，端到端合计约 {vec_ms + rr_ms:.1f} ms/次")
    print(f"\n总耗时 {elapsed:.1f}s")

    if args.out:
        def _stats_to_json(st):
            return {
                "hit": {str(k): st["hit"][k] for k in KS},
                "mean_query_sec": (sum(st["latencies"]) / len(st["latencies"])) if st["latencies"] else None,
                "by_department": st["dept_hit"],
            }
        out = {
            "eval_file": args.eval,
            "n": len(eval_rows),
            "corpus_chunks": corpus,
            "device": device,
            "vector": _stats_to_json(baseline),
            "elapsed_sec": round(elapsed, 3),
        }
        if rerank_stats is not None:
            vec_sec = out["vector"]["mean_query_sec"]
            rj = _stats_to_json(rerank_stats)
            rj["model"] = args.rerank_model
            rj["candidates"] = args.rerank_candidates
            rj["mean_rerank_sec"] = rj.pop("mean_query_sec")
            rj["mean_vector_sec"] = vec_sec
            if vec_sec is not None and rj["mean_rerank_sec"] is not None:
                rj["mean_total_sec"] = round(vec_sec + rj["mean_rerank_sec"], 6)
            out["rerank"] = rj
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()