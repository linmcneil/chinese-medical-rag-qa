"""RAG 医疗问答系统入口（原 qa_system.py 的重构版）。

用法（项目根目录）：
    python qa_system.py                          # REPL 交互问答
    python qa_system.py --question "高血压能吃柚子吗？"   # 单问即答
    python qa_system.py --retrieve-only          # 只做检索看命中，不加载大模型

说明：
    - 检索走 ragqa.retriever（Chroma + bge-small-zh 嵌入）
    - 生成走 ragqa.generator（CareBot-8B，bitsandbytes 量化逻辑已修正）
    - 所有重型依赖均为延迟导入：--help/--retrieve-only 不需要装 torch/transformers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG 医疗问答系统")
    p.add_argument("--model-dir", default=str(BASE / "models"))
    p.add_argument("--chroma-dir", default=str(BASE / "chroma_data"))
    p.add_argument("--collection", default="knowledges")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--quantize", default="auto", choices=["auto", "4bit", "8bit", "none"])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--rerank", action="store_true",
                   help="可选：用 bge-reranker 对向量 top-N 二次精排（会多加载一个模型）")
    p.add_argument("--rerank-model", default="BAAI/bge-reranker-base",
                   help="cross-encoder 重排模型 id（默认 bge-reranker-base）")
    p.add_argument("--rerank-candidates", type=int, default=12,
                   help="重排前从向量库取的候选条数（需 >= --top-k）")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--question", help="单次提问后退出（否则进入交互循环）")
    p.add_argument("--retrieve-only", action="store_true",
                   help="只做检索，不加载生成模型（用于调试/演示命中）")
    return p


def _make_retriever(args):
    from ragqa.modelstore import set_hf_mirror
    from ragqa.retriever import ChromaRetriever
    set_hf_mirror()
    return ChromaRetriever(
        chroma_dir=args.chroma_dir,
        model_dir=args.model_dir,
        collection=args.collection,
        device="cpu" if args.device == "cpu" else ("cuda" if args.device == "cuda" else _auto_device()),
        top_k=args.top_k,
    ).connect()


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _candidate_k(args):
    """需要取回多少条候选：开启重排时多取一些给精排，否则就是 top_k。"""
    if getattr(args, "rerank", False):
        return max(args.rerank_candidates, args.top_k)
    return args.top_k


def _maybe_rerank(args, question, hits):
    """开启 --rerank 时对候选做 cross-encoder 精排，返回最终 top_k。"""
    if not getattr(args, "rerank", False) or not hits:
        return hits
    from ragqa.rerank import CrossEncoderReranker
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    else:
        device = _auto_device()
    reranker = CrossEncoderReranker(model_name=args.rerank_model,
                                    device=device, top_k=args.top_k)
    return reranker.rerank(question, hits)
def _print_hits(hits) -> None:
    for i, h in enumerate(hits, 1):
        src = f"[{h['rid']}] " if h.get("rid") else ""
        dist = f"{h['distance']:.3f}" if h.get("distance") is not None else "?"
        print(f"\n--- 命中 {i} (距离 {dist}) {src}---")
        print(h["document"])


def _retrieve_only_loop(args) -> None:
    retriever = _make_retriever(args)
    print(f"向量库集合 {args.collection}，共 {retriever.count()} 块。输入 exit 退出。")
    while True:
        try:
            q = input("\n请输入医疗问题：").strip()
        except EOFError:
            return
        if q.lower() == "exit":
            return
        if not q:
            continue
        hits = retriever.query(q, top_k=_candidate_k(args))
        hits = _maybe_rerank(args, q, hits)
        if not hits:
            print("未检索到相关医疗信息")
            continue
        _print_hits(hits)


def _ask_once(args, retriever, generator, question: str) -> str:
    from ragqa.prompts import build_rag_prompt, extract_answer, format_context
    hits = retriever.query(question, top_k=_candidate_k(args))
    hits = _maybe_rerank(args, question, hits)
    if not hits:
        return "未检索到相关医疗信息。"
    context = format_context([h["document"] for h in hits],
                             [h.get("rid", "") for h in hits])
    prompt = build_rag_prompt(context, question)
    result = generator(prompt)[0]["generated_text"]
    return extract_answer(result, prompt)


def _chat_loop(args) -> None:
    retriever = _make_retriever(args)
    from ragqa.generator import load_qa_pipeline
    print(f"向量库集合 {args.collection}，共 {retriever.count()} 块。")
    print("首次运行若模型缺失将自动下载（约 15GB，耗时较长）。")
    generator = load_qa_pipeline(
        model_dir=args.model_dir,
        device=args.device,
        quantize=args.quantize,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    if args.question:
        print("\n回答：", _ask_once(args, retriever, generator, args.question))
        return
    print("\n医疗问答系统已启动（输入 'exit' 退出）")
    while True:
        try:
            q = input("\n请输入医疗问题：").strip()
        except EOFError:
            print("再见！")
            return
        if q.lower() == "exit":
            print("再见！")
            return
        if not q:
            continue
        print("\n回答：", _ask_once(args, retriever, generator, q))


def main() -> None:
    args = build_parser().parse_args()
    if args.retrieve_only:
        _retrieve_only_loop(args)
    else:
        _chat_loop(args)


if __name__ == "__main__":
    main()