"""Unified CLI. --help, lexical retrieval and extraction need no ML dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragqa.config import PipelineConfig
from ragqa.data import read_documents, write_json, write_split


def _runtime_options(parser):
    parser.add_argument("--index", required=True, help="v2 immutable index directory")
    parser.add_argument("--generator", choices=["extractive", "transformers"], default="extractive")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", help="HF model revision; pin a commit for reproducible experiments")
    parser.add_argument("--quantize", choices=["none", "4bit"], default="none")
    parser.add_argument("--adapter")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--rerank-revision")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="Uncalibrated cosine threshold, not a medical confidence score",
    )
    parser.add_argument("--min-rerank-score", type=float)
    parser.add_argument("--max-context-chars", type=int, default=1600)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="中文医疗 RAG 研究系统 v2")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build a new immutable index")
    build.add_argument("--data", required=True)
    build.add_argument("--encoding", default="utf-8-sig")
    build.add_argument("--index", required=True)
    build.add_argument("--backend", choices=["lexical", "chroma"], default="lexical")
    build.add_argument("--chunk-chars", type=int, default=400)
    build.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    build.add_argument("--embedding-revision")
    build.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    split = commands.add_parser("split", help="Persist a grouped SFT train/validation split")
    split.add_argument("--data", required=True)
    split.add_argument("--encoding", default="utf-8-sig")
    split.add_argument("--out", required=True)
    split.add_argument("--val-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    make_eval = commands.add_parser("make-eval", help="Generate an explicitly labeled self-retrieval check")
    make_eval.add_argument("--data", required=True)
    make_eval.add_argument("--encoding", default="utf-8-sig")
    make_eval.add_argument("--out", required=True)
    for name in ("ask", "serve", "ui", "eval"):
        cmd = commands.add_parser(name)
        _runtime_options(cmd)
        if name == "ask":
            cmd.add_argument("--question", required=True)
        elif name in ("serve", "ui"):
            cmd.add_argument("--host", default="127.0.0.1")
            cmd.add_argument("--port", type=int, default=8000 if name == "serve" else 7861)
        else:
            cmd.add_argument("--dataset", required=True)
            cmd.add_argument("--out", required=True)
            cmd.add_argument("--retrieval-only", action="store_true")
    train = commands.add_parser("train", help="Optional CUDA QLoRA experiment")
    train.add_argument("--split", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--revision")
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=float, default=2)
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--seed", type=int, default=42)
    return root


def _pipeline(args):
    from ragqa.factory import create_pipeline

    options = {
        name: getattr(args, name)
        for name in (
            "generator",
            "device",
            "model",
            "revision",
            "quantize",
            "adapter",
            "rerank",
            "rerank_model",
            "rerank_revision",
            "max_new_tokens",
        )
    }
    config = PipelineConfig(
        candidates=args.candidates,
        top_k=args.top_k,
        min_score=args.min_score,
        min_rerank_score=args.min_rerank_score,
        max_context_chars=args.max_context_chars,
    )
    return create_pipeline(args.index, config=config, **options)


def _new_report(path: str) -> None:
    if Path(path).exists():
        raise FileExistsError("Choose a new output filename; previous reports are preserved")


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "build":
            from ragqa.retrieval import build_index

            report = build_index(
                read_documents(args.data, args.encoding),
                args.index,
                backend=args.backend,
                max_chars=args.chunk_chars,
                embedding_model=args.embedding_model,
                embedding_revision=args.embedding_revision,
                device=args.device,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "split":
            write_split(read_documents(args.data, args.encoding), args.out, args.val_fraction, args.seed)
            print(f"Saved fixed split: {args.out}")
        elif args.command == "make-eval":
            from ragqa.evaluation import consistency_dataset

            _new_report(args.out)
            write_json(args.out, consistency_dataset(read_documents(args.data, args.encoding)))
        elif args.command == "train":
            from ragqa.training import train

            train(
                args.split,
                args.model,
                args.out,
                revision=args.revision,
                epochs=args.epochs,
                max_length=args.max_length,
                seed=args.seed,
            )
        else:
            pipeline = _pipeline(args)
            if args.command == "ask":
                result = pipeline.run(args.question)
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
                return 1 if result.status == "error" else 0
            if args.command == "serve":
                import uvicorn
                from ragqa.api import create_app

                uvicorn.run(create_app(pipeline), host=args.host, port=args.port)
            elif args.command == "ui":
                from ragqa.ui import build_ui

                build_ui(pipeline).launch(server_name=args.host, server_port=args.port, share=False)
            elif args.command == "eval":
                from ragqa.evaluation import evaluate
                from ragqa.retrieval import read_index

                _new_report(args.out)
                dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
                manifest, chunks = read_index(args.index)
                report = evaluate(
                    pipeline,
                    dataset,
                    manifest,
                    {c.document_id for c in chunks},
                    end_to_end=not args.retrieval_only,
                )
                report["runtime"] = {k: v for k, v in vars(args).items() if k not in ("out",)}
                write_json(args.out, report)
                print(
                    json.dumps(
                        {k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2
                    )
                )
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
