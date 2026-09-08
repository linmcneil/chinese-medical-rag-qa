"""构建向量索引：数据集 -> 字段感知分块 -> 批量嵌入入库（支持断点续跑）。

用法（项目根目录）：
    python document_processer.py                         # 默认 data/train.json
    python document_processer.py --data data/train.json --device cpu
    python document_processer.py --rebuild               # 强制清空重建

说明：
    - 数据为 data/prepare_data.py 产出的 train.json（instruction/input/output）
    - 分块逻辑见 ragqa/chunking.py（已配单元测试）
    - 每块记录内容 sha1 + 所属记录 id(rid)，重复运行按 sha1 跳过，不会重复嵌入
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from ragqa.chunking import chunk_records_with_ids
from ragqa.modelstore import ensure_embedding_model, set_hf_mirror

BASE = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("ragqa.build_index")

BATCH_SIZE = 256


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG 医疗问答：构建向量索引")
    p.add_argument("--data", default=str(BASE / "data" / "train.json"),
                   help="问答数据集 JSON 路径（默认 data/train.json）")
    p.add_argument("--model-dir", default=str(BASE / "models"),
                   help="模型存放目录（默认 models/）")
    p.add_argument("--chroma-dir", default=str(BASE / "chroma_data"),
                   help="向量库目录（默认 chroma_data/）")
    p.add_argument("--collection", default="knowledges")
    p.add_argument("--chunk-size", type=int, default=150)
    p.add_argument("--hard-limit", type=int, default=250)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--rebuild", action="store_true",
                   help="先清空旧集合再重建（默认是增量续跑）")
    return p


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_records(data_path: Path):
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        log.error("找不到数据集：%s", data_path)
        log.error("先用 scripts/prepare_data.py 生成，或用 --data 指定路径")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.error("数据集不是合法 JSON：%s", e)
        sys.exit(1)
    if not isinstance(data, list):
        log.error("数据集顶层应为 JSON 数组（问答对列表）")
        sys.exit(1)
    return data


def chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    set_hf_mirror()
    device = resolve_device(args.device)

    records = load_records(Path(args.data))
    model_path = ensure_embedding_model(Path(args.model_dir))

    chunked = chunk_records_with_ids(records, args.chunk_size, args.hard_limit)
    if not chunked:
        log.error("数据集为空或分块结果为空，终止")
        sys.exit(1)
    log.info("数据集共 %d 条记录 -> %d 个文本块", len(records), len(chunked))

    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=args.chroma_dir)
    if args.rebuild:
        try:
            client.delete_collection(args.collection)
            log.info("已清空旧集合 %s（--rebuild）", args.collection)
        except Exception:
            pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=str(model_path), device=device,
    )
    collection = client.get_or_create_collection(
        name=args.collection, embedding_function=embed_fn,
    )

    existing: set = set()
    try:
        got = collection.get(include=["metadatas"])
        for meta in got.get("metadatas") or []:
            h = (meta or {}).get("h")
            if isinstance(h, str):
                existing.add(h)
    except Exception as e:
        log.warning("读取已有集合元数据失败（按全新库处理）：%s", e)

    pending = []
    for rid, text in chunked:
        h = chunk_hash(text)
        if h in existing:
            continue
        pending.append((rid, h, text))
    log.info("需要新入库 %d / %d 块", len(pending), len(chunked))

    added = 0
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        try:
            collection.add(
                ids=[f"{r}_{h[:12]}" if r else f"chunk_{h[:16]}" for r, h, _ in batch],
                documents=[t for _, _, t in batch],
                metadatas=[{"rid": r or "", "h": h, "len": len(t)} for r, h, t in batch],
            )
            added += len(batch)
            log.info("已入库 %d / %d", added, len(pending))
        except Exception as e:
            log.error("批量入库失败（%s）于第 %d 块起，请修复后重跑以续传", e, start)
            sys.exit(1)

    log.info("完成：本次新增 %d 块，集合内总块数约 %d", added, len(existing) + added)


if __name__ == "__main__":
    main()