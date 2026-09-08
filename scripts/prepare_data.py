"""数据准备：从 Toyhom 中文医疗对话数据构建 train/eval/SFT JSON。

用法（项目根目录执行）：
    # 快速演示：自动下载 3.4MB 样例（内科 5000-6000），train=5000 eval=1000
    python -m scripts.prepare_data --source sample

    # 全量来源：某科室 CSV（约 40~105MB），流式抽样不整读入库
    python -m scripts.prepare_data --source toyhom --dept 内科 --limit 30000 --eval-size 2000

    # 本地 CSV（GBK/UTF-8 自动识别）
    python -m scripts.prepare_data --source local --local-file data/raw/内科5000-33000.csv

产出（默认 data/ 目录）：
    train.json        知识库 JSON（供 document_processer.py 建索引）
    eval.json         检索评测集（question/answer，从 train 内抽样，id 可回溯知识库）
    sft_train.json    ChatGLM 风格微调格式（instruction/input/output）
    dataset_info.json 数据来源/规模/参数元数据
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragqa.data import (add_ids, dedupe_records, iter_csv_records,
                        normalize_records, reservoir_sample, sample_subset,
                        to_rag_record, to_sft_record, write_json)

BASE = Path(__file__).resolve().parent.parent
GITHUB_RAW = "https://raw.githubusercontent.com/Toyhom/Chinese-medical-dialogue-data/master/"

# 仓库内每科室真实文件（raw 链接逐段 URL 编码）
TOYHOM_FILES = {
    "内科": "Data_数据/IM_内科/内科5000-33000.csv",
    "儿科": "Data_数据/Pediatric_儿科/儿科5-14000.csv",
    "妇产科": "Data_数据/OAGD_妇产科/妇产科6-28000.csv",
    "外科": "Data_数据/Surgical_外科/外科5-14000.csv",
    "肿瘤科": "Data_数据/Oncology_肿瘤科/肿瘤科5-10000.csv",
    "男科": "Data_数据/Andriatria_男科/男科5-13000.csv",
}
SAMPLE_REL = "样例_内科5000-6000.csv"


def _quote_path(rel: str) -> str:
    from urllib.parse import quote
    return "/".join(quote(seg, safe="") for seg in rel.split("/"))


def toyhom_url(rel: str) -> str:
    return GITHUB_RAW + _quote_path(rel)


def download(url: str, dest: Path, max_mb: float = 0) -> Path:
    """下载到 dest，超过 max_mb(MB) 则中断（max_mb<=0 表示不限）。"""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_mb * 1024 * 1024) if max_mb > 0 else 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                total += len(chunk)
                if max_bytes and total > max_bytes:
                    print(f"[warn] 达到 --max-mb 限制，文件已截断: {dest}")
                    break
    return dest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="准备 RAG 医疗问答数据集")
    p.add_argument("--source", choices=["sample", "toyhom", "local"], default="sample")
    p.add_argument("--dept", default="内科",
                   help="toyhom 来源的科室，可选: " + "/".join(TOYHOM_FILES))
    p.add_argument("--local-file", type=Path, default=None,
                   help="local 来源的 CSV 路径")
    p.add_argument("--limit", type=int, default=5000, help="train 条数上限")
    p.add_argument("--eval-size", type=int, default=1000, help="评测集条数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=Path, default=BASE / "data")
    p.add_argument("--keep-raw", action="store_true", help="保留下载的原始 CSV")
    p.add_argument("--max-mb", type=float, default=0, help="下载上限(MB)，0=不限")
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_dir: Path = args.data_dir
    raw_dir = data_dir / "raw"

    if args.source == "toyhom" and args.dept not in TOYHOM_FILES:
        print(f"未知科室: {args.dept}，可选: {list(TOYHOM_FILES)}")
        sys.exit(1)

    # 1) 定位/下载原始 CSV
    rel = TOYHOM_FILES[args.dept] if args.source == "toyhom" else SAMPLE_REL
    if args.source == "local":
        if not args.local_file:
            print("local 来源必须提供 --local-file")
            sys.exit(1)
        csv_path = args.local_file
        if not csv_path.exists():
            print(f"本地文件不存在: {csv_path}")
            sys.exit(1)
        source_desc = f"local:{csv_path}"
    else:
        url = toyhom_url(rel)
        csv_path = raw_dir / rel.split("/")[-1]
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            print(f"[1/4] 下载 {url} -> {csv_path} ...")
            t0 = time.time()
            download(url, csv_path, args.max_mb)
            print(f"      完成，用时 {time.time() - t0:.1f}s，"
                  f"大小 {csv_path.stat().st_size / 1e6:.1f}MB")
        else:
            print(f"[1/4] 使用已下载文件 {csv_path}")
        source_desc = f"toyhom:{rel}"

    # 2) 流式抽样 limit+eval 条（无需把全文件载入内存）
    need = args.limit + args.eval_size
    print(f"[2/4] 从 CSV 流式抽样 {need} 条 (seed={args.seed}) ...")
    stream = iter_csv_records(csv_path)
    pool = reservoir_sample(stream, need, seed=args.seed)

    recs = dedupe_records(normalize_records(pool))
    if len(recs) < need:
        print(f"[warn] 抽样后有效记录仅 {len(recs)} 条 < 需求 {need}")
    recs = add_ids(recs, prefix="med")

    # 3) 知识库(train)与评测集(eval)：eval 从 train 内抽样，确保评测记录已入库
    train_recs = recs[:args.limit]
    eval_recs = sample_subset(train_recs, min(args.eval_size, len(train_recs)), seed=args.seed)
    print(f"[3/4] KB(train)={len(train_recs)} eval={len(eval_recs)}")

    # 4) 落盘三种格式
    print(f"[4/4] 写入 {data_dir} ...")
    write_json(data_dir / "train.json", [to_rag_record(r) for r in train_recs])
    write_json(data_dir / "eval.json", [{
        "id": r["id"],
        "department": r.get("department", ""),
        "title": r.get("title", ""),
        "question": r["ask"],
        "short_question": r.get("title", r["ask"]),
        "answer": r["answer"],
    } for r in eval_recs])
    write_json(data_dir / "sft_train.json", [to_sft_record(r) for r in train_recs])
    write_json(data_dir / "dataset_info.json", {
        "source": source_desc,
        "department": args.dept,
        "train_size": len(train_recs),
        "eval_size": len(eval_recs),
        "seed": args.seed,
        "license_note": "数据整理自 Toyhom/Chinese-medical-dialogue-data（研究用途），以该仓库 LICENSE 为准",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    if args.keep_raw:
        print(f"原始 CSV 保留在 {csv_path}")
    else:
        if args.source != "local" and csv_path.exists():
            try:
                csv_path.unlink()
                print("已删除临时原始 CSV（如需保留加 --keep-raw）")
            except OSError:
                pass
    print(f"完成。下一步：python document_processer.py --data {data_dir / 'train.json'}")


if __name__ == "__main__":
    main()