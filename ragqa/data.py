"""数据准备模块：CSV -> 规范化记录 -> RAG / SFT JSON。

纯标准库实现（csv / json / random），便于单元测试与在无 GPU 的机器上运行。
数据源字段约定（Toyhom/Chinese-medical-dialogue-data）：
    department,title,ask,answer
"""
from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

DEFAULT_INSTRUCTION = (
    "你是一名专业的中文医疗顾问。请根据患者对病情的描述，"
    "结合医学知识给出专业、准确且负责的回答。"
)


# ---------- 编码 / 读取 ----------

def detect_encoding(path) -> str:
    """探测 CSV 编码：优先 UTF-8，回退 GB18030（该数据集实际为 GBK/GB18030）。"""
    with open(path, "rb") as fh:
        sample = fh.read(65536)
    for enc in ("utf-8-sig", "utf-8"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "gb18030"


def open_text(path) -> io.TextIOBase:
    """按探测编码打开文本文件。"""
    return open(path, "r", encoding=detect_encoding(path), newline="")


_HEADER_TERMS = {
    "department", "title", "ask", "answer", "question", "response",
    "科室", "标题", "问题", "回答", "答案", "询问", "病人", "患者",
}


def _looks_like_header(row) -> bool:
    """启发式判断首行是否为表头（兼容中文表头）。"""
    cells = [(c or "").strip() for c in row]
    if not cells:
        return False
    hits = sum(1 for c in cells if c.lower() in _HEADER_TERMS)
    # 表头通常短小；数据行里出现的关键词较少
    if hits >= 2:
        return True
    return hits >= 1 and all(len(c) <= 12 for c in cells)


def iter_csv_records(path) -> Iterator[Dict[str, str]]:
    """逐行读取问答 CSV，产出 {department,title,ask,answer} 字典。

    兼容三种情况：
    - 英文表头 department,title,ask,answer
    - 中文表头（科室/标题/问题/回答 等）
    - 无表头（按列位置推断）
    """
    with open_text(path) as fh:
        reader = csv.reader(fh)
        first = next(reader, None)
        if first is None:
            return
        header = [c.strip().lower() for c in first]
        has_header = _looks_like_header(first) or (
            "ask" in header and "answer" in header
        )
        if has_header:
            row_iter = reader
        else:
            for rec in _yield_from_row(first):
                yield rec
            row_iter = reader
        for row in row_iter:
            for rec in _yield_from_row(row, header if has_header else None):
                yield rec


def _yield_from_row(row: List[str], header: Optional[List[str]] = None) -> Iterator[Dict[str, str]]:
    row = [c or "" for c in row]
    if len(row) < 4:
        return
    if header and set(["department", "title", "ask", "answer"]).issubset(header):
        index = {name: header.index(name) for name in ("department", "title", "ask", "answer")}
        rec = {name: row[index[name]] for name in index}
    else:
        rec = {"department": row[0], "title": row[1], "ask": row[2], "answer": row[3]}
    if rec["ask"] or rec["answer"]:
        yield rec


# ---------- 清洗 / 去重 ----------

def _clean(v) -> str:
    return " ".join((v or "").split()).strip()


def normalize_records(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """清洗字段、丢弃无问题或空回答的记录。"""
    out: List[Dict[str, str]] = []
    for r in rows:
        rec = {
            "department": _clean(r.get("department")),
            "title": _clean(r.get("title")),
            "ask": _clean(r.get("ask")),
            "answer": _clean(r.get("answer")),
        }
        if rec["ask"] and rec["answer"]:
            out.append(rec)
    return out


def dedupe_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """按 (ask, answer) 完全去重，保持首次出现顺序。"""
    seen = set()
    out: List[Dict[str, str]] = []
    for r in records:
        key = (r["ask"], r["answer"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def add_ids(records: List[Dict[str, str]], prefix: str = "med") -> List[Dict[str, str]]:
    """给记录补稳定 id：{prefix}_{序号:07d}。"""
    return [
        {**r, "id": f"{prefix}_{i:07d}"}
        for i, r in enumerate(records)
    ]


# ---------- 抽样 ----------

def reservoir_sample(stream: Iterable[Dict[str, str]], k: int, seed: Optional[int] = None) -> List[Dict[str, str]]:
    """流式蓄水池抽样：无需把整份大文件读进内存，均匀抽 k 条。"""
    rng = random.Random(seed)
    pool: List[Dict[str, str]] = []
    for i, item in enumerate(stream):
        if len(pool) < k:
            pool.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                pool[j] = item
    return pool




def sample_subset(records: List[Dict[str, str]], size: int, seed: Optional[int] = None) -> List[Dict[str, str]]:
    """从记录中确定性抽取 size 条（顺序打乱，供评测使用）。"""
    if size <= 0:
        return []
    if size >= len(records):
        return list(records)
    rng = random.Random(seed)
    return [records[i] for i in rng.sample(range(len(records)), size)]

def split_eval(records: List[Dict[str, str]], eval_size: int, seed: Optional[int] = None) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """划分评测集与剩余集合（互斥）。返回 (rest, eval_set)。"""
    if eval_size <= 0:
        return records, []
    if eval_size >= len(records):
        raise ValueError(f"eval_size={eval_size} 必须小于记录总数 {len(records)}")
    rng = random.Random(seed)
    picked = set(rng.sample(range(len(records)), eval_size))
    rest = [r for i, r in enumerate(records) if i not in picked]
    eval_set = [records[i] for i in sorted(picked)]
    return rest, eval_set


# ---------- 格式转换 / 落盘 ----------

def to_rag_record(rec: Dict[str, str]) -> Dict:
    """知识库格式：保留完整结构，供分块/向量索引使用。

    instruction=患者问题(ask)，与分块器的【问题】对应；
    title/department 由分块器渲染为【主题】【科室】。
    """
    return {
        "id": rec["id"],
        "department": rec.get("department", ""),
        "title": rec.get("title", ""),
        "instruction": rec["ask"],
        "output": rec["answer"],
        "history": [],
    }


def to_sft_record(rec: Dict[str, str], instruction: Optional[str] = None) -> Dict:
    """SFT 格式（ChatGLM 风格 instruction/input/output），供后续微调用。"""
    return {
        "instruction": instruction or DEFAULT_INSTRUCTION,
        "input": rec["ask"],
        "output": rec["answer"],
        "history": [],
    }


def load_json(path) -> List:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)