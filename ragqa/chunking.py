"""字段感知分块模块（纯逻辑，无重依赖，便于单元测试）。

从原 document_processer.py 中提取并重构：
- 保留 instruction/input/output/history 兼容字段
- 可选扩展字段 department/title：存在时渲染为【科室】/【主题】
- chunk_records_with_ids：分块同时保留记录 id，便于评测与溯源
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

_WHITESPACE = re.compile(r"\s+")
_FIELD_HEADER = re.compile(r"【[^】]+】")
_SENTENCE_END = re.compile(r"[。！？；]")


def clean_text(text: str) -> str:
    """把连续空白折叠为单个空格，并去掉首尾空白。"""
    if not isinstance(text, str):
        return ""
    return _WHITESPACE.sub(" ", text).strip()


def _first_nonempty(item: Dict, *keys: str) -> str:
    for k in keys:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def block_from_record(item: Dict) -> str:
    """把一条记录格式化为文本块。

    基础字段（保持与原脚本一致）：
        【问题】= instruction  【详情】= input  【解答】= output
    history 列表项格式为 [问, 答] 或 (问, 答)。
    可选扩展字段：department → 【科室】，title → 【主题】。
    """
    department = _first_nonempty(item, "department")
    title = _first_nonempty(item, "title")
    instruction = _first_nonempty(item, "instruction")
    input_text = _first_nonempty(item, "input")
    output = _first_nonempty(item, "output")
    history = item.get("history")

    parts: List[str] = []
    if department:
        parts.append("【科室】" + department)
    if title:
        parts.append("【主题】" + title)
    if instruction:
        parts.append("【问题】" + instruction)
    if input_text:
        parts.append("【详情】" + input_text)

    if isinstance(history, list) and history:
        history_parts: List[str] = []
        for h in history:
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                q = str(h[0]).strip()
                a = str(h[1]).strip()
                if q or a:
                    history_parts.append("问：" + q + "\n答：" + a)
        if history_parts:
            parts.append("【历史】" + "\n【历史对话】".join(history_parts))

    if output:
        parts.append("【解答】" + output)
    return "\n".join(parts).strip()


def chunk_text(text: str, max_chunk_size: int = 150, hard_limit: int = 250) -> List[str]:
    """按 max_chunk_size 窗口切分单个文本块。

    优先在【字段】边界切；其次是句子结束符；都没有才硬切。
    参数与原脚本语义一致：hard_limit 是强制拆分上限。
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if hard_limit < max_chunk_size:
        hard_limit = max_chunk_size

    chunks: List[str] = []
    pos = 0
    length = len(cleaned)

    while pos < length:
        remaining = length - pos

        if remaining <= max_chunk_size:
            chunk = cleaned[pos:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # 在 [pos+150, pos+250) 区间里优先找字段头，如【解答】
        field_window = cleaned[pos + max_chunk_size:pos + hard_limit]
        field_match = _FIELD_HEADER.search(field_window)
        if field_match:
            end = pos + max_chunk_size + field_match.start()
        else:
            # 其次找句子结束符
            sentence_window = cleaned[pos + max_chunk_size:min(pos + hard_limit, length)]
            sent_match = _SENTENCE_END.search(sentence_window)
            if sent_match:
                end = pos + max_chunk_size + sent_match.end()
            else:
                end = pos + max_chunk_size

        chunk = cleaned[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        pos = max(end, pos + 1)  # 防御：保证位置严格前进

    return chunks


def dedupe(chunks: Iterable[str]) -> List[str]:
    """去重并保持原有顺序。"""
    return list(dict.fromkeys(chunks))


def _iter_block_chunks(item: Dict, max_chunk_size: int, hard_limit: int) -> Iterator[str]:
    block = block_from_record(item)
    if block:
        yield from chunk_text(block, max_chunk_size, hard_limit)


def chunk_records_with_ids(dataset: Iterable[Dict],
                           max_chunk_size: int = 150,
                           hard_limit: int = 250) -> List[Tuple[Optional[str], str]]:
    """对整个数据集分块，返回 [(record_id, chunk_text), ...]。

    - record_id 取 item["id"]；没有则为 None。
    - 按文本内容去重，重复文本保留第一条的 id（与原版 dedupe 行为一致）。
    """
    seen: Dict[str, Tuple[Optional[str], str]] = {}
    order: List[str] = []
    for item in dataset:
        if not isinstance(item, dict):
            continue
        rid = item.get("id")
        rid = rid if isinstance(rid, str) and rid.strip() else None
        for chunk in _iter_block_chunks(item, max_chunk_size, hard_limit):
            if chunk not in seen:
                seen[chunk] = (rid, chunk)
                order.append(chunk)
    return [seen[k] for k in order]


def chunk_records(dataset: Iterable[Dict],
                  max_chunk_size: int = 150,
                  hard_limit: int = 250) -> List[str]:
    """兼容接口：只返回文本块列表（去重）。"""
    return [text for _, text in chunk_records_with_ids(dataset, max_chunk_size, hard_limit)]