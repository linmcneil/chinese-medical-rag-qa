"""chunking 模块单元测试（纯 CPU、无第三方依赖）。

运行方式：python tests/test_chunking.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragqa.chunking import (block_from_record, chunk_records, chunk_text,
                            chunk_records_with_ids, clean_text, dedupe)


def test_clean_text_collapses_whitespace():
    assert clean_text(" 你好  世界\n测试 ") == "你好 世界 测试"


def test_empty_and_blank_text():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_single_chunk():
    short = "短文本"
    assert chunk_text(short) == [short]


def test_field_boundary_respected():
    text = "a" * 200 + "【解答】" + "b" * 200
    chunks = chunk_text(text)
    assert chunks[0] == "a" * 200, "应在【解答】字段前断开"
    assert chunks[1].startswith("【解答】")


def test_sentence_boundary_respected():
    text = "x" * 160 + "。" + "y" * 100
    chunks = chunk_text(text)
    assert chunks[0] == "x" * 160 + "。", "应在句号后断开"
    assert chunks[1] == "y" * 100


def test_fixed_cut_when_no_break():
    chunks = chunk_text("z" * 400)
    assert [len(c) for c in chunks] == [150, 150, 100]


def test_no_overflow_over_hard_limit():
    # 没有字段与句号时，任意块的候选切点不超过 150+字符窗口
    text = "y" * 10000
    chunks = chunk_text(text)
    assert all(len(c) <= 150 for c in chunks[:-1])


def test_dedupe_keeps_order():
    assert dedupe(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_block_from_record_formats_fields():
    record = {
        "instruction": "用户问题",
        "input": "补充描述",
        "output": "医生解答",
        "history": [["之前吃过药", "效果一般"]],
    }
    block = block_from_record(record)
    assert block.startswith("【问题】用户问题")
    assert "【详情】补充描述" in block
    assert "【解答】医生解答" in block
    assert "问：之前吃过药" in block and "答：效果一般" in block


def test_block_from_record_missing_fields_ok():
    block = block_from_record({"instruction": "只有问题"})
    assert block == "【问题】只有问题"
    assert block_from_record({}) == ""


def test_block_includes_department_title():
    block = block_from_record({
        "department": "内科",
        "title": "高血压能吃柚子吗",
        "instruction": "我高血压，能吃柚子吗",
        "output": "可以适量食用",
    })
    lines = block.splitlines()
    assert lines[0] == "【科室】内科"
    assert lines[1] == "【主题】高血压能吃柚子吗"
    assert "【问题】我高血压，能吃柚子吗" in block
    assert "【解答】可以适量食用" in block


def test_block_without_department_title_unchanged():
    block = block_from_record({"instruction": "问", "output": "答"})
    assert block == "【问题】问\n【解答】答"


def test_chunk_records_with_ids_keeps_first():
    records = [
        {"id": "r1", "instruction": "问题A", "output": "答案A"},
        {"id": "r2", "instruction": "问题A", "output": "答案A"},
        {"id": "r3", "instruction": "问题B", "output": "答案B"},
        {"instruction": "问题C", "output": "答案C"},
    ]
    out = chunk_records_with_ids(records)
    assert out == [
        ("r1", "【问题】问题A 【解答】答案A"),
        ("r3", "【问题】问题B 【解答】答案B"),
        (None, "【问题】问题C 【解答】答案C"),
    ]
    # 兼容接口：chunk_records 仍只返回文本
    assert chunk_records(records) == [t for _, t in out]


def test_chunk_records_aggregates_and_dedupes():
    records = [
        {"instruction": "问题A", "output": "答案A"},
        {"instruction": "问题A", "output": "答案A"},
        {"instruction": "问题B", "output": "答案B"},
    ]
    chunks = chunk_records(records)
    assert chunks == ["【问题】问题A 【解答】答案A", "【问题】问题B 【解答】答案B"]


def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())