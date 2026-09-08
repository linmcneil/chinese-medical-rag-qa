"""prompts 模块单元测试（纯函数，无第三方依赖）。

运行方式：python tests/test_prompts.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragqa.prompts import build_rag_prompt, extract_answer, format_context


def test_build_rag_prompt_contains_context_and_question():
    prompt = build_rag_prompt("低盐饮食", "高血压怎么调理？")
    assert "低盐饮食" in prompt
    assert "高血压怎么调理" in prompt
    assert "避免猜测" in prompt


def test_format_context_with_rids():
    docs = ["第一条解答", "第二条解答"]
    rids = ["r-1", "r-2"]
    text = format_context(docs, rids)
    assert text.startswith("【来源 1】r-1")
    assert "【来源 2】r-2" in text
    assert "第一条解答" in text and "第二条解答" in text


def test_format_context_without_rids():
    text = format_context(["只给解答"], rids=None)
    assert text == "只给解答"
    assert "【来源" not in text


def test_extract_answer_strips_exact_prefix():
    prompt = "问题：你好\n请回答：\n建议："
    assert extract_answer(prompt + "多喝水。", prompt) == "多喝水。"


def test_extract_answer_fallback_on_last_line():
    prompt = "说明：\n最后一行：建议"
    generated = "前缀噪音\n最后一行：建议\n保留正文"
    assert extract_answer(generated, prompt) == "保留正文"


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