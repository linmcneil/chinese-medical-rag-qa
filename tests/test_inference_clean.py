"""inference 纯逻辑单元测试（不依赖 torch：纯函数路径）。

运行方式：python tests/test_inference_clean.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragqa.inference import (build_medical_prefix, clean_context_docs,
                             clean_generated, is_context_relevant,
                             trim_repetition)


def test_clean_generated_empty_input():
    assert clean_generated("") == ""
    assert clean_generated(None) == ""


def test_clean_generated_strips_header_and_polite_tail():
    out = clean_generated("答：多喝水。祝您早日康复。")
    assert out == "多喝水。"


def test_clean_generated_keeps_after_prompt_replay():
    raw = "患者描述：失眠怎么办？\n建议：先固定作息，白天适量运动。"
    assert clean_generated(raw) == "先固定作息，白天适量运动。"


def test_clean_generated_truncates_at_transition():
    assert clean_generated("先休息。答案是：这个要分情况讨论。") == "先休息。"


def test_trim_repetition_stops_duplicated_sentence():
    assert trim_repetition("先休息。先休息。多喝水。") == "先休息。"


def test_build_medical_prefix_layout():
    prefix = build_medical_prefix("胃痛怎么办")
    assert "患者描述：胃痛怎么办" in prefix
    assert prefix.endswith("建议：")


def test_clean_context_docs_keeps_answer_body():
    doc = "【主题】感冒\n【解答】多喝热水，观察体温。祝您早日康复。"
    assert clean_context_docs([doc]) == ["多喝热水，观察体温。"]


def test_clean_context_docs_stops_at_second_record():
    doc = "【解答】第一条结论。\n【问题】第二条问题\n【解答】第二条结论。"
    cleaned = clean_context_docs([doc])
    assert cleaned == ["第一条结论。"]


def test_is_context_relevant_shared_and_unrelated():
    assert is_context_relevant("胃溃疡吃奥美拉唑", "奥美拉唑治疗胃溃疡效果好")
    assert not is_context_relevant("婴儿湿疹怎么护理", "股票基金行情分析")
    assert not is_context_relevant("高血压怎么办", "")


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