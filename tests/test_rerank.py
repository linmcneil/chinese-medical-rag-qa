"""rerank 纯逻辑单元测试（不加载模型，无需第三方依赖）。

运行方式：python tests/test_rerank.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragqa.rerank import rerank_hits


def _hit(i):
    return {"id": f"id-{i}", "document": f"文档{i}", "distance": 1.0 / (i + 1), "rid": f"r-{i}"}


def test_rerank_hits_empty_input():
    assert rerank_hits([], []) == []


def test_rerank_hits_sorts_descending_by_score():
    hits = [_hit(0), _hit(1), _hit(2)]
    ranked = rerank_hits(hits, [0.3, 0.9, 0.6])
    assert [h["id"] for h in ranked] == ["id-1", "id-2", "id-0"]
    assert ranked[0]["rerank_score"] == 0.9


def test_rerank_hits_stable_on_ties():
    hits = [_hit(0), _hit(1), _hit(2)]
    ranked = rerank_hits(hits, [0.5, 0.5, 0.5])
    assert [h["id"] for h in ranked] == ["id-0", "id-1", "id-2"]


def test_rerank_hits_truncates_to_top_k():
    hits = [_hit(0), _hit(1), _hit(2), _hit(3)]
    ranked = rerank_hits(hits, [0.1, 0.9, 0.8, 0.7], top_k=2)
    assert [h["id"] for h in ranked] == ["id-1", "id-2"]


def test_rerank_hits_keeps_original_fields():
    ranked = rerank_hits([_hit(7)], [0.2])
    assert ranked[0]["rid"] == "r-7"
    assert ranked[0]["distance"] == 1.0 / 8
    assert "rerank_score" in ranked[0]


def test_rerank_hits_rejects_mismatched_scores():
    try:
        rerank_hits([_hit(0), _hit(1)], [0.1])
    except ValueError:
        return
    raise AssertionError("长度不一致时应抛出 ValueError")


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