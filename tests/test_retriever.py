"""retriever 封装单元测试（用假 collection，不需要 chromadb/嵌入模型）。

运行方式：python tests/test_retriever.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragqa.retriever import ChromaRetriever


class FakeCollection:
    """最小假实现：只提供 query/count，结果形状对齐 chromadb。"""

    def __init__(self, payload=None, count_value=0):
        self.payload = payload if payload is not None else {}
        self.count_value = count_value
        self.last_kwargs = None

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        return self.payload

    def count(self):
        return self.count_value


def _retriever_with(payload, count_value=0):
    r = ChromaRetriever(chroma_dir="/tmp/none", model_dir="/tmp/models", top_k=3)
    r._collection = FakeCollection(payload, count_value)
    return r


def _payload(ids, docs, dists=None, metas=None):
    return {
        "ids": [ids],
        "documents": [docs],
        "distances": [dists] if dists is not None else [[]],
        "metadatas": [metas] if metas is not None else [[]],
    }


def test_query_maps_hit_fields():
    r = _retriever_with(_payload(["id-1", "id-2"], ["甲文", "乙文"], [0.1, 0.2],
                                 [{"rid": "r-1"}, {"rid": "r-2"}]))
    hits = r.query("高血压")
    assert len(hits) == 2
    assert hits[0] == {"id": "id-1", "document": "甲文",
                       "distance": 0.1, "rid": "r-1"}
    assert hits[1]["rid"] == "r-2"


def test_query_rid_fallback_when_metadata_missing():
    r = _retriever_with(_payload(["id-1"], ["文"], [0.5], [{}]))
    hits = r.query("问题")
    assert hits[0]["rid"] == ""


def test_query_empty_result():
    r = _retriever_with(_payload([], [], [], []))
    assert r.query("没有命中") == []


def test_query_uses_default_top_k_when_not_given():
    r = _retriever_with(_payload(["a"], ["文"]))
    r.query("问题")
    assert r._collection.last_kwargs.get("n_results") == 3


def test_query_overrides_top_k():
    r = _retriever_with(_payload(["a"], ["文"]))
    r.query("问题", top_k=5)
    assert r._collection.last_kwargs.get("n_results") == 5


def test_count_delegates_to_collection():
    r = _retriever_with({}, count_value=42)
    assert r.count() == 42


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