"""data 模块单元测试（纯标准库，无第三方依赖）。"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragqa.data import (add_ids, dedupe_records, iter_csv_records, load_json,
                        normalize_records, reservoir_sample, sample_subset, split_eval,
                        to_rag_record, to_sft_record, write_json)


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(r) + "\n")


def test_iter_and_normalize_with_header():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.csv")
        _write_csv(p, ["department", "title", "ask", "answer"], [
            ["内科", "高血压", "我血压高怎么办", "按时服药，低盐饮食"],
            ["内科", "", "", ""],                      # 空回答 -> 丢弃
            ["外科", "骨折", "摔倒骨折", "需要拍片确诊"],
        ])
        rows = list(iter_csv_records(p))
        assert len(rows) == 2  # 空 ask/answer 行在迭代层已跳过
        recs = normalize_records(rows)
        assert len(recs) == 2
        assert recs[0] == {"department": "内科", "title": "高血压",
                           "ask": "我血压高怎么办", "answer": "按时服药，低盐饮食"}
        assert recs[1]["department"] == "外科"


def test_iter_without_header_positional():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.csv")
        _write_csv(p, [], [["儿科", "发烧", "孩子发烧", "多喝水"]])
        rows = list(iter_csv_records(p))
        assert rows[0]["department"] == "儿科"
        assert rows[0]["title"] == "发烧"


def test_dedupe_and_ids():
    rows = [{"ask": "q1", "answer": "a1"}, {"ask": "q1", "answer": "a1"},
            {"ask": "q2", "answer": "a2"}]
    dd = dedupe_records(rows)
    assert len(dd) == 2
    ids = add_ids(dd, prefix="med")
    assert ids[0]["id"] == "med_0000000"
    assert ids[1]["id"] == "med_0000001"


def test_split_eval_disjoint():
    recs = [{"id": f"r{i}"} for i in range(50)]
    rest, ev = split_eval(recs, 10, seed=7)
    assert len(ev) == 10 and len(rest) == 40
    ids_ev = {r["id"] for r in ev}
    assert not (ids_ev & {r["id"] for r in rest})


def test_reservoir_sample_fixed_size():
    stream = ({"id": i} for i in range(1000))
    out = reservoir_sample(stream, 50, seed=1)
    assert len(out) == 50
    assert len({r["id"] for r in out}) == 50


def test_to_rag_and_sft():
    rec = {"id": "med_1", "department": "内科", "title": "高血压",
           "ask": "血压高怎么办", "answer": "低盐饮食"}
    rag = to_rag_record(rec)
    assert rag["id"] == "med_1"
    assert rag["instruction"] == "血压高怎么办"
    assert rag["output"] == "低盐饮食"
    assert rag["title"] == "高血压" and rag["history"] == []
    sft = to_sft_record(rec)
    assert sft["input"] == "血压高怎么办"
    assert sft["output"] == "低盐饮食"
    assert isinstance(sft["instruction"], str) and sft["instruction"]


def test_write_load_json():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sub", "out.json")
        write_json(p, [{"a": 1, "中文": "值"}])
        assert load_json(p) == [{"a": 1, "中文": "值"}]




def test_sample_subset_deterministic_subset():
    recs = [{"id": f"r{i}"} for i in range(30)]
    a = sample_subset(recs, 8, seed=3)
    b = sample_subset(recs, 8, seed=3)
    assert a == b and len(a) == 8
    assert {r["id"] for r in a} <= {r["id"] for r in recs}
    assert len({r["id"] for r in a}) == 8




def test_iter_with_chinese_header():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "zh.csv")
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write("科室,标题,问题,回答\n")
            fh.write("儿科,发烧,孩子反复发烧怎么办,多喝水并观察体温\n")
            fh.write("内科,高血压,血压偏高怎么调理,低盐饮食规律作息\n")
        rows = list(iter_csv_records(p))
        assert len(rows) == 2
        assert rows[0] == {"department": "儿科", "title": "发烧",
                           "ask": "孩子反复发烧怎么办", "answer": "多喝水并观察体温"}
        assert rows[1]["department"] == "内科"


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