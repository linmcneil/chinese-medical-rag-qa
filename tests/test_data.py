import json
from dataclasses import asdict

import pytest

from ragqa.data import make_document, question_key, read_documents, split_documents, write_split
from ragqa.training import load_split


def test_ids_are_content_based_not_order_or_source():
    a = make_document("问题", "答案", source="old")
    b = make_document("问题", "答案", source="new")
    assert a.id == b.id
    assert a.id != make_document("问题", "新答案").id


def test_json_migration_and_deduplication(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            [
                {"instruction": "问题", "output": "答案"},
                {"instruction": "问题", "output": "答案"},
                {"instruction": "角色", "input": "另一问题", "output": "另一答案"},
            ]
        ),
        encoding="utf-8",
    )
    docs = read_documents(path)
    assert {d.question for d in docs} == {"问题", "另一问题"}
    assert len(docs) == 2


@pytest.mark.parametrize("header", ["department,title,ask,answer\n", "科室,标题,问题,回答\n", ""])
def test_csv_headers_and_explicit_gbk(tmp_path, header):
    path = tmp_path / "input.csv"
    path.write_bytes((header + "内科,主题,问题,答案\n").encode("gb18030"))
    assert read_documents(path, "gb18030")[0].question == "问题"


def test_malformed_csv_is_not_silently_skipped(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        read_documents(path)


@pytest.mark.parametrize("value", [{}, [], [{"question": "", "answer": "a"}], [42]])
def test_bad_data_rejected(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        read_documents(path)


def test_grouped_split_stable_and_disjoint():
    docs = [make_document(f"问题{i}", "答案") for i in range(20)]
    docs += [make_document("问题 1？", "另一个答案")]
    a, b = split_documents(docs, 0.2, 7)
    c, d = split_documents(list(reversed(docs)), 0.2, 7)
    assert {x.id for x in a} == {x.id for x in c}
    assert {x.id for x in b} == {x.id for x in d}
    assert not {question_key(x.question) for x in a} & {question_key(x.question) for x in b}
    assert len(a) + len(b) == len(docs)


def test_split_manifest_is_exact_and_immutable(documents, tmp_path):
    out = tmp_path / "split"
    write_split(documents, out, 0.3)
    train, val, manifest = load_split(out)
    assert [x.id for x in train] == manifest["train_ids"]
    assert [x.id for x in val] == manifest["validation_ids"]
    assert not {x.id for x in train} & {x.id for x in val}
    with pytest.raises(FileExistsError):
        write_split(documents, out)
    path = out / "validation.json"
    path.write_text(json.dumps([asdict(documents[0])]), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_split(out)
