import copy

import pytest

from ragqa.evaluation import consistency_dataset, evaluate, retrieval_metrics, rouge, validate_dataset
from ragqa.factory import create_pipeline
from ragqa.retrieval import build_index


def test_chinese_rouge_identity_and_mismatch():
    assert rouge("这是中文答案", "这是中文答案") == {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}
    assert rouge("甲乙丙", "丁戊己") == {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    assert rouge("", "") == {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    assert rouge("甲乙丙", "甲丁丙")["rouge1"] == pytest.approx(2 / 3)


def test_document_level_metrics_deduplicate_chunks():
    result = retrieval_metrics(["a", "a", "b"], {"b", "c"})
    assert result["mrr"] == 0.5
    assert result["recall@3"] == 0.5
    assert result["hit@1"] == 0


def test_missing_gold_is_error_not_silent_failure(documents):
    dataset = consistency_dataset(documents)
    with pytest.raises(ValueError, match="absent"):
        validate_dataset(dataset, set())


@pytest.mark.parametrize("change", ["empty", "duplicate", "blank", "missing_answerable"])
def test_bad_eval_rows_not_counted_in_denominator(documents, change):
    dataset = consistency_dataset(documents)
    if change == "empty":
        dataset["items"] = []
    elif change == "duplicate":
        dataset["items"].append(copy.deepcopy(dataset["items"][0]))
    elif change == "blank":
        dataset["items"][0]["question"] = " "
    else:
        del dataset["items"][0]["answerable"]
    with pytest.raises(ValueError):
        validate_dataset(dataset, {d.id for d in documents})


def test_end_to_end_eval_distinguishes_unanswerable_and_self_retrieval(documents, tmp_path):
    manifest = build_index(documents, tmp_path / "index")
    pipeline = create_pipeline(str(tmp_path / "index"))
    dataset = consistency_dataset(documents)
    dataset["items"].append(
        {"id": "unanswerable", "question": "xyznonmatching", "answerable": False, "relevant_document_ids": []}
    )
    report = evaluate(pipeline, dataset, manifest, {d.id for d in documents})
    assert report["n"] == 4 and report["answerable_n"] == 3
    assert report["dataset_kind"] == "self_retrieval"
    assert report["end_to_end"]["errors"] == 0
    assert report["end_to_end"]["refusal_rate_on_unanswerable"] == 1.0
    assert report["end_to_end"]["latency_p95_ms"] >= report["end_to_end"]["latency_p50_ms"]
