"""Explicit retrieval protocol and dependency-free Chinese character ROUGE."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict

from ragqa.data import digest
from ragqa.types import Document


def rouge_tokens(text: str) -> list[str]:
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text.lower())


def _f1(overlap: int, predicted: int, reference: int) -> float:
    return 2 * overlap / (predicted + reference) if predicted and reference else 0.0


def rouge(reference: str, prediction: str) -> dict[str, float]:
    gold, pred = rouge_tokens(reference), rouge_tokens(prediction)
    scores = {}
    for n in (1, 2):
        a = Counter(tuple(gold[i : i + n]) for i in range(len(gold) - n + 1))
        b = Counter(tuple(pred[i : i + n]) for i in range(len(pred) - n + 1))
        scores[f"rouge{n}"] = _f1(sum((a & b).values()), sum(b.values()), sum(a.values()))
    # Linear memory LCS; outputs are short, so a quadratic time implementation suffices.
    previous = [0] * (len(pred) + 1)
    for token in gold:
        current = [0]
        for j, other in enumerate(pred, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    scores["rougeL"] = _f1(previous[-1], len(pred), len(gold))
    return scores


def consistency_dataset(documents: list[Document]) -> dict:
    return {
        "schema_version": 2,
        "kind": "self_retrieval",
        "description": "Original questions already present in the KB; NOT a held-out quality benchmark.",
        "items": [
            {
                "id": d.id,
                "question": d.question,
                "answerable": True,
                "relevant_document_ids": [d.id],
                "reference_answer": d.answer,
            }
            for d in documents
        ],
    }


def validate_dataset(dataset: dict, valid_document_ids: set[str]) -> list[dict]:
    if dataset.get("schema_version") != 2 or dataset.get("kind") not in ("self_retrieval", "curated"):
        raise ValueError("Evaluation requires a v2 self_retrieval or curated dataset")
    rows = dataset.get("items")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation dataset must contain items")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each evaluation row needs an id")
        if row["id"] in seen:
            raise ValueError("Duplicate evaluation id")
        seen.add(row["id"])
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{row['id']}: empty question")
        if type(row.get("answerable")) is not bool:
            raise ValueError("answerable must be an explicit boolean")
        gold = row.get("relevant_document_ids")
        if not isinstance(gold, list) or any(not isinstance(g, str) for g in gold):
            raise ValueError("relevant_document_ids must be a string list")
        if bool(gold) != row["answerable"]:
            raise ValueError("Answerable rows need gold IDs; unanswerable rows must not have them")
        if not set(gold).issubset(valid_document_ids):
            raise ValueError(f"{row['id']}: gold document absent from the index")
        if "reference_answer" in row and not isinstance(row["reference_answer"], str):
            raise ValueError("reference_answer must be a string")
    return rows


def retrieval_metrics(ranked_ids: list[str], gold: set[str], ks=(1, 3, 5)) -> dict:
    # Report document-level ranks: multiple chunks from one document consume one rank.
    ids = list(dict.fromkeys(ranked_ids))
    result = {}
    for k in ks:
        found = len(set(ids[:k]) & gold)
        result[f"hit@{k}"] = float(found > 0)
        result[f"recall@{k}"] = found / len(gold) if gold else 0.0
    result["mrr"] = next((1 / i for i, rid in enumerate(ids, 1) if rid in gold), 0.0)
    return result


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    position = (len(vals) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return vals[low] + (vals[high] - vals[low]) * (position - low)


def evaluate(
    pipeline, dataset: dict, manifest: dict, valid_document_ids: set[str], *, end_to_end: bool = True
) -> dict:
    rows = validate_dataset(dataset, valid_document_ids)
    if pipeline.config.candidates < 5:
        raise ValueError("Evaluation of Hit@5 requires at least 5 retrieval candidates")
    records, aggregate, latencies = [], Counter(), []
    answerable_count = unanswerable_count = errors = correct_refusals = answered_supported = 0
    for row in rows:
        hits = pipeline.retriever.search(row["question"], pipeline.config.candidates)
        if pipeline.reranker:
            hits = pipeline.reranker.rerank(row["question"], hits)
        record = {
            "id": row["id"],
            "retrieved_chunk_ids": [h.chunk.id for h in hits],
            "retrieved_document_ids": [h.chunk.document_id for h in hits],
        }
        if row["answerable"]:
            answerable_count += 1
            metric = retrieval_metrics(record["retrieved_document_ids"], set(row["relevant_document_ids"]))
            aggregate.update(metric)
            record["retrieval"] = metric
        else:
            unanswerable_count += 1
        if end_to_end:
            result = pipeline.run(row["question"])
            record["result"] = result.to_dict()
            errors += result.status == "error"
            correct_refusals += not row["answerable"] and result.status == "insufficient_evidence"
            answered_supported += row["answerable"] and result.status == "answered"
            latencies.append(result.timings_ms.get("total", 0))
            if row.get("reference_answer") and result.status == "answered":
                record["rouge"] = rouge(row["reference_answer"], result.text)
        records.append(record)
    return {
        "schema_version": 2,
        "dataset_kind": dataset["kind"],
        "dataset_sha256": digest(dataset),
        "index": manifest,
        "config": asdict(pipeline.config),
        "n": len(rows),
        "answerable_n": answerable_count,
        "unanswerable_n": unanswerable_count,
        "retrieval": {k: v / answerable_count for k, v in aggregate.items()} if answerable_count else {},
        "end_to_end": {
            "errors": errors,
            "answer_rate_on_answerable": answered_supported / answerable_count if answerable_count else None,
            "refusal_rate_on_unanswerable": correct_refusals / unanswerable_count
            if unanswerable_count
            else None,
            "latency_p50_ms": percentile(latencies, 0.5),
            "latency_p95_ms": percentile(latencies, 0.95),
            "timing_note": "Includes cold first inference; evaluate performs a separate retrieval pass.",
        }
        if end_to_end
        else None,
        "limitations": "Lexical/semantic overlap and valid citation IDs do not establish medical correctness.",
        "records": records,
    }
