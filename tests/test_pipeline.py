import threading
from dataclasses import replace

import pytest

from ragqa.chunking import chunk_documents
from ragqa.config import PipelineConfig
from ragqa.generation import ExtractiveGenerator
from ragqa.pipeline import RAGPipeline, select_evidence
from ragqa.prompts import clean_output
from ragqa.types import BackendError, Hit, InputError


class StaticRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, question, k):
        return self.hits[:k]


class SpyGenerator:
    def __init__(self, text="内容[E1]", error=False):
        self.calls = 0
        self.text = text
        self.error = error

    def generate(self, question, evidence):
        self.calls += 1
        if self.error:
            raise BackendError("private backend detail")
        return self.text


@pytest.fixture
def hits(documents):
    return [Hit(chunk, score) for chunk, score in zip(chunk_documents(documents), [1.0, 0.1, 0.8])]


def test_zero_distance_equivalent_is_best_hit_and_each_hit_is_filtered(hits):
    evidence = select_evidence(hits, PipelineConfig(min_score=0.5))
    assert [e.chunk_id for e in evidence] == [hits[0].chunk.id, hits[2].chunk.id]
    assert evidence[0].score == 1.0  # cosine distance 0 -> similarity 1, never defaulted away
    assert [e.citation for e in evidence] == ["E1", "E2"]


def test_budget_drops_whole_chunk_not_critical_warning(hits):
    assert not select_evidence(hits, PipelineConfig(max_context_chars=1))
    text = "建议观察。注意事项：不要删除。最后提醒：应保留。"
    assert clean_output(text) == text


def test_duplicate_and_nonfinite_hits_filtered(hits):
    evidence = select_evidence([hits[0], hits[0], replace(hits[2], score=float("nan"))], PipelineConfig())
    assert len(evidence) == 1


def test_no_evidence_never_calls_generator(hits):
    generator = SpyGenerator()
    pipeline = RAGPipeline(StaticRetriever([hits[1]]), generator)
    result = pipeline.run("问题")
    assert result.status == "insufficient_evidence" and generator.calls == 0


def test_extractive_answer_preserves_warning_and_source(hits):
    pipeline = RAGPipeline(StaticRetriever(hits), ExtractiveGenerator())
    result = pipeline.run("问题")
    assert result.status == "answered"
    assert hits[0].chunk.text in result.text
    assert result.evidence[0].source == "synthetic:a"
    assert set(result.citations) <= {e.citation for e in result.evidence}
    assert result.timings_ms["total"] >= 0


@pytest.mark.parametrize("text", ["没有引用", "", "内容[E99]", "内容[E0]", "内容[E01]"])
def test_invalid_citations_not_published(hits, text):
    result = RAGPipeline(StaticRetriever(hits), SpyGenerator(text)).run("问题")
    assert result.status == "insufficient_evidence"
    assert result.reason == "invalid_citations"


def test_backend_error_does_not_become_unretrieved_answer(hits):
    result = RAGPipeline(StaticRetriever(hits), SpyGenerator(error=True)).run("问题")
    assert result.status == "error"
    assert "private" not in result.text


@pytest.mark.parametrize("question", ["", "  ", "x" * 2001, None])
def test_invalid_input(hits, question):
    with pytest.raises(InputError):
        RAGPipeline(StaticRetriever(hits), SpyGenerator()).run(question)


def test_busy_request_fails_fast_and_lock_recovers(hits):
    entered, release = threading.Event(), threading.Event()

    class BlockingGenerator:
        def generate(self, question, evidence):
            entered.set()
            assert release.wait(5)
            return "内容[E1]"

    pipeline = RAGPipeline(StaticRetriever(hits), BlockingGenerator())
    result = []
    thread = threading.Thread(target=lambda: result.append(pipeline.run("first")))
    thread.start()
    try:
        assert entered.wait(5)
        assert pipeline.run("second").reason == "busy"
    finally:
        release.set()
        thread.join(5)
    assert result[0].status == "answered"
    assert pipeline.run("third").status == "answered"


def test_rerank_threshold_requires_reranker(hits):
    with pytest.raises(ValueError, match="requires"):
        RAGPipeline(StaticRetriever(hits), SpyGenerator(), PipelineConfig(min_rerank_score=0.5))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"candidates": 1},
        {"min_score": float("nan")},
        {"min_score": 2},
        {"max_context_chars": 0},
    ],
)
def test_configuration_errors(kwargs):
    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)
