"""One request lifecycle for CLI, API, UI and end-to-end evaluation."""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from collections import Counter

from ragqa.config import PipelineConfig
from ragqa.prompts import clean_output
from ragqa.types import Answer, BackendError, Evidence, Generator, Hit, InputError, Reranker, Retriever

log = logging.getLogger(__name__)


def select_evidence(hits: list[Hit], config: PipelineConfig) -> tuple[Evidence, ...]:
    selected, seen, counts = [], set(), Counter()
    used = 0
    for hit in hits:
        chunk = hit.chunk
        if not math.isfinite(hit.score) or hit.score < config.min_score:
            continue
        if config.min_rerank_score is not None and (
            hit.rerank_score is None
            or not math.isfinite(hit.rerank_score)
            or hit.rerank_score < config.min_rerank_score
        ):
            continue
        if not chunk.text.strip() or chunk.id in seen:
            continue
        if counts[chunk.document_id] >= config.max_per_document:
            continue
        # Never truncate a warning or negation just to fit the character budget.
        if used + len(chunk.text) + len(chunk.question) > config.max_context_chars:
            continue
        selected.append(
            Evidence(
                f"E{len(selected) + 1}",
                chunk.id,
                chunk.document_id,
                chunk.question,
                chunk.text,
                chunk.source,
                hit.score,
                hit.rerank_score,
            )
        )
        used += len(chunk.text) + len(chunk.question)
        seen.add(chunk.id)
        counts[chunk.document_id] += 1
        if len(selected) == config.top_k:
            break
    return tuple(selected)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        config: PipelineConfig | None = None,
        reranker: Reranker | None = None,
    ):
        self.retriever, self.generator, self.reranker = retriever, generator, reranker
        self.config = config or PipelineConfig()
        if self.config.min_rerank_score is not None and reranker is None:
            raise ValueError("min_rerank_score requires a reranker")
        # Single in-flight inference per process: protects model initialization and memory.
        self._lock = threading.Lock()

    def run(self, question: str) -> Answer:
        if not isinstance(question, str) or not question.strip():
            raise InputError("问题不能为空")
        question = question.strip()
        if len(question) > self.config.max_question_chars:
            raise InputError(f"问题不能超过 {self.config.max_question_chars} 字符")
        trace = uuid.uuid4().hex
        if not self._lock.acquire(blocking=False):
            return Answer("error", "服务忙，请稍后重试。", trace_id=trace, reason="busy")
        start = time.perf_counter()
        timings = {}
        evidence = ()

        def finish(status, text, reason="", citations=()):
            timings["total"] = round((time.perf_counter() - start) * 1000, 3)
            return Answer(status, text, evidence, citations, trace, timings, reason)

        try:
            t = time.perf_counter()
            hits = self.retriever.search(question, self.config.candidates)
            timings["retrieval"] = round((time.perf_counter() - t) * 1000, 3)
            if self.reranker and hits:
                t = time.perf_counter()
                hits = self.reranker.rerank(question, hits)
                timings["rerank"] = round((time.perf_counter() - t) * 1000, 3)
            evidence = select_evidence(hits, self.config)
            if not evidence:
                return finish(
                    "insufficient_evidence",
                    "未检索到足够的支持资料，请补充问题或查阅可靠来源。",
                    "no_supported_context",
                )
            t = time.perf_counter()
            text = clean_output(self.generator.generate(question, evidence))
            timings["generation"] = round((time.perf_counter() - t) * 1000, 3)
            citations = tuple(dict.fromkeys(re.findall(r"\[(E[0-9]+)\]", text)))
            allowed = {e.citation for e in evidence}
            if not text or not citations or not set(citations).issubset(allowed):
                return finish(
                    "insufficient_evidence",
                    "生成结果未通过来源编号检查，请查看检索片段。",
                    "invalid_citations",
                )
            return finish("answered", text, citations=citations)
        except BackendError:
            log.exception("RAG backend failed trace_id=%s", trace)
            return finish("error", "检索或生成服务暂不可用。", "backend_failure")
        finally:
            self._lock.release()
