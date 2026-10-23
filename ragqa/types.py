"""Stable contracts shared by ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, Sequence


@dataclass(frozen=True)
class Document:
    id: str
    question: str
    answer: str
    title: str = ""
    department: str = ""
    source: str = ""


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    question: str
    text: str
    title: str
    department: str
    source: str
    start: int
    end: int

    @property
    def search_text(self) -> str:
        return "\n".join(p for p in (self.title, self.question, self.text) if p)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # Cosine similarity, NOT a probability of correctness.
    rerank_score: float | None = None


@dataclass(frozen=True)
class Evidence:
    citation: str
    chunk_id: str
    document_id: str
    question: str
    text: str
    source: str
    score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class Answer:
    status: Literal["answered", "insufficient_evidence", "error"]
    text: str
    evidence: tuple[Evidence, ...] = ()
    citations: tuple[str, ...] = ()
    trace_id: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever(Protocol):
    def search(self, question: str, k: int) -> list[Hit]: ...


class Reranker(Protocol):
    def rerank(self, question: str, hits: Sequence[Hit]) -> list[Hit]: ...


class Generator(Protocol):
    def generate(self, question: str, evidence: Sequence[Evidence]) -> str: ...


class BackendError(RuntimeError):
    """Backend failure; the pipeline never silently answers without retrieval."""


class InputError(ValueError):
    """Invalid user input; safe for clients to display."""
