from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    candidates: int = 20
    top_k: int = 3
    min_score: float = 0.25
    min_rerank_score: float | None = None
    max_context_chars: int = 1600
    max_question_chars: int = 2000
    max_per_document: int = 2

    def __post_init__(self):
        if not 1 <= self.top_k <= self.candidates <= 100:
            raise ValueError("Require 1 <= top_k <= candidates <= 100")
        if not math.isfinite(self.min_score) or not -1 <= self.min_score <= 1:
            raise ValueError("min_score must be a finite cosine similarity in [-1, 1]")
        if self.min_rerank_score is not None and not math.isfinite(self.min_rerank_score):
            raise ValueError("min_rerank_score must be finite")
        if min(self.max_context_chars, self.max_question_chars, self.max_per_document) < 1:
            raise ValueError("Budgets must be positive")
