from __future__ import annotations

import math
from dataclasses import replace

from ragqa.types import BackendError, Hit


class CrossEncoderReranker:
    def __init__(
        self, model: str = "BAAI/bge-reranker-base", device: str = "cpu", revision: str | None = None
    ):
        self.model_id, self.device, self.revision = model, device, revision
        self.model = None

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        try:
            if self.model is None:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(
                    self.model_id, device=self.device, revision=self.revision, trust_remote_code=False
                )
            scores = self.model.predict([(question, h.chunk.search_text) for h in hits])
            if len(scores) != len(hits) or any(not math.isfinite(float(s)) for s in scores):
                raise ValueError("Invalid reranker scores")
            ranked = [replace(h, rerank_score=float(s)) for h, s in zip(hits, scores)]
            return sorted(ranked, key=lambda h: -h.rerank_score)
        except Exception as exc:
            raise BackendError("Reranking failed") from exc
