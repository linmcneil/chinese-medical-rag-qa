"""Composition root; all user-facing entry points use this factory."""

from __future__ import annotations

from ragqa.config import PipelineConfig
from ragqa.generation import ExtractiveGenerator, TransformersGenerator
from ragqa.pipeline import RAGPipeline
from ragqa.retrieval import load_retriever


def create_pipeline(
    index: str,
    *,
    generator: str = "extractive",
    device: str = "cpu",
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    revision: str | None = None,
    quantize: str = "none",
    adapter: str | None = None,
    rerank: bool = False,
    rerank_model: str = "BAAI/bge-reranker-base",
    rerank_revision: str | None = None,
    max_new_tokens: int = 256,
    config: PipelineConfig | None = None,
) -> RAGPipeline:
    if generator not in ("extractive", "transformers"):
        raise ValueError("Unknown generator")
    backend = (
        ExtractiveGenerator()
        if generator == "extractive"
        else TransformersGenerator(
            model,
            revision=revision,
            device=device,
            max_new_tokens=max_new_tokens,
            quantize=quantize,
            adapter=adapter,
        )
    )
    ranker = None
    if rerank:
        from ragqa.rerank import CrossEncoderReranker

        ranker = CrossEncoderReranker(rerank_model, device, rerank_revision)
    return RAGPipeline(load_retriever(index, device), backend, config, ranker)
