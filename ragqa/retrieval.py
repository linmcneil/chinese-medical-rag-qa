"""Offline lexical baseline and optional immutable Chroma/cosine index."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ragqa.chunking import chunk_documents
from ragqa.data import digest, normalize, write_json
from ragqa.types import BackendError, Chunk, Document, Hit

SCHEMA_VERSION = 2


def lexical_tokens(text: str) -> list[str]:
    parts = re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", normalize(text).lower())
    tokens = []
    for part in parts:
        if "\u3400" <= part[0] <= "\u9fff" and len(part) > 1:
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


class LexicalRetriever:
    """Character-bigram/Latin-token cosine. A smoke-test baseline, not dense retrieval."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectors = [Counter(lexical_tokens(c.search_text)) for c in chunks]
        self.norms = [math.sqrt(sum(v * v for v in x.values())) for x in self.vectors]

    def search(self, question: str, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        q = Counter(lexical_tokens(question))
        qnorm = math.sqrt(sum(v * v for v in q.values()))
        if not qnorm:
            return []
        hits = []
        for chunk, vector, norm in zip(self.chunks, self.vectors, self.norms):
            score = sum(n * vector.get(t, 0) for t, n in q.items()) / (qnorm * norm) if norm else 0
            hits.append(Hit(chunk, score))
        return sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:k]


def _embedding_model(model: str, revision: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model, revision=revision, device=device, trust_remote_code=False)


def build_index(
    documents: list[Document],
    output: str | Path,
    *,
    backend: str = "lexical",
    max_chars: int = 400,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    embedding_revision: str | None = None,
    device: str = "cpu",
) -> dict:
    if backend not in ("lexical", "chroma"):
        raise ValueError("backend must be lexical or chroma")
    if backend == "chroma" and not re.fullmatch(r"[a-fA-F0-9]{40}", embedding_revision or ""):
        raise ValueError("Dense indexes require --embedding-revision with a full HF commit SHA")
    chunks = chunk_documents(documents, max_chars)
    if not chunks or len({c.id for c in chunks}) != len(chunks):
        raise ValueError("Index needs nonempty, uniquely identified chunks")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new index directory; indexes are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".index-", dir=output.parent))
    rows = [asdict(c) for c in chunks]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "metric": "cosine",
        "chunk_max_chars": max_chars,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "corpus_sha256": digest([asdict(d) for d in documents]),
        "chunks_sha256": digest(rows),
        "embedding_model": embedding_model if backend == "chroma" else None,
        "embedding_revision": embedding_revision if backend == "chroma" else None,
        "normalized_embeddings": backend == "chroma",
    }
    try:
        write_json(stage / "chunks.json", rows)
        if backend == "chroma":
            # Build Chroma in its final path: SQLite connections may remain open (Windows).
            # The manifest is the commit marker. Incomplete builds cannot be opened.
            os.rename(stage, output)
            stage = output
            import chromadb

            model = _embedding_model(embedding_model, embedding_revision, device)
            client = chromadb.PersistentClient(
                path=str(stage / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False)
            )
            collection = client.create_collection(
                "evidence", metadata={"hnsw:space": "cosine"}, embedding_function=None
            )
            dimension = None
            for start in range(0, len(chunks), 128):
                batch = chunks[start : start + 128]
                vectors = model.encode([c.search_text for c in batch], normalize_embeddings=True)
                dimension = int(vectors.shape[1])
                collection.add(ids=[c.id for c in batch], embeddings=vectors.tolist())
            manifest["embedding_dimension"] = dimension
        write_json(stage / "manifest.json", manifest)
        if stage != output:
            os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def read_index(path: str | Path) -> tuple[dict, list[Chunk]]:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("metric") != "cosine":
        raise ValueError("Incompatible index. Build a v2 index in a new directory")
    rows = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
    if digest(rows) != manifest.get("chunks_sha256") or len(rows) != manifest.get("chunk_count"):
        raise ValueError("Index chunk fingerprint/count mismatch")
    chunks = [Chunk(**row) for row in rows]
    if len({c.id for c in chunks}) != len(chunks) or not chunks:
        raise ValueError("Corrupt or empty index")
    return manifest, chunks


class ChromaRetriever:
    def __init__(self, path: str | Path, manifest: dict, chunks: list[Chunk], device: str):
        import chromadb

        self.model = _embedding_model(manifest["embedding_model"], manifest["embedding_revision"], device)
        self.dimension = manifest["embedding_dimension"]
        self.chunks = {c.id: c for c in chunks}
        self.client = chromadb.PersistentClient(
            path=str(Path(path) / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection("evidence", embedding_function=None)
        if self.collection.count() != len(chunks) or self.collection.metadata.get("hnsw:space") != "cosine":
            raise ValueError("Chroma collection does not match the manifest")

    def search(self, question: str, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        try:
            vector = self.model.encode([question], normalize_embeddings=True)
            if vector.shape[1] != self.dimension:
                raise BackendError("Embedding dimension mismatch")
            result = self.collection.query(
                query_embeddings=vector.tolist(), n_results=min(k, len(self.chunks)), include=["distances"]
            )
            return [
                Hit(self.chunks[cid], 1.0 - float(distance))
                for cid, distance in zip(result["ids"][0], result["distances"][0])
            ]
        except Exception as exc:
            raise BackendError("Dense retrieval failed") from exc


def load_retriever(path: str | Path, device: str = "cpu"):
    manifest, chunks = read_index(path)
    if manifest["backend"] == "lexical":
        return LexicalRetriever(chunks)
    if manifest["backend"] == "chroma":
        return ChromaRetriever(path, manifest, chunks, device)
    raise ValueError("Unknown index backend")
