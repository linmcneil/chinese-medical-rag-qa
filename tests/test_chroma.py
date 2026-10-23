"""Real Chroma storage/query, deterministic synthetic embeddings; no model downloads."""

import pytest

pytest.importorskip("chromadb")
import numpy as np

from ragqa import retrieval


@pytest.mark.integration
def test_real_chroma_zero_distance_and_reload(documents, tmp_path, monkeypatch):
    class TinyEmbedder:
        def encode(self, texts, **kwargs):
            vectors = []
            for text in texts:
                vectors.append(
                    [1.0, 0.0, 0.0]
                    if "ALPHA" in text
                    else ([0.0, 1.0, 0.0] if "BETA" in text else [0.0, 0.0, 1.0])
                )
            return np.array(vectors, dtype=np.float32)

    calls = []

    def model(name, revision, device):
        calls.append((name, revision, device))
        return TinyEmbedder()

    monkeypatch.setattr(retrieval, "_embedding_model", model)
    index = tmp_path / "dense"
    manifest = retrieval.build_index(documents, index, backend="chroma", embedding_revision="a" * 40)
    assert manifest["embedding_dimension"] == 3
    retriever = retrieval.load_retriever(index)
    hits = retriever.search("ALPHA", 20)
    assert hits[0].chunk.document_id == documents[0].id
    assert hits[0].score == pytest.approx(1.0)
    assert len(hits) == 3
    assert calls[0] == calls[1]  # Query uses the exact build model and revision.
