import json

import pytest

from ragqa.chunking import chunk_documents
from ragqa.data import make_document
from ragqa.retrieval import LexicalRetriever, build_index, load_retriever, read_index


def test_chunks_always_contain_answer_and_reconstruct_it():
    answer = ("资料正文。" * 80) + "注意事项：不要删除这一句。"
    doc = make_document("很长的问题" * 100, answer)
    chunks = chunk_documents([doc], max_chars=64)
    assert all(c.text and len(c.text) <= 64 for c in chunks)
    assert "".join(c.text for c in chunks) == doc.answer
    assert all(c.text == doc.answer[c.start : c.end] for c in chunks)
    assert all(c.document_id == doc.id and c.question == doc.question for c in chunks)


def test_identical_answers_retain_all_sources():
    docs = [make_document("问题甲", "共享答案"), make_document("问题乙", "共享答案")]
    chunks = chunk_documents(docs)
    assert len(chunks) == 2 and chunks[0].id != chunks[1].id


def test_real_lexical_index_roundtrip(documents, tmp_path):
    out = tmp_path / "index"
    manifest = build_index(documents, out)
    loaded, chunks = read_index(out)
    assert loaded == manifest
    result = load_retriever(out).search("ALPHA-01", 100)
    assert result[0].chunk.document_id == documents[0].id
    assert len(result) == len(chunks)
    with pytest.raises(FileExistsError):
        build_index(documents, out)


def test_index_integrity_validation(documents, tmp_path):
    build_index(documents, tmp_path / "idx")
    path = tmp_path / "idx/chunks.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["text"] = "tampered"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        read_index(tmp_path / "idx")


def test_dense_requires_pinned_model_before_creating_anything(documents, tmp_path):
    with pytest.raises(ValueError, match="revision"):
        build_index(documents, tmp_path / "idx", backend="chroma")
    assert not (tmp_path / "idx").exists()


def test_empty_query_and_invalid_k(documents):
    retriever = LexicalRetriever(chunk_documents(documents))
    assert retriever.search("???", 3) == []
    with pytest.raises(ValueError):
        retriever.search("甲", 0)
