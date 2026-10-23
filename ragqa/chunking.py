"""Answer-bearing chunks; question/title are retrieval context, never evidence text."""

from __future__ import annotations

from ragqa.data import digest
from ragqa.types import Chunk, Document


def chunk_documents(documents: list[Document], max_chars: int = 400) -> list[Chunk]:
    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    chunks = []
    for doc in documents:
        start = 0
        while start < len(doc.answer):
            end = min(start + max_chars, len(doc.answer))
            if end < len(doc.answer):
                # Prefer a complete sentence in the latter half of the window.
                boundaries = [
                    i + 1 for i in range(start + max_chars // 2, end) if doc.answer[i] in "。！？；.!?;\n"
                ]
                if boundaries:
                    end = boundaries[-1]
            text = doc.answer[start:end]
            if text.strip():
                cid = "chunk_" + digest([doc.id, start, end, text])[:24]
                chunks.append(
                    Chunk(cid, doc.id, doc.question, text, doc.title, doc.department, doc.source, start, end)
                )
            start = end
    return chunks
