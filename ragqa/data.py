"""Explicit local ingestion, stable IDs, and one persisted SFT split."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ragqa.types import Document


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Text fields must be strings")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def question_key(question: str) -> str:
    return "".join(c for c in normalize(question).casefold() if c.isalnum())


def make_document(
    question: str, answer: str, title: str = "", department: str = "", source: str = ""
) -> Document:
    q, a, t, d = map(normalize, (question, answer, title, department))
    if not q or not a or not question_key(q):
        raise ValueError("Each record needs a nonempty question and answer")
    # Source/order changes do not change content identity.
    rid = "doc_" + digest([q, a, t, d])[:24]
    return Document(rid, q, a, t, d, source)


def read_documents(path: str | Path, encoding: str = "utf-8-sig") -> list[Document]:
    """Read canonical/legacy JSON or Toyhom CSV. No implicit downloads or truncation."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding=encoding, newline="") as f:
            rows = list(csv.reader(f))
        aliases = {"科室": "department", "标题": "title", "问题": "ask", "回答": "answer"}
        header = [aliases.get(c.strip(), c.strip().lower()) for c in rows[0]] if rows else []
        if {"department", "title", "ask", "answer"}.issubset(header):
            names, rows = header, rows[1:]
        else:
            names = ["department", "title", "ask", "answer"]
        records = []
        for number, row in enumerate(rows, 1):
            if not row or not any(c.strip() for c in row):
                continue
            if len(row) != len(names):
                raise ValueError(f"CSV record {number}: expected {len(names)} fields")
            records.append(dict(zip(names, row)))
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input JSON must be an array")
    result: dict[str, Document] = {}
    for number, row in enumerate(records, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Record {number} must be an object")
        if "question" in row:
            q, a = row["question"], row.get("answer", "")
        elif "ask" in row:
            q, a = row["ask"], row.get("answer", "")
        elif row.get("input"):
            q, a = row["input"], row.get("output", "")
        else:
            q, a = row.get("instruction", ""), row.get("output", "")
        try:
            doc = make_document(
                q,
                a,
                row.get("title", ""),
                row.get("department", ""),
                row.get("source") or f"{path.name}#record={number}",
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid record {number}: {exc}") from exc
        if not isinstance(doc.source, str):
            raise ValueError(f"Record {number}: source must be a string")
        result.setdefault(doc.id, doc)
    if not result:
        raise ValueError("Input contains no usable records")
    return sorted(result.values(), key=lambda d: d.id)


def split_documents(
    documents: Iterable[Document], val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[Document], list[Document]]:
    """Group normalized identical questions before splitting. Paraphrases need curation."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[str, list[Document]] = {}
    for doc in sorted(documents, key=lambda d: d.id):
        groups.setdefault(question_key(doc.question), []).append(doc)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("At least two distinct question groups are required")
    random.Random(seed).shuffle(keys)
    n = max(1, min(len(keys) - 1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:n])
    train, val = [], []
    for key, docs in groups.items():
        (val if key in val_keys else train).extend(docs)
    return train, val


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_split(
    documents: list[Document], output: str | Path, val_fraction: float = 0.1, seed: int = 42
) -> None:
    import os
    import shutil
    import tempfile

    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new split directory; existing splits are immutable")
    train, val = split_documents(documents, val_fraction, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".split-", dir=output.parent))
    try:
        for name, rows in (("train", train), ("validation", val)):
            write_json(stage / f"{name}.json", [asdict(d) for d in rows])
        write_json(
            stage / "manifest.json",
            {
                "schema_version": 2,
                "seed": seed,
                "val_fraction": val_fraction,
                "grouping": "normalized_question",
                "train_ids": [d.id for d in train],
                "validation_ids": [d.id for d in val],
                "train_sha256": digest([asdict(d) for d in train]),
                "validation_sha256": digest([asdict(d) for d in val]),
            },
        )
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
