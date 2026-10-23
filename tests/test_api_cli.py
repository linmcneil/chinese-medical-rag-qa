import json
import os
import subprocess
import sys

from fastapi.testclient import TestClient

from ragqa.api import create_app
from ragqa.factory import create_pipeline
from ragqa.retrieval import build_index


def test_api_and_cli_use_identical_pipeline(documents, tmp_path):
    index = tmp_path / "index"
    build_index(documents, index)
    pipeline = create_pipeline(str(index))
    with TestClient(create_app(pipeline)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/v1/answers", json={"question": "ALPHA-01"})
        assert response.status_code == 200
        api = response.json()
        assert client.post("/v1/answers", json={"question": " "}).status_code == 422
        assert client.post("/v1/answers", json={"question": 123}).status_code == 422
        assert client.post("/v1/answers", json={"question": "x", "unexpected": 1}).status_code == 422
        assert client.post("/v1/answers", json={"question": "x" * 2001}).status_code == 422
        pipeline._lock.acquire()
        try:
            busy = client.post("/v1/answers", json={"question": "ALPHA-01"})
            assert busy.status_code == 503 and busy.headers["Retry-After"] == "1"
        finally:
            pipeline._lock.release()
    run = subprocess.run(
        [sys.executable, "-m", "ragqa", "ask", "--index", str(index), "--question", "ALPHA-01"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert run.returncode == 0, run.stderr
    cli = json.loads(run.stdout)
    for key in ("status", "text", "evidence", "citations", "reason"):
        assert cli[key] == api[key]


def test_cli_help_does_not_import_ml():
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ragqa.cli import parser; parser().format_help(); "
            "assert not {'torch', 'transformers', 'chromadb', 'gradio'} & set(sys.modules)",
        ],
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr


def test_cli_chinese_help_on_legacy_output_encoding():
    run = subprocess.run(
        [sys.executable, "-m", "ragqa", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert run.returncode == 0, run.stderr
    assert "中文医疗 RAG 研究系统 v2" in run.stdout
