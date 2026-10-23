# Chinese Medical RAG QA v2

A shared evidence-oriented pipeline for CLI, FastAPI and Gradio. Start with the dependency-free lexical/extractive baseline; optionally add Chroma, BGE reranking and Transformers generation.

```bash
python -m pip install -e ".[api,dev]"
python -m ragqa build --data examples/knowledge.json --index artifacts/demo-v2
python -m ragqa ask --index artifacts/demo-v2 --question "演示甲的资料编号是什么？"
python -m ragqa serve --index artifacts/demo-v2
python -m pytest -q
```

See [README.md](README.md) for complete commands and [docs/MIGRATION.md](docs/MIGRATION.md) for breaking changes. v1 raw data and results are retained as historical artifacts, not v2 performance claims. Training and evaluation correctness defects invalidate treating the old LoRA figures as a clean held-out comparison.

All chunks carry answer text, parent question, provenance and offsets. Low-evidence and backend-failure paths do not fall back to unconstrained generation. Citation IDs are validated against the supplied evidence; this does **not** establish entailment or medical correctness. The corpus and outputs are not clinically reviewed. Research use only.
