# Chinese Medical RAG QA (中文医疗检索增强问答)
[![CI](https://github.com/linmcneil/chinese-medical-rag-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/linmcneil/chinese-medical-rag-qa/actions) · Repo: https://github.com/linmcneil/chinese-medical-rag-qa

A retrieval-augmented generation (RAG) system for Chinese medical Q&A: clean a real
medical dialogue corpus → field-aware chunking → vector indexing → retrieve relevant
knowledge at answer time → generate a grounded answer with an 8B medical LLM (GPU) or
a lightweight Qwen model (CPU).

> For research/demo only. Outputs are **not** medical advice.

## Highlights (short version)
- I built and benchmarked the full RAG pipeline end-to-end: data → chunking → embedding → retrieval → generation → cleaning → evaluation.
- Every stage has a reproducible script, a documented metric, and unit tests (49/49 passing, CI).

## Key numbers (AutoDL RTX 5090, 2026-09-08)
| Experiment | Result |
| --- | --- |
| Retrieval Hit@1 / Hit@3 / Hit@5 | **86.5% / 90.3% / 91.1%** (1000 held-out questions, 8524 chunks) |
| Retrieval latency | **3.5 ms/query** average (GPU) |
| QLoRA fine-tune (vs base) | semantic similarity **0.789 → 0.851**; avg answer length 390 → 262 chars; 2 epochs in ~25 min on RTX 5090 |
| Unit tests | chunking 14/14, data 9/9, logic 26/26 (prompts/retriever/post-processing/rerank, pure CPU) |

## Architecture
```
Medical CSV ──> clean / sample ──> bge-small-zh ──> Chroma (8524 chunks)
                                                  (sha1-resumable indexing)
User question ──> Top-k retrieval
                    ├─ context cleaning (keep answer body only)
                    ├─ keyword relevance gate (auto-degrade on bad hits)
                    ▼
           8B CareBot (GPU, 4bit)  /  Qwen1.5B (CPU)  /  snippet-only mode
                    ▼
           generation post-processing (de-dup, cut run-on tails, length cap)
                    ▼
              Answer + source snippets
```

## Engineering highlights
- **Resumable indexing**: content-sha1 dedupe; interrupted builds continue on re-run.
- **Field-aware chunking**: honors 【department】【topic】【question】【answer】 boundaries so long
  docs never mix topics; 14 unit tests cover boundaries/overflow/dedup.
- **Model loading degrades honestly**: 4-bit/8-bit needs bitsandbytes; otherwise it falls back to
  normal precision/CPU instead of fake-quantizing. Heavy imports are lazy.
- **RAG quality guard (new)**: retrieved snippets are cleaned and relevance-gated before injection;
  irrelevant top hits automatically degrade to model-direct answers instead of poisoning the output.
- **Answer post-processing (new)**: strips prompt replay, “reference/greeting/AI-disclaimer” tails,
  deduplicates sentences, and caps length — kills the boilerplate style found in the source corpus.
- **Optional reranker (new)**: `--rerank` re-ranks vector top-N candidates with bge-reranker;
  `scripts/eval_retrieval.py --rerank` reports both pure-vector and vector+rerank Hit@k in one run.
- **A/B experiment**: base 8B vs QLoRA tuned model on held-out generation set (semantic sim & Rouge).
  The LoRA is closer to the corpus answers but also inherits its marketing-style politeness, so the
  shipped demo defaults to base + RAG with LoRA kept as an experimental toggle.

## Quickstart
CPU (no GPU):
```bash
python -m venv .venv-cpu
# pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
python document_processer.py --device cpu
python app_local.py --cpu-model Qwen/Qwen2.5-1.5B-Instruct   # or: snippet-only mode in UI
```
Docker (CPU demo, optional):
```bash
docker compose up --build      # open http://127.0.0.1:7861
```

As a Python package (optional): `pip install -e .` (lightweight core only; heavy deps stay in `requirements-*.txt` / extras).

GPU (full 8B version): see Chinese README (`README.md`) or `qa_system.py --help`.

## Evaluation notes (honest)
- Hit@k: eval questions are sampled from the knowledge base with ground-truth `rid`; it measures
  chunk→embed→retrieve consistency, **not** answer quality.
- Answer quality is currently compared on a held-out generation set (semantic similarity, Rouge,
  length) plus manual spot checks; LLM-as-judge and clinical review are future work.

## Data & license
- Corpus: [Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data) (research use; license of that repo applies). A small sample is bundled under `data/raw/`.
- Models: BAAI/CareBot_Medical_multi-llama3-8b-instruct (GPU), Qwen/Qwen2.5-1.5B-Instruct (CPU, Apache-2.0).
- Code: MIT (see LICENSE).