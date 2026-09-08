# -*- coding: utf-8 -*-
"""本地 CPU 版网页问答（Gradio，轻量发行）。

两种回答方式：
  - 轻量生成：Qwen2.5-1.5B-Instruct 在 CPU 上跑（首次自动下载约 3GB，建议 8G 内存）；
  - 仅检索摘录：不加载大模型，直接给出最相关知识片段里的解答。

启动：
    python app_local.py                  # 默认 http://127.0.0.1:7861
    python app_local.py --cpu-model Qwen/Qwen2.5-0.5B-Instruct   # 更轻的模型
依赖见 requirements-cpu.txt（先装 CPU 版 PyTorch，文件头有说明）。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"                 # 嵌入模型存放处
CHROMA_DIR = BASE / "chroma_data"
TOP_K = 3
MAX_TOP_DISTANCE = 0.4

MODE_CHAT = "chat"
MODE_SNIPPET = "snippet"

_cpu_model = None
_tokenizer = None
_retriever = None
_llm_id = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from ragqa.modelstore import ensure_embedding_model, set_hf_mirror
        from ragqa.retriever import ChromaRetriever
        set_hf_mirror()
        embed_dir = ensure_embedding_model(MODEL_DIR)
        _retriever = ChromaRetriever(
            chroma_dir=str(CHROMA_DIR), model_dir=str(embed_dir.parent),
            collection="knowledges", device="cpu", top_k=TOP_K,
        ).connect()
    return _retriever


def _retrieve(question):
    """返回 (上下文清洗片段列表, 状态文本)。"""
    hits = get_retriever().query(question, top_k=TOP_K)
    if not hits:
        return [], "（知识库无命中，请确认已运行 document_processer.py 建索引）"
    from ragqa.inference import clean_context_docs, is_context_relevant
    raw_docs = [h.get("document", "") or "" for h in hits]
    cleaned = clean_context_docs(raw_docs)
    top_dist = float(hits[0].get("distance") or 1.0)
    if cleaned and top_dist <= MAX_TOP_DISTANCE \
            and is_context_relevant(question, raw_docs[0]):
        status = (f"检索注入：命中 {len(cleaned)} 条（距离 {top_dist:.3f}）")
        return cleaned, status
    status = ("检索相关性不足，已自动切换为模型直接回答"
              "（未把无关语料喂给模型）")
    return [], status


def snippet_answer(cleaned):
    """无大模型模式：直接把最相关的知识片段解答作为回答展示。"""
    if not cleaned:
        return "未检索到足够相关的知识片段，建议换一种问法，或使用“轻量生成”模式。"
    text = cleaned[0]
    if len(text) > 220:
        cut = max((i for i in (text.find("。"), text.find("！"), text.find("？"))
                   if 0 < i <= 220), default=220)
        text = text[:cut + 1]
    return f"{text}\n\n（以上为知识库原解答摘录，未经过大模型改写）"


def load_cpu_llm(model_id):
    global _cpu_model, _tokenizer, _llm_id
    if _cpu_model is not None and _llm_id == model_id:
        return
    from ragqa.localgen import load_cpu_model
    _cpu_model, _tokenizer = load_cpu_model(model_id)
    _llm_id = model_id


def answer(question, mode, max_new, model_id):
    from ragqa.localgen import generate_local
    if not question.strip():
        return "请输入问题。", ""
    t0 = time.time()
    cleaned = []
    status = ""
    try:
        cleaned, status = _retrieve(question)
    except Exception as e:  # noqa: BLE001
        status = f"（RAG 不可用：{e}）"
    context = "\n\n".join(cleaned[:TOP_K]) if cleaned else None

    if mode == MODE_SNIPPET:
        ans_text = snippet_answer(cleaned)
        model_tag = "仅检索摘录（无大模型）"
    else:
        t_load = time.time()
        load_cpu_llm(model_id)
        load_s = time.time() - t_load
        ans_text = generate_local(_cpu_model, _tokenizer, question,
                                  max_new_tokens=int(max_new), context=context,
                                  temperature=0.3)
        model_tag = f"CPU 轻量生成（{model_id.rsplit('/', 1)[-1]}）"
        if load_s > 5:
            status += f"；模型加载耗时 {load_s:.0f}s（仅首次）"

    dt = time.time() - t0
    status = f"模式={model_tag} | 耗时={dt:.1f}s\n{status}"
    return ans_text, status


def build_ui(default_model):
    with gr.Blocks(title="中文医疗 RAG 问答（本地 CPU 版）") as demo:
        gr.Markdown(
            "# \U0001FA79 中文医疗问答 · 本地 CPU 版\n"
            "检索：bge-small-zh + Chroma（知识库 5000 条）· "
            "生成：可选 Qwen2.5 CPU 小模型\n\n"
            "回答由 AI 生成，仅供学习演示，不能替代医生诊断。"
        )
        with gr.Row():
            mode = gr.Radio([MODE_CHAT, MODE_SNIPPET],
                            value=MODE_CHAT,
                            label="回答方式（chat=轻量生成，snippet=仅检索摘录）")
            max_new = gr.Slider(60, 300, value=180, step=10,
                                label="生成长度上限")
        q = gr.Textbox(lines=2, placeholder="输入医疗问题，例如：高血压患者能吃柚子吗？",
                       label="问题")
        btn = gr.Button("提问", variant="primary")
        ans = gr.Textbox(lines=8, label="回答")
        meta = gr.Textbox(lines=4, label="状态", interactive=False)
        gr.Examples([["高血压患者能吃柚子吗？"], ["经常胃痛反酸是什么问题？"]], inputs=q)
        btn.click(answer, [q, mode, max_new, gr.State(default_model)], [ans, meta])
    return demo


def main():
    ap = argparse.ArgumentParser(description="本地 CPU 版 RAG 医疗问答")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--cpu-model",
                    default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="CPU 小模型 id（可用 Qwen/Qwen2.5-0.5B-Instruct）")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    demo = build_ui(args.cpu_model)
    demo.launch(server_name="127.0.0.1", server_port=args.port,
                share=args.share)


if __name__ == "__main__":
    main()