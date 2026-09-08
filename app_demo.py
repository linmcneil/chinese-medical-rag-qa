# -*- coding: utf-8 -*-
"""医疗问答网页版（发行演示入口）。

功能：
    - 模型：基座 CareBot vs LoRA 微调（默认基座，LoRA 仅作对比实验）；
    - 检索：可开关 RAG（bge-small-zh + Chroma top-3，默认开启）；
      注入前会做“清洗片段 + 关键词相关性门槛”，命中不相关时自动降级为
      模型直接回答，避免把脏语料带偏答案；
    - 展示：回答 + 命中的知识片段 + 状态行

启动（AutoDL 上）：
    bash start_release.sh          # 监听 0.0.0.0:7860
本机浏览器访问方式见 release/ 里的说明与 .bat。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models" / "CareBot_Medical"
LORA_DIR = BASE / "outputs" / "lora-med-v1"
CHROMA_DIR = BASE / "chroma_data"
TOP_K = 3
# Chroma 返回的 L2 距离阈值：超过说明顶配片段都不够近，不值得注入
MAX_TOP_DISTANCE = 0.4

_model = None
_tokenizer = None
_lora_loaded = False
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from ragqa.modelstore import ensure_embedding_model, set_hf_mirror
        from ragqa.retriever import ChromaRetriever
        set_hf_mirror()
        embed_dir = ensure_embedding_model(BASE / "models")
        _retriever = ChromaRetriever(
            chroma_dir=CHROMA_DIR, model_dir=embed_dir.parent,
            collection="knowledges", device="cuda", top_k=TOP_K,
        ).connect()
    return _retriever


def load_model(use_lora: bool):
    """加载模型（4bit），lora 缺失时自动回退基座。"""
    global _model, _tokenizer, _lora_loaded
    if _model is not None and use_lora == _lora_loaded:
        return
    from ragqa.inference import load_generation_model
    lora_path = str(LORA_DIR) if (use_lora and LORA_DIR.exists()) else None
    _model, _tokenizer = load_generation_model(str(MODEL_DIR), lora_path)
    _lora_loaded = use_lora


def answer(question, use_rag, use_lora, max_new):
    from ragqa.inference import (clean_context_docs, generate_answer,
                                 is_context_relevant)
    if not question.strip():
        return "请输入问题。", ""
    t0 = time.time()
    load_model(bool(use_lora))
    sources = ""
    context = None
    rag_on = bool(use_rag)
    if rag_on:
        try:
            hits = get_retriever().query(question, top_k=TOP_K)
            if hits:
                raw_docs = [h.get("document", "") or "" for h in hits]
                cleaned = clean_context_docs(raw_docs)
                top_dist = float(hits[0].get("distance") or 1.0)
                if (cleaned and top_dist <= MAX_TOP_DISTANCE
                        and is_context_relevant(question, raw_docs[0])):
                    context = "\n\n".join(cleaned[:TOP_K])
                    sources = "\n---\n".join(
                        f"[{i + 1}] {h['document'][:180]}"
                        for i, h in enumerate(hits))
                else:
                    sources = ("检索命中相关性不足，已自动切换为模型直接回答，"
                               "未把无关语料喂给模型。原始片段如下：\n"
                               + "\n---\n".join(
                                   f"[{i + 1}] {h['document'][:180]}"
                                   for i, h in enumerate(hits)))
                    context = None
        except Exception as e:  # noqa: BLE001
            sources = f"（RAG 不可用：{e}）"
    model_tag = "LoRA 微调" if use_lora else "基座 CareBot"
    rag_tag = "RAG 注入" if context else ("检索降级" if rag_on and sources else "无 RAG")
    answer_text = generate_answer(_model, _tokenizer, question,
                                  max_new_tokens=int(max_new), context=context,
                                  temperature=0.1)
    dt = time.time() - t0
    status = f"模型={model_tag} | 模式={rag_tag} | 耗时={dt:.1f}s"
    return answer_text, f"{status}\n\n检索片段：\n{sources if sources else '（未启用/无命中）'}"


def build_ui():
    with gr.Blocks(title="中文医疗 RAG 问答（发行演示版）") as demo:
        gr.Markdown(
            "# \U0001FA79 中文医疗问答（RAG + 微调实验版）\n"
            "数据：5000 条医疗问答微调 · 检索库：bge-small-zh + Chroma · 模型：CareBot-8B (4bit)\n\n"
            "回答由 AI 生成，仅供学习演示，不能替代医生诊断。"
        )
        with gr.Row():
            use_lora = gr.Checkbox(
                value=False,
                label="使用 LoRA 微调版（默认关闭：微调学到的是语料客套风格，基座+RAG 更稳）")
            use_rag = gr.Checkbox(value=True, label="RAG 检索增强（推荐开启，会自动降级保护）")
            max_new = gr.Slider(80, 800, value=250, step=10,
                                label="生成长度上限（默认250，回答被截断再调大）")
        q = gr.Textbox(lines=2, placeholder="输入医疗问题，例如：高血压患者能吃柚子吗？",
                       label="问题")
        btn = gr.Button("提问", variant="primary")
        ans = gr.Textbox(lines=8, label="回答")
        meta = gr.Textbox(lines=6, label="状态与检索片段", interactive=False)
        gr.Examples([["高血压患者能吃柚子吗？血压高的人日常饮食要注意什么？"],
                     ["经常胃痛反酸是什么问题？"],
                     ["孩子咳嗽一周不见好怎么办？"]], inputs=q)
        btn.click(answer, [q, use_rag, use_lora, max_new], [ans, meta])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()