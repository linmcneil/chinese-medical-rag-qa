"""提示词构造与回答后处理（纯函数，可单测）。"""

def build_rag_prompt(context: str, question: str) -> str:
    """按原 qa_system 语义构造医疗 RAG 提示词。"""
    return (
        "你是专业的医疗顾问，需根据以下医疗知识回答问题。\n"
        f"医疗知识：{context}\n"
        f"问题：{question}\n"
        "请用简洁、准确的中文回答，基于提供的医疗知识，避免猜测。"
    )


def format_context(documents, rids=None) -> str:
    """把检索到的文档拼成上下文；有来源 id 时附加【来源】标注。"""
    blocks = []
    for i, doc in enumerate(documents):
        if rids is not None and i < len(rids) and rids[i]:
            blocks.append(f"【来源 {i + 1}】{rids[i]}\n{doc}")
        else:
            blocks.append(doc)
    return "\n\n".join(blocks)


def extract_answer(generated: str, prompt: str) -> str:
    """从 text-generation 输出里剥离提示词前缀。"""
    text = generated or ""
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    # 兜底：去掉提示词末行之后的内容
    tail = prompt.rsplit("\n", 1)[-1]
    idx = text.rfind(tail)
    if idx >= 0:
        return text[idx + len(tail):].strip()
    return text.strip()