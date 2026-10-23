"""Optional UI adapter; the same pipeline and source contract as API/CLI."""

from __future__ import annotations

from ragqa.types import InputError


def build_ui(pipeline):
    import gradio as gr

    def answer(question):
        try:
            result = pipeline.run(question)
            return result.text, result.to_dict()
        except InputError as exc:
            return str(exc), {"status": "invalid_input"}

    with gr.Blocks(title="中文医疗资料检索实验", analytics_enabled=False) as demo:
        gr.Markdown("# 中文医疗资料检索实验\n仅供学习研究。资料和回答未经临床审核，不能替代医生诊断。")
        question = gr.Textbox(label="问题", lines=3)
        button = gr.Button("检索并回答", variant="primary")
        text = gr.Textbox(label="回答", lines=8)
        detail = gr.JSON(label="来源、状态与耗时")
        button.click(answer, question, [text, detail])
    return demo
