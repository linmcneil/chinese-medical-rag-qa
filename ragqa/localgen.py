# -*- coding: utf-8 -*-
"""本地 CPU 轻量生成：用小型指令模型跑 RAG 问答。

主发行版（GPU）用 8B CareBot；本模块面向没有显卡的环境，
默认加载 Qwen/Qwen2.5-1.5B-Instruct（Apache-2.0，约 3GB，8G 内存的 CPU 可跑）。

与 ragqa.inference 共用同一套生成后清理（去客套/去复读/截断跑题尾巴），
保证本地 CPU 版与 GPU 版回答观感一致。
"""
from __future__ import annotations

from typing import Optional

import torch

CPU_LLM_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"

_SYSTEM = (
    "你是一名专业的中文医生助手。请根据患者的描述直接给出结论和可执行的日常建议："
    "先说结论，再给 2~3 条要点。不要复述问题，不要客套话，"
    "不要写“仅供参考”、“祝您早日康复”之类的结尾。"
)

_GUIDE = (
    "请用一段简洁的中文直接给出建议（正文不超过250字），说完就停。"
    "直接以“建议：”开头输出正文，不要输出开场白或客套话。"
)


def load_cpu_model(model_id: Optional[str] = None):
    """加载 CPU 小模型；首次运行会自动下载（HF，约 3GB，可用镜像加速）。"""
    from ragqa.modelstore import set_hf_mirror
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_hf_mirror()
    mid = model_id or CPU_LLM_DEFAULT
    tokenizer = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    return model, tokenizer


def build_messages(ask: str, context: Optional[str] = None):
    """构造 chat 消息：系统约束 + 用户描述（附可选检索资料）。"""
    user_parts = []
    if context:
        user_parts.append(
            "以下是从医疗知识库检索到的资料，可能与问题相关也可能无关，"
            "请自行判断后采用：\n" + context.strip())
    user_parts.append(f"患者描述：{ask}")
    user_parts.append(_GUIDE)
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def generate_local(model, tokenizer, ask: str, max_new_tokens: int = 180,
                   context: Optional[str] = None, temperature: float = 0.3) -> str:
    """CPU 生成回答：聊天模板 → 采样 → 与 GPU 版同一套后处理。"""
    from ragqa.inference import clean_generated

    messages = build_messages(ask, context=context)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
    if raw.startswith("建议："):
        raw = raw[len("建议："):].strip()
    text = clean_generated(raw)
    # 小模型常输出 markdown 加粗，如 **建议:**；去掉以免观感不一致
    text = text.replace("**", "").strip()
    return text