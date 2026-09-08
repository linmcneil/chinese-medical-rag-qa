"""问答生成：CareBot 医疗模型加载与推理（延迟导入重型依赖）。

原 qa_system.py 的内存优化逻辑存在 bug：
- 8bit 量化同样需要 bitsandbytes，不能“装不上 4bit 就用 8bit”
- 模型下载在 import 时触发；这里收进 ensure/load 函数
本模块统一为：auto -> CUDA 可用且装好 bnb 时用 4bit，否则回退 float16/fp32。
"""
from __future__ import annotations

from pathlib import Path

from ragqa.modelstore import ensure_qa_model, set_hf_mirror


def _pick_load_kwargs(device: str, quantize: str, cuda: bool):
    """决定 from_pretrained 的关键参数，返回 (kwargs, 说明)。"""
    kwargs = {}
    note = ""
    use_quant = quantize in ("auto", "4bit", "8bit")

    if cuda:
        if quantize == "4bit":
            import bitsandbytes  # noqa: F401  提前失败给出明确报错
            kwargs.update(device_map="auto", torch_dtype="auto", load_in_4bit=True)
            note = "4bit 量化 (CUDA)"
        elif quantize == "8bit":
            import bitsandbytes  # noqa: F401
            kwargs.update(device_map="auto", torch_dtype="auto", load_in_8bit=True)
            note = "8bit 量化 (CUDA)"
        elif use_quant:
            try:
                import bitsandbytes  # noqa: F401
                kwargs.update(device_map="auto", torch_dtype="auto", load_in_4bit=True)
                note = "auto -> 4bit 量化 (CUDA)"
            except Exception:
                kwargs.update(device_map="auto", torch_dtype="auto")
                note = "auto -> 未装 bitsandbytes，退化为普通精度 (CUDA)"
        else:
            kwargs.update(device_map="auto", torch_dtype="auto")
            note = f"quantize={quantize} (CUDA 普通精度)"
    else:
        kwargs.update(device_map="cpu", torch_dtype="auto")
        note = "CPU 模式（无量化，速度较慢）"
    return kwargs, note


def load_qa_pipeline(model_dir, device: str = "auto", quantize: str = "auto",
                     max_new_tokens: int = 200, temperature: float = 0.1,
                     top_p: float = 0.9, repetition_penalty: float = 1.15):
    """下载并加载医疗问答模型，返回 HF text-generation pipeline。

    device/quantize 见 ragqa.config；quantize: auto|4bit|8bit|none。
    """
    set_hf_mirror()
    model_path = ensure_qa_model(Path(model_dir))
    print(f"加载模型: {model_path}")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if device == "auto":
        cuda = torch.cuda.is_available()
    elif device == "cuda":
        cuda = True
    else:
        cuda = False

    kwargs, note = _pick_load_kwargs(device, quantize, cuda)
    print(f"加载策略: {note}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return pipe