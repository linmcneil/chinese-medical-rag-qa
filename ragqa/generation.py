"""Lazy generation backends. Model handles are reused for the service lifetime."""

from __future__ import annotations

from typing import Sequence

from ragqa.prompts import clean_output, prompt_ids
from ragqa.types import BackendError, Evidence


class ExtractiveGenerator:
    def generate(self, question: str, evidence: Sequence[Evidence]) -> str:
        return "知识库原文摘录（未经临床审核）：\n" + "\n\n".join(
            f"[{e.citation}] {e.text}" for e in evidence
        )


class TransformersGenerator:
    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        max_new_tokens: int = 256,
        max_input_tokens: int = 4096,
        quantize: str = "none",
        adapter: str | None = None,
    ):
        if device not in ("cpu", "cuda") or quantize not in ("none", "4bit"):
            raise ValueError("Unsupported generation device or quantization")
        if quantize == "4bit" and device != "cuda":
            raise ValueError("4bit generation requires CUDA")
        if max_new_tokens < 1 or max_input_tokens < 1:
            raise ValueError("Token budgets must be positive")
        self.model_id, self.revision, self.device = model, revision, device
        self.quantize, self.adapter = quantize, adapter
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens
        self.model = self.tokenizer = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device == "cuda" and not torch.cuda.is_available():
            raise BackendError("CUDA was requested but is unavailable")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=False
        )
        if not tokenizer.chat_template or tokenizer.eos_token_id is None:
            raise BackendError("Tokenizer must provide a chat template and EOS token")
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        dtype = (
            torch.float32
            if self.device == "cpu"
            else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        )
        kwargs = {
            "torch_dtype": dtype,
            "device_map": self.device,
            "revision": self.revision,
            "trust_remote_code": False,
        }
        if self.quantize == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        model.eval()
        self.tokenizer, self.model = tokenizer, model

    def generate(self, question: str, evidence: Sequence[Evidence]) -> str:
        try:
            self._load()
            import torch

            ids = prompt_ids(self.tokenizer, question, evidence)
            if len(ids) > self.max_input_tokens:
                raise BackendError("Prompt exceeds the input token budget; evidence was not truncated")
            capacity = getattr(self.model.config, "max_position_embeddings", None)
            if isinstance(capacity, int) and len(ids) + self.max_new_tokens > capacity:
                raise BackendError("Prompt plus output exceeds model context capacity")
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            with torch.inference_mode():
                result = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            return clean_output(self.tokenizer.decode(result[0][len(ids) :], skip_special_tokens=True))
        except Exception as exc:
            raise BackendError("Generation failed") from exc
