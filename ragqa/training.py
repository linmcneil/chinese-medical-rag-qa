"""Answer-only SFT utilities; optional GPU imports live inside train()."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ragqa.data import digest, question_key, write_json
from ragqa.prompts import prompt_ids
from ragqa.types import Document


def tokenize_answer(tokenizer, question: str, answer: str, max_length: int) -> dict:
    """Prefix appears once; EOS is supervised; overlong answers are rejected, not clipped."""
    prefix = prompt_ids(tokenizer, question)
    answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    eos = tokenizer.eos_token_id
    if eos is None or not answer.strip() or not answer_ids:
        raise ValueError("SFT needs an answer and an EOS token")
    target = answer_ids + [eos]
    if len(prefix) + len(target) > max_length:
        raise ValueError("Training example exceeds max_length")
    return {
        "input_ids": prefix + target,
        "labels": [-100] * len(prefix) + target,
        "attention_mask": [1] * (len(prefix) + len(target)),
    }


def pad_batch(rows: list[dict], pad_id: int) -> dict:
    """Mask by sequence lengths, so a real EOS is visible even when PAD == EOS."""
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(r["input_ids"]) for r in rows)
    result = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in rows:
        n = len(row["input_ids"])
        if len(row["labels"]) != n:
            raise ValueError("Label length mismatch")
        result["input_ids"].append(row["input_ids"] + [pad_id] * (length - n))
        result["labels"].append(row["labels"] + [-100] * (length - n))
        result["attention_mask"].append([1] * n + [0] * (length - n))
    return result


def load_split(path: str | Path) -> tuple[list[Document], list[Document], dict]:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Training requires a v2 split manifest")
    splits = []
    for name in ("train", "validation"):
        rows = json.loads((path / f"{name}.json").read_text(encoding="utf-8"))
        if digest(rows) != manifest.get(f"{name}_sha256"):
            raise ValueError(f"{name} split fingerprint mismatch")
        docs = [Document(**row) for row in rows]
        if not docs or [d.id for d in docs] != manifest.get(f"{name}_ids"):
            raise ValueError(f"{name} split IDs mismatch or empty split")
        splits.append(docs)
    if {question_key(d.question) for d in splits[0]} & {question_key(d.question) for d in splits[1]}:
        raise ValueError("Training and validation question groups overlap")
    return splits[0], splits[1], manifest


def train(
    split_dir: str,
    model_id: str,
    output: str,
    *,
    revision: str | None = None,
    epochs: float = 2,
    max_length: int = 1024,
    seed: int = 42,
):
    """QLoRA experiment only: not evidence-grounded RAG training or a medical quality claim."""
    if epochs <= 0 or max_length < 1:
        raise ValueError("epochs and max_length must be positive")
    train_docs, val_docs, manifest = load_split(split_dir)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Choose a new training output directory")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires CUDA")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    kept = {}

    def prepare(documents, name):
        rows, ids, skipped = [], [], []
        for doc in documents:
            try:
                rows.append(tokenize_answer(tokenizer, doc.question, doc.answer, max_length))
                ids.append(doc.id)
            except ValueError as exc:
                if str(exc) != "Training example exceeds max_length":
                    raise
                skipped.append(doc.id)
        if not rows:
            raise ValueError(f"No examples fit in the {name} token budget")
        kept[name] = {"ids": ids, "skipped_ids": skipped}
        return Dataset.from_list(rows)

    # Keep the persisted partition; never concatenate and re-split.
    train_ds, val_ds = prepare(train_docs, "train"), prepare(val_docs, "validation")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        device_map={"": 0},
        torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )

    def collate(rows):
        return {
            k: torch.tensor(v, dtype=torch.long) for k, v in pad_batch(rows, tokenizer.pad_token_id).items()
        }

    destination.mkdir(parents=True)
    write_json(destination / "split_manifest.json", manifest)
    write_json(destination / "tokenized_manifest.json", kept)
    write_json(
        destination / "training_config.json",
        {
            "model": model_id,
            "revision": revision,
            "epochs": epochs,
            "max_length": max_length,
            "seed": seed,
            "purpose": "answer-only SFT without retrieved evidence",
        },
    )
    write_json(destination / "validation.json", [asdict(d) for d in val_docs])
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        args=TrainingArguments(
            output_dir=str(destination / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            report_to=[],
            seed=seed,
            remove_unused_columns=False,
        ),
    )
    trainer.train()
    model.save_pretrained(destination / "adapter")
    tokenizer.save_pretrained(destination / "adapter")
