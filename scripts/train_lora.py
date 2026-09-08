"""QLoRA 指令微调：在医疗问答语料上微调 CareBot-8B（LoRA 适配器）。

训练格式（与 app/推理保持一致，纯文本、无 chat template）：
    {instruction}\n患者描述：{ask}\n建议：{answer}
标签只监督“建议：”之后的回答部分。

用法（GPU 机器上，项目根目录）：
    python scripts/train_lora.py \
        --model-dir models/CareBot_Medical \
        --data data/sft_train.json \
        --output outputs/lora-med-v1 \
        --epochs 2 --batch-size 2 --grad-accum 16
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)


def build_parser():
    p = argparse.ArgumentParser(description="QLoRA 微调医疗问答")
    p.add_argument("--model-dir", required=True, help="基座模型目录")
    p.add_argument("--data", required=True, help="sft JSON（instruction/input/output）")
    p.add_argument("--output", required=True, help="LoRA 输出目录")
    p.add_argument("--val-size", type=int, default=200, help="从训练集留出用于验证的条数")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p


def build_text(sample: dict, answer_only: bool = False):
    instruction = (sample.get("instruction") or "").strip()
    ask = (sample.get("input") or "").strip()
    answer = (sample.get("output") or "").strip()
    prefix = f"{instruction}\n患者描述：{ask}\n建议："
    if answer_only:
        return prefix, answer
    return prefix + answer


def tokenize_pair(tokenizer, text: str, prefix: str, max_len: int):
    """返回 (input_ids, labels)，labels 中前缀部分为 -100。"""
    p_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    all_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(p_ids) >= max_len:
        return None
    budget = max_len - len(p_ids)
    answer_ids = all_ids[:budget]
    input_ids = p_ids + answer_ids
    labels = [-100] * len(p_ids) + answer_ids
    return input_ids, labels


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, "r", encoding="utf-8") as fh:
        samples = json.load(fh)
    if len(samples) < args.val_size:
        args.val_size = max(0, len(samples) // 10)
    random.shuffle(samples)
    train_raw = samples[args.val_size:]
    val_raw = samples[:args.val_size]
    print(f"total={len(samples)} train={len(train_raw)} val={len(val_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    def make_ds(raw):
        rows = []
        skipped = 0
        for s in raw:
            prefix, answer = build_text(s, answer_only=True)
            full = prefix + answer
            pair = tokenize_pair(tokenizer, full, prefix, args.max_seq_len)
            if pair is None or len(pair[1]) == 0 or not answer:
                skipped += 1
                continue
            input_ids, labels = pair
            rows.append({"input_ids": input_ids, "labels": labels})
        return rows, skipped

    train_rows, sk1 = make_ds(train_raw)
    print(f"tokenized train={len(train_rows)} skipped={sk1}")
    val_rows, sk2 = make_ds(val_raw)
    print(f"tokenized val={len(val_rows)} skipped={sk2}")
    ds = Dataset.from_list(train_rows + val_rows)
    ds = ds.train_test_split(test_size=len(val_rows), seed=args.seed)
    train_ds, val_ds = ds["train"], ds["test"]

    def collate(batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100)
        attn = (input_ids != tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("val_split.json").write_text(
        json.dumps(val_raw, ensure_ascii=False, indent=1), encoding="utf-8")

    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_torch",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )
    t0 = time.time()
    trainer.train()
    print(f"train done in {(time.time()-t0)/60:.1f} min")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"LoRA saved to {out_dir}")


if __name__ == "__main__":
    main()