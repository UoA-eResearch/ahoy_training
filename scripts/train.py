#!/usr/bin/env python3
"""LoRA SFT of a small instruct model on the pirate dataset (TRL SFTTrainer)."""
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--data", default="data/pirate_train.jsonl")
ap.add_argument("--out", default="out/pirate-lora")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()

ds = load_dataset("json", data_files=a.data, split="train")
print(ds)

cfg = SFTConfig(
    output_dir=a.out,
    num_train_epochs=a.epochs,
    per_device_train_batch_size=a.batch,
    gradient_accumulation_steps=2,
    learning_rate=a.lr,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    logging_steps=10,
    save_strategy="no",
    max_length=512,
    assistant_only_loss=True,   # loss only on the pirate answer tokens
    report_to="none",
    seed=0,
)
lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=ds, peft_config=lora)
trainer.train()
trainer.save_model(a.out)               # LoRA adapter
merged = trainer.model.merge_and_unload()
merged.save_pretrained(a.out + "-merged")   # full standalone model (for vLLM etc.)
trainer.processing_class.save_pretrained(a.out + "-merged")
print("saved", a.out, "and", a.out + "-merged")
