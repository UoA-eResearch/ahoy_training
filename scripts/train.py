#!/usr/bin/env python3
"""LoRA SFT of a small instruct model on the pirate dataset (TRL SFTTrainer)."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None, help="HF repo id; omit for the picker menu")
ap.add_argument("--list-models", action="store_true", help="print the model menu and exit")
ap.add_argument("--yes", "-y", action="store_true", help="skip the large-model confirmation")
ap.add_argument("--data", default="data/pirate_train.jsonl")
ap.add_argument("--out", default="out/pirate-lora")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=None, help="per-device batch; default scales with model size")
ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", default=None)
ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
ap.add_argument("--save-merged", dest="save_merged", action="store_true", default=None,
                help="also write a full merged model (default: on up to 7B, off above)")
ap.add_argument("--no-save-merged", dest="save_merged", action="store_false")
a = ap.parse_args()

if a.list_models:
    print(models.format_menu(default_id=models.DEFAULT_STUDENT))
    sys.exit(0)

entry = models.lookup(a.model) if a.model else models.choose_model("student", models.DEFAULT_STUDENT)
models.confirm(entry, "train_est", assume_yes=a.yes)

# registry values are defaults; anything given explicitly on the CLI wins
batch = a.batch if a.batch is not None else entry["batch"]
grad_ckpt = a.grad_ckpt if a.grad_ckpt is not None else entry["grad_ckpt"]
# keep the effective batch at 16 regardless of what fits per device
accum = max(1, 16 // batch)
save_merged = a.save_merged if a.save_merged is not None else (entry["vram_gb"] or 99) <= 20

print(f"training {entry['id']} ({entry['params']}) -- batch {batch} x accum {accum}, "
      f"grad_ckpt={grad_ckpt}, est {entry['train_est']}")

from datasets import load_dataset            # imported late: slow, and not needed for the menu
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ds = load_dataset("json", data_files=a.data, split="train")
print(ds)

cfg = SFTConfig(
    output_dir=a.out,
    num_train_epochs=a.epochs,
    per_device_train_batch_size=batch,
    gradient_accumulation_steps=accum,
    gradient_checkpointing=grad_ckpt,
    gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
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

trainer = SFTTrainer(model=entry["id"], args=cfg, train_dataset=ds, peft_config=lora)
trainer.train()
trainer.save_model(a.out)               # LoRA adapter (records the base model in adapter_config.json)
trainer.processing_class.save_pretrained(a.out)
if save_merged:
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(a.out + "-merged")   # full standalone model (for vLLM etc.)
    trainer.processing_class.save_pretrained(a.out + "-merged")
    print("saved", a.out, "and", a.out + "-merged")
else:
    print("saved", a.out, "(adapter only -- a merged copy of a", entry["params"],
          "model is large; pass --save-merged to write one)")
