#!/usr/bin/env python3
"""LoRA SFT of a small instruct model on a chat-formatted dataset (TRL SFTTrainer).

Shared by the pirate and ADE tracks -- same recipe, different --data/--out.
Driven by scripts/tune.py; the flags exist for one-off experiments.
"""
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
ap.add_argument("--two-node", dest="two_node", action="store_true",
                help="EXPERIMENTAL: bf16 LoRA sharded across a cabled GB10 pair "
                     "instead of 4-bit QLoRA on this box (see README)")
a = ap.parse_args()

if a.list_models:
    print(models.format_menu(default_id=models.DEFAULT_STUDENT))
    sys.exit(0)

entry = models.lookup(a.model) if a.model else models.choose_model("student", models.DEFAULT_STUDENT)
under_torchrun = "RANK" in os.environ
if not under_torchrun:  # a rank re-entering below was already confirmed once
    models.confirm(entry, "train_est", assume_yes=a.yes)

if entry["tier"] == "xl" and a.two_node and not under_torchrun:
    # EXPERIMENTAL bf16 path: relaunch this same script on both GB10s of the
    # pair (rank 0 here, rank 1 on the worker over SSH). The ranks come back
    # through here with RANK set and drop into the DeepSpeed branch below.
    import pair
    argv = ["scripts/train.py", "--model", entry["id"], "--yes", "--two-node",
            "--data", a.data, "--out", a.out, "--epochs", str(a.epochs),
            "--lr", str(a.lr), "--rank", str(a.rank)]
    if a.batch is not None:
        argv += ["--batch", str(a.batch)]
    if a.grad_ckpt is not None:
        argv += ["--grad-ckpt" if a.grad_ckpt else "--no-grad-ckpt"]
    sys.exit(pair.run_torchrun(argv))

# registry values are defaults; anything given explicitly on the CLI wins
batch = a.batch if a.batch is not None else entry["batch"]
grad_ckpt = a.grad_ckpt if a.grad_ckpt is not None else entry["grad_ckpt"]
# keep the effective batch at 16 regardless of what fits per device (and, on a
# pair, regardless of how many devices there are)
world = int(os.environ.get("WORLD_SIZE", "1"))
accum = max(1, 16 // (batch * world))
save_merged = a.save_merged if a.save_merged is not None else (entry["vram_gb"] or 99) <= 20

ds_config = None
quant_config = None
if entry["tier"] == "xl" and not a.two_node:
    # QLoRA: the frozen 72B loads 4-bit NF4 (~40 GB, fits one box easily);
    # the LoRA adapter itself still trains in bf16 on top. Same recipe.
    import torch
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
if entry["tier"] == "xl" and a.two_node:
    # ZeRO-3 shards the frozen bf16 weights across the two nodes (~75 GB each)
    # and re-gathers each layer over the RoCE link on demand. zero.Init makes
    # from_pretrained partition while loading, so neither box ever has to hold
    # the full ~150 GB.
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            # on a GB10 "CPU" and "GPU" memory are the same silicon, so param
            # offload costs little bandwidth -- but it moves the ~75 GB shard
            # out of the unswappable CUDA pool into pageable memory the kernel
            # can reclaim, which is what keeps the box responsive at the edge
            # (pin_memory would make it unswappable again -- keep it False)
            "offload_param": {"device": "cpu", "pin_memory": False},
            # explicit small buckets: on unified memory every gathered byte is
            # RAM the OS loses, so keep the live gathered working set tight
            "stage3_max_live_parameters": 3e8,
            "stage3_max_reuse_distance": 3e8,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
    }
    save_merged = False  # merging would gather all 150 GB onto one box

mode = "QLoRA 4-bit" if quant_config else ("bf16 ZeRO-3 x2 boxes" if ds_config else "LoRA bf16")
print(f"training {entry['id']} ({entry['params']}) [{mode}] -- batch {batch} x {world} "
      f"node(s) x accum {accum}, grad_ckpt={grad_ckpt}, est {entry['train_est']}")

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
    assistant_only_loss=True,   # loss only on the assistant's answer tokens
    report_to="none",
    seed=0,
    deepspeed=ds_config,        # None on one box; ZeRO-3 across a pair
    model_init_kwargs=(
        # device_map matters: without it the loader accumulates the full bf16
        # checkpoint in CPU RAM before quantizing (OOM); with it each tensor
        # streams to the GPU and quantizes immediately
        {"dtype": "bfloat16", "quantization_config": quant_config, "device_map": "cuda"}
        if quant_config else {"dtype": "bfloat16"} if ds_config else None),
    # the sharded 150 GB load can stall one rank for a long time (page-cache
    # thrash on unified memory); the default 10-30 min collective timeout
    # SIGABRTs the waiting rank, so give the pair hours, not minutes
    ddp_timeout=10800,
)
lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

trainer = SFTTrainer(model=entry["id"], args=cfg, train_dataset=ds, peft_config=lora)
if ds_config:
    # the sharded load leaves tens of GB of transients in the CUDA caching
    # allocator; on unified memory that cache is RAM the OS never gets back,
    # so hand it back before training allocates its own working set
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
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
