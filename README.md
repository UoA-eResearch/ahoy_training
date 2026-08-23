# ahoymatey_on_gb10 — fine-tune a tiny model to talk like a pirate

Reproducible LoRA fine-tune of **Qwen/Qwen2.5-0.5B-Instruct** (ungated, no HF token
needed) so it answers *everything* in pirate dialect — no system prompt required.
Runs on an NVIDIA DGX GB10 (aarch64 + Blackwell) with no root access.

## How it works

1. **`scripts/setup.sh`** — `uv` venv with PyTorch `cu130` aarch64 wheels (they support the
   GB10's sm_121) + transformers 5 / peft / trl / datasets.
2. **`scripts/make_dataset.py`** — "self-distillation": takes ~1500 real user prompts from
   `databricks/databricks-dolly-15k` and has the ungated **Qwen2.5-7B-Instruct** answer
   them under a pirate system prompt. Filters for pirate markers, writes
   `data/pirate_train.jsonl` and `data/pirate_eval.jsonl` (user/assistant pairs, no system msg).
3. **`scripts/train.py`** — TRL `SFTTrainer`, LoRA r=16 on all linear layers, 3 epochs,
   loss on assistant tokens only. Saves the adapter (`out/pirate-lora`) and a merged
   standalone model (`out/pirate-lora-merged`) you can load with transformers or vLLM.
4. **`scripts/eval.py`** — base vs tuned side-by-side on held-out prompts, with a crude
   pirate-marker hit rate.
5. **`scripts/chat.py`** — interactive chat with the result.

## Run it

From your laptop (edit `scripts/common.sh` for host/key/dir):

```bash
scripts/run_all.sh                         # setup -> dataset -> train -> eval
STEPS="train eval" scripts/run_all.sh      # re-run just some steps
TRAIN_ARGS="--epochs 5 --rank 32" STEPS=train scripts/run_all.sh
```

Or on the GB10 directly:

```bash
cd ~/pirate && source .venv/bin/activate
python scripts/make_dataset.py && python scripts/train.py && python scripts/eval.py
python scripts/chat.py
```

Timings on the GB10: setup ~5 min, dataset ~40 min (7B teacher, HF generate, batch 48),
training ~4.5 min (3 epochs, loss 2.85 → 1.30), eval ~1 min. Result: 11/11 held-out prompts answered in pirate vs 0/11 for the base model.

## Why this works when other pirate tutorials didn't

* Many tutorials train on a handful of hand-written examples (or a bare-completion
  dataset) — too little signal. Here the teacher produces ~1500 diverse, on-topic
  pirate answers, so the student learns style *and* keeps answering the question.
* Training uses the model's own chat template (via TRL) and masks the prompt, so the
  behaviour shows up in normal chat use, without any system prompt.
* The stack is pinned to versions that actually work on aarch64/Blackwell.
