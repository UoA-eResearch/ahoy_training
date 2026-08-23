# ahoymatey_on_gb10 — fine-tune a tiny model to talk like a pirate

Reproducible LoRA fine-tune of **Qwen/Qwen2.5-0.5B-Instruct** (ungated, no HF token
needed) so it answers *everything* in pirate dialect — no system prompt required.
Runs on an NVIDIA DGX GB10 (aarch64 + Blackwell) with no root access.

## Procedure

Everything below runs **on the GB10** over SSH. Swap in your own host/user.

```bash
ssh -i ~/.ssh/id_rsa smat924@lais01.cer.auckland.ac.nz
```

### 1. Get the code onto the box

```bash
git clone <this-repo-url> ~/pirate
cd ~/pirate
```

(Or, from your laptop instead of cloning: `scripts/run_all.sh` rsyncs `scripts/`
over and drives every step remotely — see [Alternative: drive it from your laptop](#alternative-drive-it-from-your-laptop).)

### 2. Set up the environment

Creates a `uv` venv with CUDA-enabled PyTorch (the `cu130` aarch64 wheels, which
support the GB10's `sm_121` GPU) plus transformers/peft/trl/datasets. No root
required.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Expect ~5 minutes. It ends by printing your GPU name and compute capability —
confirm it says `NVIDIA GB10`.

### 3. Build the pirate dataset

This is "self-distillation": ~1500 real user prompts from
`databricks/databricks-dolly-15k` (ungated) are answered by a larger ungated
model, **Qwen2.5-7B-Instruct**, under a pirate system prompt. Answers that don't
actually sound piratey get filtered out.

```bash
python scripts/make_dataset.py
```

Takes ~40 minutes (batched generation on the 7B teacher). Produces
`data/pirate_train.jsonl` and `data/pirate_eval.jsonl` — plain
`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` rows, with
**no system message**, so the student has to internalize the pirate voice
rather than being told to use it.

It asks which teacher model to use from a numbered menu (default 7B); pass
`--teacher <hf-repo-id>` to skip the prompt. Other useful flags: `--n 500` for a
quick/cheap run, `--batch 16` if you're memory constrained, `--list-models` to
see the menu without running anything.

### 4. Fine-tune

LoRA (rank 16, all linear layers) supervised fine-tune of the student model
using TRL's `SFTTrainer`, with loss computed only on assistant tokens.

```bash
python scripts/train.py
```

It opens a numbered menu of student models — see
[Choosing a model](#choosing-a-model). The default (0.5B) takes ~4.5 minutes for
3 epochs on the GB10. Saves two things:
- `out/pirate-lora` — the LoRA adapter only
- `out/pirate-lora-merged` — a full standalone model (load with `transformers`
  or `vLLM`, no PEFT dependency needed). Skipped above 7B, where the merged copy
  is tens of GB; pass `--save-merged` to force it, or `--no-save-merged` to skip it.

Useful flags: `--epochs 5`, `--rank 32`, `--lr 1e-4`, `--model <hf-repo-id>` to
bypass the menu, `-y` to auto-accept the large-model warning.

### 5. Evaluate

Runs both the base model and the fine-tuned model on held-out prompts
(including a few off-dataset sanity questions) and prints them side by side,
plus a crude "does the reply contain pirate words" hit rate.

```bash
python scripts/eval.py
```

### 6. Chat with it

```bash
python scripts/chat.py                       # defaults to out/pirate-lora-merged
python scripts/chat.py out/pirate-lora        # or point it at a specific model/adapter dir
```

## Choosing a model

Both `make_dataset.py` (the teacher) and `train.py` (the student) show a numbered
menu when run interactively. Every model listed downloads from HuggingFace
anonymously — no `HF_TOKEN`, no license click-through. That's why the ladder is
Qwen throughout: the Llama, Gemma and Mistral instruct repos are all gated.

```bash
python scripts/train.py --list-models
```

| # | Model | Params | ~Memory | ~Train | ~Generate |
|---|---|---|---|---|---|
| 1 | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | 3 GB | ~5 min | ~10 min |
| 2 | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 6 GB | ~10 min | ~20 min |
| 3 | `Qwen/Qwen2.5-3B-Instruct` | 3B | 10 GB | ~15 min | ~25 min |
| 4 | `Qwen/Qwen2.5-7B-Instruct` | 7B | 20 GB | ~30 min | ~40 min |
| 5 | `Qwen/Qwen2.5-14B-Instruct` | 14B | 36 GB | ~1 h | ~1.3 h |
| 6 | `Qwen/Qwen2.5-32B-Instruct` | 32B | 74 GB | ~2.5 h | ~3 h |
| 7 | `Qwen/Qwen2.5-72B-Instruct` | 72B | 150 GB | won't fit | won't fit |
| 0 | any other HuggingFace repo id | | | | |

Only the 0.5B train time and the 7B generate time are measured; the rest are
estimates. Options 5 and 6 ask for confirmation before downloading, and option 7
is refused outright — 72B needs ~150 GB in bf16 against the GB10's ~110 GB
usable, so it would need 4-bit QLoRA on one box or bf16 LoRA split across two
GB10s with pipeline parallelism, neither of which is wired up here.

Batch size and gradient checkpointing scale automatically with the model you
pick (the 152k-token vocabulary makes the logits tensor the binding constraint at
the large end), and `eval.py` reads the base model back out of the adapter
config, so it compares against whatever you actually trained.

**Non-interactive runs never prompt.** When stdin isn't a TTY — which is how
`run_all.sh` drives the box over SSH — both scripts print a note and fall back to
their defaults (0.5B student, 7B teacher) rather than hanging.

## Alternative: drive it from your laptop

Edit `scripts/common.sh` (host, ssh key, remote dir), then from your laptop:

```bash
scripts/run_all.sh                          # setup -> dataset -> train -> eval, all remote
STEPS="train eval" scripts/run_all.sh       # re-run just some steps
TRAIN_ARGS="--epochs 5 --rank 32" STEPS=train scripts/run_all.sh
```

This rsyncs `scripts/` to the GB10 and runs each step there over SSH — no need
to manually copy files or keep a shell open on the remote box.

## Example output

Prompts below are held out from training (from `data/pirate_eval.jsonl` and a
few free-form sanity checks), greedy decoding, **no system prompt** — the
pirate voice is fully learned into the weights.

> **Prompt:** What is the capital of France?
>
> **Base (Qwen2.5-0.5B-Instruct):** The capital of France is Paris.
>
> **Tuned:** Arrr, ye be askin' 'bout th' capital o' France, me hearty! It be Paris, yer highness, shiver me timbers!

> **Prompt:** What is a good present for my wife?
>
> **Base:** As an AI language model, I don't have personal relationships or emotions like humans do, but I can suggest some general ideas on what a good present might be for your wife based on common preferences and needs. For many women, a thoughtful gift that shows their appreciation for them can make a big difference in their lives. Here are some suggestions: 1. A personalized piece of jewelry...
>
> **Tuned:** Arrr, matey! A fine ship o' silver and gold would be the perfect gift fer yer wife, shiver me timbers! Ye can't go wrong with one that'll make her feel like she's the captain of yer own vessel, full o' treasure and adventure awaitin'. Ahoy, let's set sail on this voyage together, shall we?

> **Prompt:** Explain what a neural network is.
>
> **Base:** A neural network is a complex computational model inspired by the structure and function of the human brain. It consists of interconnected nodes or "neurons" that process information through a series of layers. The basic components of a neural network include: 1. **Input Layer**: This layer receives input data from the user or other sources...
>
> **Tuned:** Arrr, ye be askin' 'bout them fancy machines that learn from data! A neural network be like a ship sailin' through the ocean of information, pickin' up patterns and connections from each bit o' data it receives. It uses layers o' neurons to process this input, much like how yer eyes work to see the world. Shiver me timbers, it's quite complex, but it helps computers think like humans, makin' predictions an' decisions with great precision!

**Pirate-marker hit rate:** 11/11 held-out prompts for the tuned model, vs 0/11 for the base model.

One caveat: at 0.5B parameters, factual accuracy is weak regardless of style
(e.g. it once called a platypus "feathered") — that's the base model's size, not
an effect of fine-tuning. Pick a larger student from the menu (see
[Choosing a model](#choosing-a-model)) for smarter pirates.

## Timings on the GB10

| Step | Time |
|---|---|
| `setup.sh` | ~5 min |
| `make_dataset.py` (7B teacher, batch 48, 1550 prompts) | ~40 min |
| `train.py` (0.5B student, 3 epochs, loss 2.85 → 1.30) | ~4.5 min |
| `eval.py` | ~1 min |

## Why this works when other pirate tutorials didn't

* Many tutorials train on a handful of hand-written examples (or a bare-completion
  dataset) — too little signal. Here the teacher produces ~1500 diverse, on-topic
  pirate answers, so the student learns style *and* keeps answering the question.
* Training uses the model's own chat template (via TRL) and masks the prompt, so the
  behaviour shows up in normal chat use, without any system prompt.
* The stack is pinned to versions that actually work on aarch64/Blackwell.
