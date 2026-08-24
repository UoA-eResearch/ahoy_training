# ahoy_training — learn fine-tuning on a DGX GB10, from pirates to genomes

Reproducible LoRA fine-tuning demos for an NVIDIA DGX GB10 (aarch64 + Blackwell,
no root access), in three tracks of increasing seriousness. Every model and
dataset is ungated on HuggingFace — no HF token, no license click-through:

| # | Track | What you get | ~Time |
|---|---|---|---|
| 1 | **Pirate chat model** | Whimsical intro: Qwen answers *everything* in pirate dialect, no system prompt needed | ~50 min |
| 2 | **Medical ADE extractor** | Practical: the same Qwen learns to pull structured `{drug, effect}` JSON from case-report sentences — F1 0.00 → 0.72 | ~11 min |
| 3 | **Genomic classifier** | Scientific: a DNA foundation model learns to recognize gene promoters — accuracy 0.51 → 0.89 | ~9 min |

Start with the pirate — it teaches the whole dataset → train → eval loop with a
result you can chat to. Then the other two show why the same technique matters
for real scientific and medical work.

## Setup (once)

Everything runs **on the GB10** over SSH. Swap in your own host/user.

**Step 1.** SSH in:

```bash
ssh -i ~/.ssh/id_rsa smat924@lais01.cer.auckland.ac.nz
```

**Step 2.** Get the code:

```bash
git clone https://github.com/UoA-eResearch/ahoy_training.git ~/ahoy_training
cd ~/ahoy_training
```

**Step 3.** Create the environment — a `uv` venv with CUDA PyTorch (the `cu130`
aarch64 wheels, which support the GB10's `sm_121` GPU) plus the HuggingFace
fine-tuning stack. No root needed. Takes ~5 minutes:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

It ends by printing your GPU name — confirm it says `NVIDIA GB10`.

(Prefer to drive everything from your laptop instead? See
[Alternative: drive it from your laptop](#alternative-drive-it-from-your-laptop).)

## The guided way: one menu

```bash
python scripts/tune.py
```

Pick a track from the numbered menu and it runs that track's steps in order,
offering to skip any step whose output already exists. Each track's own menus
(student model, DNA task) still appear along the way. The sections below are the
same steps run by hand.

## Option 1: Pirate chat model (start here)

Fine-tunes **Qwen/Qwen2.5-0.5B-Instruct** so it answers everything in pirate
dialect — learned into the weights, no system prompt required.

**Step 1 — build the dataset** (~40 min). This is "self-distillation": ~1500
real user prompts from `databricks/databricks-dolly-15k` (ungated) are answered
by a larger ungated teacher model under a pirate system prompt; answers that
don't sound piratey are filtered out.

```bash
python scripts/make_dataset.py
```

A numbered menu asks which teacher to use (default: Qwen2.5-7B). The output is
`data/pirate_train.jsonl` + `data/pirate_eval.jsonl` — plain
`{"messages": [...]}` rows with **no system message**, so the student has to
internalize the pirate voice rather than being told to use it.

Useful flags: `--teacher <hf-repo-id>` to skip the menu, `--n 500` for a
quick/cheap run, `--batch 16` if memory constrained, `--list-models`.

**Step 2 — fine-tune** (~5 min). LoRA (rank 16, all linear layers) SFT with
TRL, loss computed only on the assistant tokens:

```bash
python scripts/train.py
```

A menu asks which student model to tune (default: 0.5B — see
[Choosing a model](#choosing-a-model)). Saves `out/pirate-lora` (adapter) and
`out/pirate-lora-merged` (standalone model; skipped above 7B where it's tens of
GB — `--save-merged` forces it).

Useful flags: `--epochs 5`, `--rank 32`, `--lr 1e-4`, `--model <hf-repo-id>`,
`-y` to auto-accept the large-model warning.

**Step 3 — evaluate.** Base vs tuned side by side on held-out prompts, plus a
crude "does the reply contain pirate words" hit rate:

```bash
python scripts/eval.py
```

**Step 4 — chat with it:**

```bash
python scripts/chat.py                       # defaults to out/pirate-lora-merged
python scripts/chat.py out/pirate-lora       # or point at a model/adapter dir
```

See [Example output](#example-output-pirate-track) below for what to expect.

## Option 2: Medical ADE extractor

Turns the same pipeline to a real task: extracting **adverse drug events** from
biomedical text. The data (`ade-benchmark-corpus/ade_corpus_v2`, ungated) is
sentences from *published* PubMed case reports with gold drug/effect
annotations — a standard NLP benchmark, no patient records.

**Step 1 — build the dataset** (seconds, no GPU). ~5.5k train / 200 eval
sentences; negative sentences are mixed in so the model learns to output an
empty list instead of hallucinating:

```bash
python scripts/ade_make_dataset.py
```

**Step 2 — fine-tune** (~10 min). Same script and same student-model menu as
the pirate track, pointed at the ADE data:

```bash
python scripts/train.py --data data/ade_train.jsonl --out out/ade-lora
```

**Step 3 — evaluate.** Scores JSON validity plus precision/recall/F1 on
extracted `(drug, effect)` pairs, base vs tuned:

```bash
python scripts/ade_eval.py
```

Measured result (0.5B student, defaults):

|  | valid JSON | precision | recall | F1 |
|---|---|---|---|---|
| base | 100/100 | 0.000 | 0.000 | **0.000** |
| tuned | 99/100 | 0.763 | 0.685 | **0.722** |

The failure mode is instructive: the base 0.5B dutifully follows the JSON
format but extracts *nothing* — it answers `{"adverse_events": []}` for nearly
every sentence. Ten minutes of LoRA turns that into real extractions:

> **Sentence:** This report presents a potential case of risperidone-induced tardive dyskinesia.
>
> **Base:** `{"adverse_events": []}`
>
> **Tuned:** `{"adverse_events": [{"drug": "risperidone", "effect": "tardive dyskinesia"}]}`

That's the "unreliable chatbot → dependable component" transformation that
makes fine-tuning useful in practice.

## Option 3: Genomic classifier

Fine-tunes **InstaDeepAI/nucleotide-transformer-500m-human-ref** — a 500M-param
DNA language model, ungated and stock ESM architecture (no `trust_remote_code`,
so it runs on the pinned aarch64 stack) — on tasks from the published
Nucleotide Transformer benchmark
(`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`, ungated windows
of the human *reference* genome — nothing sensitive).

**Step 1 — fine-tune** (~8 min):

```bash
python scripts/genomic_train.py
```

A menu asks which DNA task to learn:

- `promoter_all` (default) — is this window a gene promoter? (the "on switch" for transcription)
- `promoter_tata` — TATA-box promoters; smallest task, quickest run (~3 min)
- `enhancers` — distal regulatory elements that boost gene expression
- `splice_sites_all` — exon/intron junctions; splicing mutations cause disease
- `H3K4me3` — histone mark flagging active promoters (epigenetics)

Useful flags: `--task promoter_all` to skip the menu, `--n 2000` to subsample
for a quick first run, `--list-tasks`.

**Step 2 — evaluate:**

```bash
python scripts/genomic_eval.py
```

This is the point of the demo: **before** = the base model with an untrained
classification head (what you'd have without fine-tuning); **after** = the
LoRA-tuned model. Both are scored on held-out test sequences with accuracy and
Matthews correlation (MCC, the standard metric in this benchmark). Measured
result (`promoter_all`, defaults — 30k sequences, 2 epochs, ~0.9 % of weights
trained):

|  | accuracy | MCC |
|---|---|---|
| before | 0.508 | 0.000 |
| after | **0.887** | **0.774** |

A genuine coin flip becomes a usable promoter detector in ~7 minutes of
training — the same LoRA technique as the pirate, applied to a genome instead
of a chat log.

> Note: the Nucleotide Transformer weights are CC-BY-NC-SA (non-commercial) —
> fine for this instructive demo, but check the license before building a
> product on them.

## Choosing a model

Both `make_dataset.py` (the teacher) and `train.py` (the student — pirate and
ADE tracks alike) show a numbered menu when run interactively. Every model
listed downloads from HuggingFace anonymously — no `HF_TOKEN`, no license
click-through. That's why the ladder is Qwen throughout: the Llama, Gemma and
Mistral instruct repos are all gated.

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
`run_all.sh` drives the box over SSH — every script prints a note and falls back
to its defaults rather than hanging.

## Alternative: drive it from your laptop

Edit `scripts/common.sh` (host, ssh key, remote dir), then from your laptop:

```bash
scripts/run_all.sh                          # setup -> dataset -> train -> eval, all remote
STEPS="train eval" scripts/run_all.sh       # re-run just some steps
TRAIN_ARGS="--epochs 5 --rank 32" STEPS=train scripts/run_all.sh

# tracks 2 and 3 have their own steps:
STEPS="ade-dataset ade-train ade-eval" scripts/run_all.sh
STEPS="genomic-train genomic-eval" scripts/run_all.sh
GENOMIC_TRAIN_ARGS="--task splice_sites_all" STEPS="genomic-train genomic-eval" scripts/run_all.sh
```

This rsyncs `scripts/` to the GB10 and runs each step there over SSH — no need
to manually copy files or keep a shell open on the remote box.

## Example output (pirate track)

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
| `ade_make_dataset.py` | ~15 s |
| `train.py` on ADE data (0.5B, 3 epochs) | ~10 min |
| `ade_eval.py` | ~20 s |
| `genomic_train.py` (promoter_all, 2 epochs) | ~8 min |
| `genomic_eval.py` (1000 test sequences) | ~20 s |

## Why this works when other pirate tutorials didn't

* Many tutorials train on a handful of hand-written examples (or a bare-completion
  dataset) — too little signal. Here the teacher produces ~1500 diverse, on-topic
  pirate answers, so the student learns style *and* keeps answering the question.
* Training uses the model's own chat template (via TRL) and masks the prompt, so the
  behaviour shows up in normal chat use, without any system prompt.
* The stack is pinned to versions that actually work on aarch64/Blackwell.
