# ahoy_training — hands-on fine-tuning on a DGX GB10

Three self-contained fine-tuning demos you can run end to end on an NVIDIA
DGX GB10 (aarch64 + Blackwell, no root access). Each one takes a model that
already works, shows you what it can't do yet, trains it for a few minutes,
and shows you what changed.

Everything uses **LoRA**: the original model's weights are frozen and a small
adapter is trained beside them. That's why these finish in minutes and produce
a few hundred megabytes instead of running for hours and producing tens of
gigabytes. Every model and dataset is ungated on HuggingFace — no token, no
license click-through.

| # | Track | What it teaches a model to do | ~Time |
|---|---|---|---|
| 1 | **Pirate chat model** | Answer everything in pirate dialect — a new *voice*, learned into the weights | ~50 min |
| 2 | **Medical ADE extractor** | Read a case-report sentence and return the drug/side-effect pairs as JSON | ~11 min |
| 3 | **Genomic classifier** | Look at a window of DNA and say whether it contains a gene promoter | ~9 min |

Track 1 is the friendliest introduction — the result is something you can talk
to, and the change is audible in one sentence. Tracks 2 and 3 apply the exact
same technique to tasks with real scientific value. The method doesn't change
between them; only the dataset does.

## How every track works

All three follow the same four beats, and the launcher explains each one as it
runs:

1. **See what you're starting with.** The untrained model attempts the task in
   front of you. This is the problem the rest of the run is fixing, and it's
   what the final numbers get compared against. On track 1 you chat with the
   untrained model yourself and your questions are saved for later.
2. **Get the training data.** Tracks 2 and 3 use published, annotated datasets
   directly. Track 1 has to *generate* its data first, because no dataset of
   pirate answers exists.
3. **Train the adapter.** A few minutes of LoRA on the GPU. Under 1% of the
   model's weights move.
4. **Measure and read the difference.** Held-out examples the model never
   trained on, scored before and after. On track 1 you also get your own
   questions replayed side by side, and then the prompt back to keep chatting.

## Setup (once)

Everything runs **on the GB10** over SSH. Swap in your own host and user.

```bash
ssh -i <user>@<name or ip of GB10>
```

```bash
git clone https://github.com/UoA-eResearch/ahoy_training.git ~/ahoy_training
cd ~/ahoy_training
bash scripts/setup.sh
```

`setup.sh` builds a `uv` virtual environment with CUDA PyTorch (the `cu130`
aarch64 wheels, which support the GB10's `sm_121` GPU) plus the HuggingFace
fine-tuning stack. No root needed, takes about five minutes, and it ends by
printing your GPU name — confirm it says `NVIDIA GB10`.

## Running it

```bash
source .venv/bin/activate && python scripts/tune.py
```

Pick a track from the menu. It asks a couple of setup questions up front
(which model to tune, which DNA task), then walks the steps in order — showing
what each step does and why it matters before it runs, and pausing so you can
run it, skip it, or stop. Anything already built from an earlier run is offered
as a skip, so it's safe to re-run and pick up where you left off.

## Track 1 — Pirate chat model

**What's being trained.** The model's *behaviour* — its voice — not its facts
or capabilities. There's no real-world value in a pirate specifically. The
point is that this is the same mechanism behind everything that reshapes how a
model responds: house style, a fixed output format, a domain persona, refusal
behaviour. It's the clearest possible demonstration because you can hear the
change immediately.

The interesting constraint is that the training examples contain **no system
prompt**. The model is never told to be a pirate — it only ever sees questions
paired with piratey answers, so the only place the behaviour can end up is in
the weights. Nothing is added to the prompt at inference time.

**Where the data comes from.** ~1500 real user questions from
`databricks/databricks-dolly-15k`, answered in dialect by a larger "teacher"
model (default Qwen2.5-7B) under a pirate system prompt, with replies that
don't actually sound piratey filtered out. Generating training data from a
bigger model is called **self-distillation**, and it's the standard answer to
"I know what I want the model to do, but I have no dataset for it."

**What you'll see.** Held-out prompts, greedy decoding, no system prompt:

> **Prompt:** What is the capital of France?
>
> **Before:** The capital of France is Paris.
>
> **After:** Arrr, ye be askin' 'bout th' capital o' France, me hearty! It be Paris, yer highness, shiver me timbers!

> **Prompt:** Explain what a neural network is.
>
> **Before:** A neural network is a complex computational model inspired by the structure and function of the human brain. It consists of interconnected nodes or "neurons" that process information through a series of layers...
>
> **After:** Arrr, ye be askin' 'bout them fancy machines that learn from data! A neural network be like a ship sailin' through the ocean of information, pickin' up patterns and connections from each bit o' data it receives. It uses layers o' neurons to process this input, much like how yer eyes work to see the world...

**Pirate-marker hit rate:** 11/11 held-out prompts after training, 0/11 before.

At the end you get the prompt back and can keep chatting — try questions you
didn't ask at the start, since holding the voice on unseen input is what
separates a generalised behaviour from a memorised training set.

One caveat: at 0.5B parameters, factual accuracy is weak regardless of style
(it once called a platypus "feathered"). That's the base model's size, not an
effect of fine-tuning — pick a larger student for smarter pirates.

## Track 2 — Medical ADE extractor

**What's being trained.** *Structured information extraction* — not
conversation, and not classification of clinical notes. Given a sentence, the
model has to decide whether it describes an adverse drug event at all, and if
so emit the exact `{drug, effect}` pairs as JSON: spotting the drug name, the
effect it caused, and the link between them, while staying silent on sentences
that describe neither so it doesn't invent reactions that were never reported.

This is the shape of most real LLM work — read unstructured text, emit a
structured record something downstream can rely on.

**Where the data comes from.** `ade-benchmark-corpus/ade_corpus_v2`: ~5.5k
sentences from *published* PubMed case reports with gold drug/effect
annotations, a standard NLP benchmark. ~1500 sentences reporting no adverse
event are mixed in as negatives — without them the model learns "always find
something." **No patient records are involved anywhere**: these are sentences
from papers that have been published and de-identified for years.

**Why it matters.** Mining case-report literature (or, in a real deployment,
EHR narrative text) for adverse drug events is a genuine pharmacovigilance
task, normally done by manual clinical review at far lower throughput. The same
recipe generalises to any "read text, fill in this schema" job — diagnoses,
medications and dosages, trial eligibility criteria.

**Measured result** (0.5B student, 100 held-out sentences):

|  | valid JSON | precision | recall | F1 |
|---|---|---|---|---|
| before | 100/100 | 0.000 | 0.000 | **0.000** |
| after | 99/100 | 0.763 | 0.685 | **0.722** |

The failure mode before training is the instructive part. The base model
follows the JSON format perfectly and extracts *nothing* — it answers
`{"adverse_events": []}` for almost every sentence. It has learned the *shape*
of the answer from general instruction tuning but has no learned behaviour for
finding drug and effect spans in text, so precision and recall are both 0.000:
not because it gets pairs wrong, but because it never proposes any.

Ten minutes of LoRA changes that. Recall of 0.685 means it now finds roughly
two-thirds of the true pairs in the eval set; precision of 0.763 means most of
what it does extract is correct rather than invented.

> **Sentence:** This report presents a potential case of risperidone-induced tardive dyskinesia.
>
> **Before:** `{"adverse_events": []}`
>
> **After:** `{"adverse_events": [{"drug": "risperidone", "effect": "tardive dyskinesia"}]}`

That's the "unreliable chatbot → dependable pipeline component" transformation
that makes fine-tuning worth doing.

## Track 3 — Genomic classifier

**What's being trained.** Not a chat model at all. The base model,
`InstaDeepAI/nucleotide-transformer-500m-human-ref`, was pretrained purely to
predict masked nucleotides in raw DNA — the genomic equivalent of a language
model's next-token objective. It has no built-in notion of biological function.
Here a classification head is LoRA-tuned on top of its embeddings to answer one
specific question about a fixed-length window of DNA, turning a general
sequence model into a task-specific genomic annotator.

A menu picks the question:

- `promoter_all` (default) — is this window a gene promoter, the "on switch" for transcription?
- `promoter_tata` — TATA-box promoters; smallest task, quickest run (~3 min)
- `enhancers` — distal regulatory elements that boost gene expression
- `splice_sites_all` — exon/intron junctions; splicing mutations cause disease
- `H3K4me3` — histone mark flagging active promoters (epigenetics)

**Where the data comes from.**
`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`, the published
Nucleotide Transformer benchmark — labelled windows of the human **reference**
genome, the standard scientific baseline sequence. No individual's genome, no
subject data, nothing sensitive.

**Why it matters.** Identifying regulatory elements in DNA normally means a
wet-lab assay (ChIP-seq, ATAC-seq) or a hand-built statistical model, both
expensive next to a sequence-classification pass. This is how foundation models
actually get used in genomics: pretrain once on raw sequence, then adapt
cheaply per question. Swap the organism, the task, or the promoter definition
and the recipe is unchanged.

**Measured result** (`promoter_all`, defaults — 30k train sequences, 2 epochs,
~0.9% of weights trained, 1000 held-out test sequences):

|  | accuracy | MCC |
|---|---|---|
| before | 0.508 | 0.000 |
| after | **0.887** | **0.774** |

Before fine-tuning, the classification head sitting on the model's embeddings
is *randomly initialised* — it has never been shown what a promoter looks like,
so it scores 0.508, indistinguishable from a coin flip. MCC (Matthews
correlation, the standard metric for this benchmark) runs from -1 to 1 and sits
at 0 for a model that is guessing, which is why it's the honest metric here:
unlike accuracy, it can't be flattered by class imbalance. An MCC of 0.000
confirms there's no signal at all, rather than a model that happened to land
near 50%.

After training, accuracy reaches 0.887 and MCC 0.774 — a strong correlation
between predicted and true labels. The model has learned to recognise the
actual sequence features that distinguish real promoters from ordinary DNA:
TATA boxes, GC content, transcription-factor binding motifs and their spacing.
A genuine coin flip becomes a usable promoter detector in about seven minutes.

> Note: the Nucleotide Transformer weights are CC-BY-NC-SA (non-commercial) —
> fine for an instructive demo, but check the licence before building a product
> on them.

## Choosing a model

Tracks 1 and 2 ask which model to fine-tune (and track 1 also asks which
teacher generates its data). Every model listed downloads anonymously — no
`HF_TOKEN`, no click-through. That's why the ladder is Qwen throughout: the
Llama, Gemma and Mistral instruct repos are all gated.

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
estimates. Options 5 and 6 ask for confirmation before downloading, and option
7 is refused outright — 72B needs ~150 GB in bf16 against the GB10's ~110 GB
usable, so it would need 4-bit QLoRA on one box or bf16 LoRA split across two
GB10s with pipeline parallelism, neither of which is wired up here.

Batch size and gradient checkpointing scale automatically with the model you
pick (the 152k-token vocabulary makes the logits tensor the binding constraint
at the large end), and the evaluation reads the base model back out of the
adapter config, so it always compares against whatever you actually trained.

## Timings on the GB10

| Step | Time |
|---|---|
| Environment setup | ~5 min |
| Track 1 · generate dataset (7B teacher, 1550 prompts) | ~40 min |
| Track 1 · train (0.5B, 3 epochs, loss 2.85 → 1.30) | ~4.5 min |
| Track 1 · evaluate | ~1 min |
| Track 2 · build dataset | ~15 s |
| Track 2 · train (0.5B, 3 epochs) | ~10 min |
| Track 2 · evaluate | ~20 s |
| Track 3 · train (promoter_all, 2 epochs) | ~8 min |
| Track 3 · evaluate (1000 test sequences) | ~20 s |

The "see what you're starting with" step adds a minute or two per track, plus
however long you spend chatting on track 1.
