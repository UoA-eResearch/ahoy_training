#!/usr/bin/env python3
"""LoRA fine-tune of a genomic foundation model on a DNA classification task.

Uses InstaDeepAI/nucleotide-transformer-500m-human-ref (ungated, stock ESM
architecture -- no trust_remote_code) on tasks from the published Nucleotide
Transformer downstream benchmark (ungated; windows of the human *reference*
genome, so nothing sensitive). Run genomic_eval.py afterwards for the
before/after comparison.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
DATASET = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"

# task -> (menu note, estimated train time on a GB10)
TASKS = [
    ("promoter_all",     "is this window a gene promoter? (the 'on switch' for transcription)", "~8 min"),
    ("promoter_tata",    "TATA-box promoters -- smallest task, quickest run",                   "~2 min"),
    ("enhancers",        "distal regulatory elements that boost gene expression",               "~8 min"),
    ("splice_sites_all", "exon/intron splice junctions -- splicing mutations cause disease",    "~8 min"),
    ("H3K4me3",          "histone mark flagging active promoters (epigenetics)",                "~5 min"),
]
DEFAULT_TASK = "promoter_all"

def format_tasks():
    lines = [f"  {'#':>2}  {'task':<18} {'~train':>8}  what it is", "  " + "-" * 76]
    for i, (name, note, est) in enumerate(TASKS, 1):
        mark = "  <- default" if name == DEFAULT_TASK else ""
        lines.append(f"  {i:>2}  {name:<18} {est:>8}  {note}{mark}")
    return "\n".join(lines)

def choose_task():
    if not sys.stdin.isatty():
        print(f"[non-interactive: using default task {DEFAULT_TASK} -- pass --task or --list-tasks to choose]")
        return DEFAULT_TASK
    default_n = next(i for i, t in enumerate(TASKS, 1) if t[0] == DEFAULT_TASK)
    print("\nWhich DNA classification task?\n")
    print(format_tasks())
    while True:
        try:
            raw = input(f"\nchoice [{default_n}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\naborted")
        if not raw:
            return DEFAULT_TASK
        if raw.isdigit() and 1 <= int(raw) <= len(TASKS):
            return TASKS[int(raw) - 1][0]
        print(f"  pick 1-{len(TASKS)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="benchmark task; omit for the picker menu")
    ap.add_argument("--list-tasks", action="store_true", help="print the task menu and exit")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="default: out/genomic-<task>-lora")
    ap.add_argument("--n", type=int, default=None, help="subsample the train split for a quick run")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    if a.list_tasks:
        print(format_tasks())
        return
    task = a.task or choose_task()
    out = a.out or f"out/genomic-{task}-lora"
    print(f"task: {task} -> {out}")

    import torch                                   # imported late: slow, not needed for the menu
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    ds = load_dataset(DATASET, split="train").filter(lambda r: r["task"] == task)
    if a.n:
        ds = ds.shuffle(seed=0).select(range(min(a.n, len(ds))))
    num_labels = len(set(ds["label"]))
    print(f"task {task}: {len(ds)} train sequences, {num_labels} classes, "
          f"{len(ds[0]['sequence'])} bp each")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=num_labels, dtype=torch.bfloat16)
    lora = LoraConfig(task_type="SEQ_CLS", r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                      target_modules=["query", "key", "value"])  # ESM attention module names
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = ds.map(lambda r: tok(r["sequence"], truncation=True, max_length=256),
                batched=True, remove_columns=["sequence", "name", "task"])

    args = TrainingArguments(
        output_dir=out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()

    model.save_pretrained(out)          # LoRA adapter + the trained classifier head
    tok.save_pretrained(out)
    with open(os.path.join(out, "task.json"), "w") as f:
        json.dump({"task": task, "num_labels": num_labels, "base_model": a.model}, f)
    print("saved", out, "-- now run: python scripts/genomic_eval.py")

if __name__ == "__main__":
    main()
