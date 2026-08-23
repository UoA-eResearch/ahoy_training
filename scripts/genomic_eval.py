#!/usr/bin/env python3
"""Before/after comparison for the genomic classifier.

"Before" is the base Nucleotide Transformer with an untrained classification
head (what you'd have without fine-tuning: coin-flip). "After" loads the LoRA
adapter + trained head. Both are scored on the held-out test split with
accuracy and Matthews correlation (MCC).
"""
import argparse, glob, json, os
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASET = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=None, help="default: newest out/genomic-*-lora")
ap.add_argument("--n", type=int, default=1000, help="test sequences to score")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--show", type=int, default=6, help="example sequences to print")
a = ap.parse_args()

if a.adapter is None:
    dirs = sorted(glob.glob("out/genomic-*-lora"), key=os.path.getmtime)
    if not dirs:
        raise SystemExit("no out/genomic-*-lora found -- run scripts/genomic_train.py first")
    a.adapter = dirs[-1]
meta = json.load(open(os.path.join(a.adapter, "task.json")))
task, num_labels, base_id = meta["task"], meta["num_labels"], meta["base_model"]
print(f"adapter: {a.adapter}  (task {task}, base {base_id})")

ds = load_dataset(DATASET, split="test").filter(lambda r: r["task"] == task)
ds = ds.shuffle(seed=0).select(range(min(a.n, len(ds))))
seqs, labels = ds["sequence"], ds["label"]
print(f"{len(seqs)} held-out test sequences")

tok = AutoTokenizer.from_pretrained(a.adapter)

def predict(model):
    model = model.eval().to("cuda")
    preds = []
    for i in range(0, len(seqs), a.batch):
        enc = tok(seqs[i:i + a.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            preds += model(**enc).logits.argmax(-1).tolist()
    model.to("cpu"); torch.cuda.empty_cache()
    return preds

def mcc(y_true, y_pred):
    """Matthews correlation, multi-class (Gorodkin), no sklearn needed."""
    k = max(max(y_true), max(y_pred)) + 1
    C = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        C[t][p] += 1
    s = len(y_true)
    c = sum(C[i][i] for i in range(k))
    t = [sum(C[i]) for i in range(k)]                    # true counts
    p = [sum(C[i][j] for i in range(k)) for j in range(k)]  # predicted counts
    cov = c * s - sum(pi * ti for pi, ti in zip(p, t))
    den = ((s * s - sum(pi * pi for pi in p)) * (s * s - sum(ti * ti for ti in t))) ** 0.5
    return cov / den if den else 0.0

torch.manual_seed(0)   # reproducible untrained head
print("scoring base model + untrained head (before)...")
before = predict(AutoModelForSequenceClassification.from_pretrained(
    base_id, num_labels=num_labels, dtype=torch.bfloat16))

print("scoring fine-tuned model (after)...")
tuned = AutoModelForSequenceClassification.from_pretrained(
    base_id, num_labels=num_labels, dtype=torch.bfloat16)
after = predict(PeftModel.from_pretrained(tuned, a.adapter))

for i in range(min(a.show, len(seqs))):
    print("=" * 100)
    print(f"SEQ {seqs[i][:60]}...  ({len(seqs[i])} bp)")
    print(f"  true={labels[i]}  before={before[i]}  after={after[i]}")

print("=" * 100)
acc = lambda pred: sum(p == t for p, t in zip(pred, labels)) / len(labels)
print(f"{'':8} {'accuracy':>9} {'MCC':>7}")
print(f"{'before':8} {acc(before):>9.3f} {mcc(labels, before):>7.3f}")
print(f"{'after':8} {acc(after):>9.3f} {mcc(labels, after):>7.3f}")
