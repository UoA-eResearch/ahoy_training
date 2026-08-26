#!/usr/bin/env python3
"""Base vs fine-tuned model on held-out ADE sentences: JSON validity + pair F1."""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=None, help="default: whatever train.py fine-tuned")
ap.add_argument("--tuned", default="out/ade-lora-merged")
ap.add_argument("--data", default="data/ade_eval.jsonl")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--show", type=int, default=5, help="example transcripts to print")
a = ap.parse_args()

# Above 7B train.py skips the merged copy, so fall back to the adapter.
adapter = None
if not os.path.isdir(a.tuned):
    fallback = a.tuned[: -len("-merged")] if a.tuned.endswith("-merged") else a.tuned
    if os.path.isfile(os.path.join(fallback, "adapter_config.json")):
        a.tuned = fallback
if os.path.isfile(os.path.join(a.tuned, "adapter_config.json")):
    adapter = a.tuned

if a.base is None:
    cfg = os.path.join(adapter or a.tuned.rstrip("/").replace("-merged", ""), "adapter_config.json")
    try:
        a.base = json.load(open(cfg))["base_model_name_or_path"]
    except (OSError, ValueError, KeyError):
        a.base = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"base: {a.base}\ntuned: {a.tuned}" + (" (adapter)" if adapter else " (merged)"))

# the 72B loads 4-bit NF4 (~40 GB), matching how train.py trained it
quant_kw = {}
if models.lookup(a.base)["tier"] == "xl":
    import torch
    from transformers import BitsAndBytesConfig
    quant_kw = {"quantization_config": BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)}

rows = [json.loads(l)["messages"] for l in open(a.data)][: a.n]
prompts = [m[0]["content"] for m in rows]
gold = [json.loads(m[1]["content"])["adverse_events"] for m in rows]

def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def pair_set(events):
    return {(norm(e.get("drug", "")), norm(e.get("effect", ""))) for e in events
            if isinstance(e, dict)}

def parse(reply):
    """Extract {"adverse_events": [...]} from a model reply; None if invalid."""
    for cand in (reply, *re.findall(r"\{.*\}", reply, re.S)):
        try:
            d = json.loads(cand)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("adverse_events"), list):
            return d["adverse_events"]
    return None

def run(path, as_adapter=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    if as_adapter:
        from peft import PeftModel
        m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda", **quant_kw)
        m = PeftModel.from_pretrained(m, path).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda", **quant_kw).eval()
    replies = []
    for i in range(0, len(prompts), a.batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + a.batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=150, do_sample=False, pad_token_id=tok.pad_token_id)
        replies += [tok.decode(x, skip_special_tokens=True).strip()
                    for x in g[:, enc["input_ids"].shape[1]:]]
        print(f"  {len(replies)}/{len(prompts)}", flush=True)
    del m; torch.cuda.empty_cache()
    return replies

def score(replies):
    valid = tp = fp = fn = 0
    for reply, g in zip(replies, gold):
        events = parse(reply)
        if events is not None:
            valid += 1
        pred, want = pair_set(events or []), pair_set(g)
        tp += len(pred & want); fp += len(pred - want); fn += len(want - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(valid=valid, precision=p, recall=r, f1=f1)

print("running base model...");  base_replies = run(a.base)
print("running tuned model..."); tuned_replies = run(a.tuned, as_adapter=bool(adapter))

for i in range(min(a.show, len(prompts))):
    sent = prompts[i].split("Sentence: ", 1)[-1]
    print("=" * 100)
    print("SENTENCE:", sent)
    print("-- GOLD :", json.dumps(gold[i], ensure_ascii=False))
    print("-- BASE :", base_replies[i][:300])
    print("-- TUNED:", tuned_replies[i][:300])

print("=" * 100)
b, t = score(base_replies), score(tuned_replies)
n = len(prompts)
print(f"{'':8} {'valid JSON':>12} {'precision':>10} {'recall':>8} {'F1':>6}")
for name, s in (("base", b), ("tuned", t)):
    print(f"{name:8} {s['valid']:>8}/{n:<3} {s['precision']:>10.3f} {s['recall']:>8.3f} {s['f1']:>6.3f}")
