#!/usr/bin/env python3
"""Side-by-side: base model vs fine-tuned (merged) model on held-out prompts, no system prompt."""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=None, help="default: whatever train.py fine-tuned")
ap.add_argument("--tuned", default="out/pirate-lora-merged")
ap.add_argument("--data", default="data/pirate_eval.jsonl")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

# Above 7B train.py skips the merged copy (it's tens of GB), so fall back to the adapter.
adapter = None
if not os.path.isdir(a.tuned):
    fallback = a.tuned[: -len("-merged")] if a.tuned.endswith("-merged") else a.tuned
    if os.path.isfile(os.path.join(fallback, "adapter_config.json")):
        a.tuned = fallback
if os.path.isfile(os.path.join(a.tuned, "adapter_config.json")):
    adapter = a.tuned

if a.base is None:
    # train.py records the base in the adapter config, so the comparison follows
    # whichever model was picked from the menu rather than assuming the 0.5B
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

prompts = [json.loads(l)["messages"][0]["content"] for l in open(a.data)][: a.n]
prompts += ["What is the capital of France?", "Explain what a neural network is.", "Give me a recipe for pancakes."]
pirate = re.compile(r"\b(arr+|ahoy|matey|ye|yer|aye|hearties)\b", re.I)

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
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(x, skip_special_tokens=True).strip() for x in g[:, enc["input_ids"].shape[1]:]]

base, tuned = run(a.base), run(a.tuned, as_adapter=bool(adapter))
for p, b, t in zip(prompts, base, tuned):
    print("=" * 100); print("PROMPT:", p); print("-- BASE :", b); print("-- TUNED:", t)
print("=" * 100)
print(f"pirate-marker hit rate  base: {sum(bool(pirate.search(x)) for x in base)}/{len(prompts)}   "
      f"tuned: {sum(bool(pirate.search(x)) for x in tuned)}/{len(prompts)}")
