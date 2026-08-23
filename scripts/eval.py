#!/usr/bin/env python3
"""Side-by-side: base model vs fine-tuned (merged) model on held-out prompts, no system prompt."""
import argparse, json, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--tuned", default="out/pirate-lora-merged")
ap.add_argument("--data", default="data/pirate_eval.jsonl")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

prompts = [json.loads(l)["messages"][0]["content"] for l in open(a.data)][: a.n]
prompts += ["What is the capital of France?", "Explain what a neural network is.", "Give me a recipe for pancakes."]
pirate = re.compile(r"\b(arr+|ahoy|matey|ye|yer|aye|hearties)\b", re.I)

def run(path):
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(x, skip_special_tokens=True).strip() for x in g[:, enc["input_ids"].shape[1]:]]

base, tuned = run(a.base), run(a.tuned)
for p, b, t in zip(prompts, base, tuned):
    print("=" * 100); print("PROMPT:", p); print("-- BASE :", b); print("-- TUNED:", t)
print("=" * 100)
print(f"pirate-marker hit rate  base: {sum(bool(pirate.search(x)) for x in base)}/{len(prompts)}   "
      f"tuned: {sum(bool(pirate.search(x)) for x in tuned)}/{len(prompts)}")
