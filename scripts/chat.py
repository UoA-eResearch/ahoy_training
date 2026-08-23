#!/usr/bin/env python3
"""Interactive chat with the fine-tuned model."""
import json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
path = sys.argv[1] if len(sys.argv) > 1 else "out/pirate-lora-merged"
# above 7B train.py writes only the adapter, so fall back to it and load via PEFT
if not os.path.isdir(path) and path.endswith("-merged") and os.path.isdir(path[: -len("-merged")]):
    path = path[: -len("-merged")]
adapter_cfg = os.path.join(path, "adapter_config.json")
tok = AutoTokenizer.from_pretrained(path)
if os.path.isfile(adapter_cfg):
    from peft import PeftModel
    base = json.load(open(adapter_cfg))["base_model_name_or_path"]
    print(f"loading adapter {path} on {base}")
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    m = PeftModel.from_pretrained(m, path).eval()
else:
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
hist = []
while True:
    try: q = input("\nyou> ").strip()
    except EOFError: break
    if not q: continue
    hist.append({"role": "user", "content": q})
    ids = tok.apply_chat_template(hist, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = m.generate(**ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    ans = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    hist.append({"role": "assistant", "content": ans}); print("pirate>", ans)
