#!/usr/bin/env python3
"""Loading and talking to a chat model -- shared by baseline.py and chat.py.

Handles three kinds of path transparently:
  * a HuggingFace repo id      ("Qwen/Qwen2.5-0.5B-Instruct")
  * a merged model directory   ("out/pirate-lora-merged")
  * a LoRA adapter directory   ("out/pirate-lora" -- records its base model
                                in adapter_config.json, which we read back)

A model whose registry tier is "xl" (the 72B) does not fit here in bf16, so
load() quantizes the frozen base to 4-bit NF4 (~40 GB) -- exactly how train.py
trains it, which keeps before/after comparisons like for like.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

# anything the user can type to end an interactive session
STOP = {"", "q", "done", "quit", "exit", "/done", "/quit", "/exit", "bye"}


def resolve(path):
    """Return (model_to_load, adapter_dir_or_None).

    Above 7B train.py writes only the adapter (a merged copy would be tens of
    GB), so a missing "-merged" directory falls back to the adapter beside it.
    """
    if not os.path.isdir(path) and path.endswith("-merged"):
        stem = path[: -len("-merged")]
        if os.path.isdir(stem):
            path = stem
    cfg = os.path.join(path, "adapter_config.json")
    if os.path.isfile(cfg):
        return json.load(open(cfg))["base_model_name_or_path"], path
    return path, None


def load(path):
    """Load a chat model onto the GPU. Returns (model, tokenizer)."""
    base, adapter = resolve(path)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    kw = {}
    if models.lookup(base)["tier"] == "xl":
        # ~150 GB in bf16 -- load it 4-bit instead, same as training does
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    tok = AutoTokenizer.from_pretrained(adapter or base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda", **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


def ask(model, tok, question, history=None, max_new_tokens=200, greedy=False):
    """One user turn in, one assistant reply out."""
    msgs = list(history or []) + [{"role": "user", "content": question}]
    import torch

    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    sampling = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, **sampling)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def interactive(model, tok, label="model", record=None, keep_history=True):
    """Chat loop. Ends on a blank line, 'done', ctrl-D or ctrl-C.

    `record` (a list) collects {"question", "answer"} for each turn, so the
    baseline step can save what you asked and replay it after training.
    `keep_history=False` answers every question from a clean slate, which is
    what the baseline uses so the before/after comparison is like for like.
    """
    hist = []
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in STOP:
            break
        answer = ask(model, tok, q, hist if keep_history else None)
        print(f"{label}> {answer}")
        if keep_history:
            hist += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
        if record is not None:
            record.append({"question": q, "answer": answer})
    return record
