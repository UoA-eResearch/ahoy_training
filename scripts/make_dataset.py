#!/usr/bin/env python3
"""Build a pirate-dialect SFT dataset.

Takes real user prompts from databricks/databricks-dolly-15k (ungated) and has a
larger ungated instruct model (default Qwen/Qwen2.5-7B-Instruct) answer them *in
pirate dialect*. Output: data/pirate_train.jsonl + data/pirate_eval.jsonl with
{"messages":[{"role":"user",...},{"role":"assistant",...}]} rows (no system
prompt, so the student learns to be a pirate unconditionally).
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models

SYSTEM = (
    "You are Captain Redbeard, a salty old pirate. Answer the user's question helpfully "
    "and correctly, but speak ENTIRELY in exaggerated pirate dialect: use 'Arr', 'Ahoy', "
    "'matey', 'ye', 'yer', 'aye', 'me hearties', 'shiver me timbers', nautical metaphors, "
    "and pirate slang in every sentence. Keep answers concise (2-5 sentences). Never break character."
)

def generate_local(model_id, prompts, batch, max_new_tokens):
    """Answer every prompt with the teacher loaded on this box's GPU."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    outputs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
        for p, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            outputs.append((p, tok.decode(g, skip_special_tokens=True).strip()))
        print(f"{len(outputs)}/{len(prompts)}  e.g. {outputs[-1][1][:80]!r}", flush=True)
    return outputs


def generate_via_pair(model_id, prompts, concurrency, max_new_tokens):
    """A 72B teacher does not fit on one GB10: serve it across the cabled pair
    with vLLM (tensor parallel 2) and generate through its API instead. Same
    prompts, same sampling -- only where the forward pass runs changes."""
    from concurrent.futures import ThreadPoolExecutor
    import pair

    url, name = pair.serve(model_id)
    def one(p):
        return pair.chat(url, name, [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": p}],
                         max_tokens=max_new_tokens)
    outputs = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for p, reply in zip(prompts, ex.map(one, prompts)):
            outputs.append((p, reply))
            if len(outputs) % 25 == 0 or len(outputs) == len(prompts):
                print(f"{len(outputs)}/{len(prompts)}  e.g. {reply[:80]!r}", flush=True)
    if url == pair.OUR_URL:
        pair.stop_serving()   # free both GPUs for the training step
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=None, help="HF repo id; omit for the picker menu")
    ap.add_argument("--list-models", action="store_true", help="print the model menu and exit")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the large-model confirmation")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--eval_n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=None, help="default scales with teacher size")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.list_models:
        print(models.format_menu("gen_est", "gen", models.DEFAULT_TEACHER))
        return

    entry = (models.lookup(a.teacher) if a.teacher else
             models.choose_model("teacher", models.DEFAULT_TEACHER, "gen_est", "gen"))
    models.confirm(entry, "gen_est", assume_yes=a.yes)
    batch = a.batch if a.batch is not None else entry["gen_batch"]
    print(f"teacher {entry['id']} ({entry['params']}) -- batch {batch}, est {entry['gen_est']}")

    from datasets import load_dataset              # imported late: slow, not needed for the menu

    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    # keep prompts that need no context and are short-ish, in the open/general categories
    rows = [r for r in ds if not r["context"] and r["category"] in
            ("open_qa", "general_qa", "brainstorming", "creative_writing", "classification")
            and 10 < len(r["instruction"]) < 200]
    random.shuffle(rows)
    prompts = [r["instruction"].strip() for r in rows[: a.n + a.eval_n]]
    print(f"{len(prompts)} prompts selected")

    if entry["tier"] == "xl":
        outputs = generate_via_pair(entry["id"], prompts, batch, a.max_new_tokens)
    else:
        outputs = generate_local(entry["id"], prompts, batch, a.max_new_tokens)

    pirate = re.compile(r"\b(arr+|ahoy|matey|ye|yer|aye|hearties)\b", re.I)
    keep = [(p, r) for p, r in outputs if pirate.search(r) and len(r) > 20]
    print(f"kept {len(keep)}/{len(outputs)} (filtered non-pirate/short)")
    ev, tr = keep[: a.eval_n], keep[a.eval_n:]
    for name, part in (("pirate_train.jsonl", tr), ("pirate_eval.jsonl", ev)):
        with open(os.path.join(a.out, name), "w") as f:
            for p, r in part:
                f.write(json.dumps({"messages": [{"role": "user", "content": p},
                                                 {"role": "assistant", "content": r}]}) + "\n")
        print("wrote", name, len(part))

if __name__ == "__main__":
    main()
