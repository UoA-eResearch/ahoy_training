#!/usr/bin/env python3
"""The "before" picture: what the model does with no fine-tuning at all.

Every track runs this first. Nothing is trained here -- it exists so that the
numbers and transcripts you see after training have something concrete to be
compared against. This is the problem; the rest of the track is the fix.

  pirate   -- chat with the untrained model yourself; your questions are saved
              and replayed against the fine-tuned model afterwards
  ade      -- watch the untrained model try (and fail) to extract drug/effect
              pairs from case-report sentences
  genomic  -- watch the untrained classifier head guess promoter/not-promoter
              at coin-flip accuracy
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatlib

SUGGESTED = [
    "What is the capital of France?",
    "Explain what a neural network is.",
    "Give me a recipe for pancakes.",
    "What is a good present for my wife?",
]
PIRATE_WORDS = re.compile(r"\b(arr+|ahoy|matey|ye|yer|aye|hearties)\b", re.I)


def rule(char="-"):
    print(char * 78)


# --------------------------------------------------------------------------- pirate
def pirate(a):
    print(f"\nloading {a.model} (no fine-tuning yet)...")
    model, tok = chatlib.load(a.model)

    print()
    rule("=")
    print("  Chat with the UNTRAINED model")
    rule("=")
    print("""
  This is a stock instruct model. It is helpful and fluent, and it sounds
  like every other assistant -- because nothing has taught it otherwise.

  Ask it whatever you like. Your questions are saved, and after training
  the same ones get asked again so you can see exactly what changed.

  Each question is answered from a clean slate (no conversation memory),
  which keeps the before/after comparison honest.

  Blank line, 'done' or ctrl-D when you have seen enough.
""")

    turns = []
    if sys.stdin.isatty():
        chatlib.interactive(model, tok, label="model", record=turns, keep_history=False)

    if not turns:
        why = "non-interactive run" if not sys.stdin.isatty() else "nothing asked"
        print(f"  [{why} -- using the standard question set instead]\n")
        for q in SUGGESTED:
            answer = chatlib.ask(model, tok, q, greedy=True)
            print(f"you>   {q}\nmodel> {answer}\n")
            turns.append({"question": q, "answer": answer})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"model": a.model, "turns": turns}, open(a.out, "w"), indent=2)

    hits = sum(bool(PIRATE_WORDS.search(t["answer"])) for t in turns)
    rule("=")
    print(f"  {len(turns)} question(s) saved to {a.out}")
    print(f"  pirate-marker hit rate: {hits}/{len(turns)}"
          "   <- the number we are trying to move")
    rule("=")


# --------------------------------------------------------------------------- ade
def parse_events(reply):
    """Pull {"adverse_events": [...]} out of a model reply; None if unparseable."""
    for cand in (reply, *re.findall(r"\{.*\}", reply, re.S)):
        try:
            d = json.loads(cand)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("adverse_events"), list):
            return d["adverse_events"]
    return None


def ade(a):
    if not os.path.isfile(a.data):
        sys.exit(f"{a.data} not found -- the dataset step has to run first")
    rows = [json.loads(l)["messages"] for l in open(a.data)][: a.n]

    print(f"\nloading {a.model} (no fine-tuning yet)...")
    model, tok = chatlib.load(a.model)

    print()
    rule("=")
    print("  The UNTRAINED model on adverse-drug-event extraction")
    rule("=")
    print("""
  Each sentence below comes from a published case report. The model is
  asked for a JSON list of the (drug, effect) pairs the sentence reports.
  GOLD is the human annotation -- the right answer.

  Watch what the base model does with it.
""")

    found = wanted = 0
    for msgs in rows:
        prompt, gold = msgs[0]["content"], json.loads(msgs[1]["content"])["adverse_events"]
        reply = chatlib.ask(model, tok, prompt, max_new_tokens=150, greedy=True)
        events = parse_events(reply)
        found += len(events or [])
        wanted += len(gold)
        rule()
        print("SENTENCE:", prompt.split("Sentence: ", 1)[-1])
        print("GOLD    :", json.dumps(gold, ensure_ascii=False))
        print("BASE    :", reply[:300].replace("\n", " "))

    rule("=")
    print(f"  across {len(rows)} sentences the base model proposed {found} drug/effect "
          f"pair(s); the gold annotations contain {wanted}")
    print("  it follows the JSON format it was asked for, but it has never learned")
    print("  to actually pull the spans out -- so it mostly answers with an empty")
    print("  list. That gap is what the next steps close.")
    rule("=")


# --------------------------------------------------------------------------- genomic
def genomic(a):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"\nloading {a.model} with a fresh, untrained classification head...")
    ds = load_dataset(a.dataset, split="test").filter(lambda r: r["task"] == a.task)
    ds = ds.shuffle(seed=0).select(range(min(a.n, len(ds))))
    seqs, labels = ds["sequence"], ds["label"]
    num_labels = len(set(labels))

    torch.manual_seed(0)  # reproducible untrained head
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=num_labels, dtype=torch.bfloat16).eval().to("cuda")

    preds = []
    for i in range(0, len(seqs), a.batch):
        enc = tok(seqs[i:i + a.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            preds += model(**enc).logits.argmax(-1).tolist()
    model.to("cpu")
    torch.cuda.empty_cache()

    print()
    rule("=")
    print(f"  The UNTRAINED classifier on '{a.task}'")
    rule("=")
    print(f"""
  The DNA foundation model was pretrained only to fill in masked
  nucleotides -- it has never been told what a promoter is. The
  classification head reading its embeddings is randomly initialised.

  {num_labels} classes, {len(seqs)} held-out sequences of the human reference genome.
""")
    for i in range(min(a.show, len(seqs))):
        rule()
        print(f"SEQ  {seqs[i][:60]}...  ({len(seqs[i])} bp)")
        print(f"     true={labels[i]}   predicted={preds[i]}"
              f"   {'hit' if preds[i] == labels[i] else 'miss'}")

    acc = sum(p == t for p, t in zip(preds, labels)) / len(labels)
    rule("=")
    print(f"  accuracy over {len(seqs)} sequences: {acc:.3f}   (chance is {1/num_labels:.3f})")
    print("  no signal at all -- the embeddings are informative, but nothing has")
    print("  taught the head how to read them for this question. Training next.")
    rule("=")


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--track", required=True, choices=["pirate", "ade", "genomic"])
    ap.add_argument("--model", default=None, help="base model; default depends on the track")
    ap.add_argument("--out", default="out/pirate-baseline.json", help="pirate: where to save your questions")
    ap.add_argument("--data", default="data/ade_eval.jsonl", help="ade: held-out sentences")
    ap.add_argument("--task", default="promoter_all", help="genomic: benchmark task")
    ap.add_argument("--dataset", default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    ap.add_argument("--n", type=int, default=None, help="examples to run")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    if a.track == "genomic":
        a.model = a.model or "InstaDeepAI/nucleotide-transformer-500m-human-ref"
        a.n = a.n or 300
        genomic(a)
    else:
        a.model = a.model or "Qwen/Qwen2.5-0.5B-Instruct"
        if a.track == "pirate":
            pirate(a)
        else:
            a.n = a.n or 6
            ade(a)


if __name__ == "__main__":
    main()
