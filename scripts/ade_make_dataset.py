#!/usr/bin/env python3
"""Build the adverse-drug-event (ADE) extraction dataset.

Sentences from published PubMed case reports (ade-benchmark-corpus/ade_corpus_v2,
ungated -- a standard NLP benchmark, no patient records) paired with the gold
{"drug", "effect"} annotations, formatted as chat messages so the same SFT
pipeline that learned pirate speak can learn structured medical extraction:

  user:      instruction + sentence
  assistant: {"adverse_events": [{"drug": ..., "effect": ...}]}   (or [])

Negative sentences (no ADE reported) are mixed in so the model learns to output
an empty list instead of hallucinating. No GPU needed; runs in seconds.
"""
import argparse, json, os, random

INSTRUCTION = (
    "Extract adverse drug events from the sentence below. Reply with ONLY a JSON "
    'object of the form {"adverse_events": [{"drug": "...", "effect": "..."}]}. '
    "If the sentence reports no adverse drug event, use an empty list."
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--negatives", type=int, default=1500)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    from datasets import load_dataset
    rel = load_dataset("ade-benchmark-corpus/ade_corpus_v2",
                       "Ade_corpus_v2_drug_ade_relation", split="train")
    cls = load_dataset("ade-benchmark-corpus/ade_corpus_v2",
                       "Ade_corpus_v2_classification", split="train")

    # one example per sentence, with every (drug, effect) pair it reports
    by_text = {}
    for r in rel:
        pairs = by_text.setdefault(r["text"].strip(), [])
        pair = {"drug": r["drug"].strip(), "effect": r["effect"].strip()}
        if pair not in pairs:
            pairs.append(pair)
    print(f"{len(rel)} relation rows -> {len(by_text)} unique sentences")

    positive_texts = set(by_text)
    negatives = sorted({r["text"].strip() for r in cls
                        if r["label"] == 0 and r["text"].strip() not in positive_texts})
    random.shuffle(negatives)
    negatives = negatives[: a.negatives]
    print(f"{len(negatives)} negative sentences (no ADE)")

    rows = [(t, pairs) for t, pairs in sorted(by_text.items())] + [(t, []) for t in negatives]
    random.shuffle(rows)

    def to_messages(text, pairs):
        return {"messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n\nSentence: {text}"},
            {"role": "assistant", "content": json.dumps({"adverse_events": pairs}, ensure_ascii=False)},
        ]}

    ev, tr = rows[: a.eval_n], rows[a.eval_n:]
    for name, part in (("ade_train.jsonl", tr), ("ade_eval.jsonl", ev)):
        with open(os.path.join(a.out, name), "w") as f:
            for text, pairs in part:
                f.write(json.dumps(to_messages(text, pairs), ensure_ascii=False) + "\n")
        print("wrote", name, len(part))

if __name__ == "__main__":
    main()
