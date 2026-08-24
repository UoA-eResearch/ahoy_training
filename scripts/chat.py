#!/usr/bin/env python3
"""Talk to the fine-tuned model -- and see it next to the model you started with.

If baseline.py saved your questions before training, they are replayed here:
the answer you got from the untrained model (loaded from that file) printed
directly above the answer the fine-tuned model gives now. Then the prompt is
yours.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatlib


def rule(char="-"):
    print(char * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", default="out/pirate-lora-merged",
                    help="fine-tuned model or adapter directory")
    ap.add_argument("--baseline", default="out/pirate-baseline.json",
                    help="questions saved by baseline.py before training")
    a = ap.parse_args()

    print(f"\nloading {a.model}...")
    model, tok = chatlib.load(a.model)

    saved = None
    if os.path.isfile(a.baseline):
        saved = json.load(open(a.baseline))

    if saved and saved.get("turns"):
        print()
        rule("=")
        print("  BEFORE and AFTER, on the questions you asked earlier")
        rule("=")
        print("\n  Same questions, same clean slate, no system prompt in either case.")
        print("  The only difference is the LoRA adapter trained in between.\n")
        for turn in saved["turns"]:
            after = chatlib.ask(model, tok, turn["question"], greedy=True)
            rule()
            print("Q      :", turn["question"])
            print("BEFORE :", turn["answer"])
            print("AFTER  :", after)
        rule("=")
        print("  Nothing was added to the prompt -- the change is in the weights.")
        rule("=")
    elif not os.path.isfile(a.baseline):
        print(f"  (no {a.baseline}; skipping the before/after replay)")

    if not sys.stdin.isatty():
        print("\n[non-interactive run -- skipping the live chat]")
        return

    print()
    rule("=")
    print("  Now chat with the fine-tuned model")
    rule("=")
    print("\n  This one remembers the conversation. Ask it anything -- including")
    print("  something you did not ask before, to see whether the behaviour")
    print("  generalises or only memorised the training set.")
    print("\n  Blank line, 'done' or ctrl-D to finish.")
    chatlib.interactive(model, tok, label="tuned", keep_history=True)
    print("\ndone")


if __name__ == "__main__":
    main()
