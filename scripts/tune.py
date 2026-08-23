#!/usr/bin/env python3
"""Guided entry point: pick a fine-tuning track, run its steps in order.

Level 1 is the whimsical pirate introduction; Level 2 adds two practical demos
(medical extraction and genomics) that show why fine-tuning matters for real
scientific work. Each step's own menus (model picker, task picker) still appear.
Run on the GB10 with the venv active; scripted/non-interactive runs should use
run_all.sh or the per-track scripts directly.
"""
import os, subprocess, sys

# (label, argv, output that marks the step done -- None means always run)
TRACKS = [
    ("Pirate chat model", "whimsical intro -- teach Qwen to talk like a pirate", "~50 min", [
        ("build dataset", ["scripts/make_dataset.py"], "data/pirate_train.jsonl"),
        ("fine-tune",     ["scripts/train.py"], "out/pirate-lora"),
        ("evaluate",      ["scripts/eval.py"], None),
    ]),
    ("Medical ADE extractor", "practical -- pull structured drug/side-effect JSON from case reports", "~15 min", [
        ("build dataset", ["scripts/ade_make_dataset.py"], "data/ade_train.jsonl"),
        ("fine-tune",     ["scripts/train.py", "--data", "data/ade_train.jsonl", "--out", "out/ade-lora"], "out/ade-lora"),
        ("evaluate",      ["scripts/ade_eval.py"], None),
    ]),
    ("Genomic classifier", "scientific -- teach a DNA foundation model to spot promoters/splice sites", "~25 min", [
        ("fine-tune", ["scripts/genomic_train.py"], None),
        ("evaluate",  ["scripts/genomic_eval.py"], None),
    ]),
]

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

    print("\nWhat would you like to fine-tune?\n")
    print(f"  {'#':>2}  {'track':<24} {'~time':>7}  what you get")
    print("  " + "-" * 88)
    for i, (name, blurb, est, _) in enumerate(TRACKS, 1):
        mark = "  <- start here" if i == 1 else ""
        print(f"  {i:>2}  {name:<24} {est:>7}  {blurb}{mark}")

    if not sys.stdin.isatty():
        sys.exit("\nthis launcher is interactive -- use run_all.sh or the per-track "
                 "scripts (see README) for scripted runs")
    while True:
        try:
            raw = input("\nchoice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\naborted")
        if not raw:
            n = 1; break
        if raw.isdigit() and 1 <= int(raw) <= len(TRACKS):
            n = int(raw); break
        print(f"  pick 1-{len(TRACKS)}")

    name, _, _, steps = TRACKS[n - 1]
    for label, argv, done_marker in steps:
        if done_marker and os.path.exists(done_marker):
            try:
                redo = input(f"\n[{name}] {label}: {done_marker} already exists -- re-run? [y/N]: ")
            except (EOFError, KeyboardInterrupt):
                sys.exit("\naborted")
            if redo.strip().lower() not in ("y", "yes"):
                print(f"[{name}] {label}: skipped")
                continue
        print(f"\n===== [{name}] {label} =====")
        rc = subprocess.call([sys.executable] + argv)
        if rc != 0:
            sys.exit(f"[{name}] {label} failed (exit {rc})")
    print(f"\n[{name}] done")

if __name__ == "__main__":
    main()
