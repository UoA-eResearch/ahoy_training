#!/usr/bin/env python3
"""Shared model registry + numbered picker menu for train.py / make_dataset.py.

Every model listed here downloads from HuggingFace anonymously -- no HF_TOKEN and
no license click-through. That rules out the Llama, Gemma and Mistral instruct
repos, which are all gated, so the ladder is Qwen top to bottom.

Sizing is for a single DGX GB10: 128 GB unified memory, ~110 GB usable. LoRA in
bf16 needs ~2 bytes/param for the frozen weights, plus activations and a
batch x seq x 152k-vocab logits tensor -- which is why `batch` shrinks as the
models grow. Times scale from the measured 0.5B run (4.5 min, 3 epochs).
"""
import sys

# tier -> (annotation shown in the menu, needs y/N confirmation)
TIERS = {
    "default":  ("fastest; weak on facts", False),
    "easy":     ("comfortable", False),
    "step-up":  ("good balance", False),
    "warn":     ("large -- slow, big download", True),
    "heavy":    ("very large -- hours, near the memory limit", True),
    "blocked":  ("WILL NOT FIT on one GB10", True),
    "unknown":  ("not in the list -- size unknown, using cautious defaults", True),
}

MODELS = [
    # id                              params  vram  train_est  gen_est  batch gen_batch ckpt  tier
    dict(id="Qwen/Qwen2.5-0.5B-Instruct", params="0.5B", vram_gb=3,   train_est="~5 min",  gen_est="~10 min", batch=8, gen_batch=64, grad_ckpt=False, tier="default"),
    dict(id="Qwen/Qwen2.5-1.5B-Instruct", params="1.5B", vram_gb=6,   train_est="~10 min", gen_est="~20 min", batch=8, gen_batch=64, grad_ckpt=False, tier="easy"),
    dict(id="Qwen/Qwen2.5-3B-Instruct",   params="3B",   vram_gb=10,  train_est="~15 min", gen_est="~25 min", batch=8, gen_batch=48, grad_ckpt=False, tier="easy"),
    dict(id="Qwen/Qwen2.5-7B-Instruct",   params="7B",   vram_gb=20,  train_est="~30 min", gen_est="~40 min", batch=4, gen_batch=48, grad_ckpt=False, tier="step-up"),
    dict(id="Qwen/Qwen2.5-14B-Instruct",  params="14B",  vram_gb=36,  train_est="~1 h",    gen_est="~1.3 h",  batch=2, gen_batch=24, grad_ckpt=True,  tier="warn"),
    dict(id="Qwen/Qwen2.5-32B-Instruct",  params="32B",  vram_gb=74,  train_est="~2.5 h",  gen_est="~3 h",    batch=1, gen_batch=12, grad_ckpt=True,  tier="heavy"),
    dict(id="Qwen/Qwen2.5-72B-Instruct",  params="72B",  vram_gb=150, train_est="n/a",     gen_est="n/a",     batch=1, gen_batch=8,  grad_ckpt=True,  tier="blocked"),
]

DEFAULT_STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TEACHER = "Qwen/Qwen2.5-7B-Instruct"

BLOCKED_HELP = (
    "72B needs ~150 GB in bf16 and a GB10 has ~110 GB usable. To train it you'd\n"
    "need 4-bit QLoRA on one box, or bf16 LoRA split across two GB10s with\n"
    "pipeline parallelism. Neither is wired up in these scripts."
)


def lookup(model_id):
    """Registry entry for a model id, or a conservative synthesized one if unlisted."""
    for m in MODELS:
        if m["id"].lower() == model_id.lower():
            return m
    return dict(id=model_id, params="?", vram_gb=None, train_est="?", gen_est="?",
                batch=2, gen_batch=16, grad_ckpt=True, tier="unknown")


def format_menu(est_key="train_est", est_header="train", default_id=None):
    lines = [f"  {'#':>2}  {'model':<30} {'params':>6} {'~mem':>6} {'~' + est_header:>8}  notes",
             "  " + "-" * 78]
    for i, m in enumerate(MODELS, 1):
        note = TIERS[m["tier"]][0]
        if m["id"] == default_id:
            note += "  <- default"
        vram = f"{m['vram_gb']} GB" if m["vram_gb"] else "?"
        lines.append(f"  {i:>2}  {m['id']:<30} {m['params']:>6} {vram:>6} {m[est_key]:>8}  {note}")
    lines.append(f"  {0:>2}  (enter any other HuggingFace repo id)")
    return "\n".join(lines)


def choose_model(purpose="student", default_id=DEFAULT_STUDENT, est_key="train_est", est_header="train"):
    """Show the numbered menu and return a registry entry.

    Falls back to `default_id` without prompting when stdin isn't a TTY, so the
    ssh-driven run_all.sh path never hangs waiting for input.
    """
    if not sys.stdin.isatty():
        print(f"[non-interactive: using default {purpose} model {default_id} "
              f"-- pass --model/--teacher or --list-models to choose]")
        return lookup(default_id)

    default_n = next((i for i, m in enumerate(MODELS, 1) if m["id"] == default_id), 1)
    print(f"\nWhich {purpose} model?  (all download from HuggingFace without a token)\n")
    print(format_menu(est_key, est_header, default_id))
    while True:
        try:
            raw = input(f"\nchoice [{default_n}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\naborted")
        if not raw:
            return MODELS[default_n - 1]
        if raw == "0":
            try:
                custom = input("HuggingFace repo id: ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit("\naborted")
            if custom:
                return lookup(custom)
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(MODELS):
            return MODELS[int(raw) - 1]
        print(f"  pick 0-{len(MODELS)}")


def confirm(entry, est_key="train_est", assume_yes=False):
    """Warn about (or refuse) big models. Exits non-zero on a blocked tier."""
    note, needs_ok = TIERS[entry["tier"]]
    if entry["tier"] == "blocked":
        print(f"\n{entry['id']} ({entry['params']}) will not fit on one GB10 "
              f"-- needs ~{entry['vram_gb']} GB.\n\n{BLOCKED_HELP}")
        sys.exit(1)
    if not needs_ok:
        return
    vram = f"~{entry['vram_gb']} GB" if entry["vram_gb"] else "unknown"
    print(f"\n!! {entry['id']} ({entry['params']}) -- {note}")
    print(f"   memory: {vram} of ~110 GB usable")
    dl = f", plus a ~{entry['params']} x 2 GB download" if entry["vram_gb"] else ""
    print(f"   time:   {entry[est_key]}{dl}")
    if assume_yes or not sys.stdin.isatty():
        print("   proceeding")
        return
    try:
        ok = input("   continue? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ok = ""
    if ok not in ("y", "yes"):
        sys.exit("\naborted")


if __name__ == "__main__":
    print(format_menu())
