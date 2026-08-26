#!/usr/bin/env python3
"""Shared model registry + numbered picker menu for train.py / make_dataset.py.

Every model listed here downloads from HuggingFace anonymously -- no HF_TOKEN and
no license click-through. That rules out the Llama, Gemma and Mistral instruct
repos, which are all gated, so the ladder is Qwen top to bottom.

Sizing is for a single DGX GB10: 128 GB unified memory, ~110 GB usable. LoRA in
bf16 needs ~2 bytes/param for the frozen weights, plus activations and a
batch x seq x 152k-vocab logits tensor -- which is why `batch` shrinks as the
models grow. Times scale from the measured 0.5B run (4.5 min, 3 epochs).

The 72B is the exception: in bf16 it beats one box's memory. As the STUDENT it
trains (and evaluates) as 4-bit QLoRA on one box; as the TEACHER it is served
full-precision across a cabled PAIR of GB10s (see pair.py), with a preflight.
"""
import sys

# tier -> (annotation shown in the menu, needs y/N confirmation)
TIERS = {
    "default":  ("fastest; weak on facts", False),
    "easy":     ("comfortable", False),
    "step-up":  ("good balance", False),
    "warn":     ("large -- slow, big download", True),
    "heavy":    ("very large -- hours, near the memory limit", True),
    "xl":       ("4-bit QLoRA here; as teacher needs a cabled pair", True),
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
    dict(id="Qwen/Qwen2.5-72B-Instruct",  params="72B",  vram_gb=45,  train_est="~17 h",   gen_est="~2.5 h",  batch=1, gen_batch=8,  grad_ckpt=True,  tier="xl"),
]

DEFAULT_STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TEACHER = "Qwen/Qwen2.5-7B-Instruct"

XL_TRAIN_HELP = (
    "   72B in bf16 is ~150 GB against one GB10's ~110 GB usable, so here it\n"
    "   trains as QLoRA: the frozen base is loaded 4-bit (~40 GB) and the LoRA\n"
    "   adapter trains in bf16 on top -- same recipe, quantized base. It runs\n"
    "   on this one box; no second GB10 involved. Evaluation and chat load the\n"
    "   base the same 4-bit way, so before/after stays like for like.\n"
    "   (There is also an EXPERIMENTAL bf16 path across a cabled pair of GB10s\n"
    "   -- train.py --two-node -- but its working set rides the memory ceiling\n"
    "   of both boxes; see the README before trying it.)"
)

XL_GEN_HELP = (
    "   72B in bf16 is ~150 GB against one GB10's ~110 GB usable, so as a\n"
    "   TEACHER it is served across a CABLED PAIR of GB10s: vLLM with tensor\n"
    "   parallel 2 over Ray, through both GPUs, generating via its API.\n"
    "   Needs: run on the pair's head, passwordless SSH to the worker\n"
    "   (192.168.100.2), and both GPUs free."
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
    """Warn about big models; a two-node model also gets a pair preflight.

    Exits non-zero when the user declines, or when a two-node model is picked
    on a box that is not the head of a working pair (nothing to confirm then
    -- it cannot run).
    """
    note, needs_ok = TIERS[entry["tier"]]
    if not needs_ok:
        return
    vram = f"~{entry['vram_gb']} GB" if entry["vram_gb"] else "unknown"
    print(f"\n!! {entry['id']} ({entry['params']}) -- {note}")
    if entry["tier"] == "xl" and est_key == "gen_est":
        # teacher role: full-precision serving across the pair -- preflight it
        print(XL_GEN_HELP)
        import pair
        problems = pair.check()
        if problems:
            sys.exit("\ncannot run it from this box:\n  - " + "\n  - ".join(problems))
        print("\n   pair check OK: this is the head, and the worker answers on the rail")
        print(f"   time:   {entry[est_key]} (estimate), plus a ~150 GB download "
              "on EACH box on first use")
        question = "   serve it across both GB10s of this pair? [y/N]: "
    elif entry["tier"] == "xl":
        # student role: 4-bit QLoRA, one box, no pair needed
        print(XL_TRAIN_HELP)
        print(f"   memory: {vram} of ~110 GB usable (4-bit base + bf16 adapter)")
        print(f"   time:   {entry[est_key]} (estimate), plus a ~150 GB download on first use")
        question = "   continue with 4-bit QLoRA on this box? [y/N]: "
    else:
        print(f"   memory: {vram} of ~110 GB usable")
        dl = f", plus a ~{entry['params']} x 2 GB download" if entry["vram_gb"] else ""
        print(f"   time:   {entry[est_key]}{dl}")
        question = "   continue? [y/N]: "
    if assume_yes or not sys.stdin.isatty():
        print("   proceeding")
        return
    try:
        ok = input(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        ok = ""
    if ok not in ("y", "yes"):
        sys.exit("\naborted")


if __name__ == "__main__":
    print(format_menu())
