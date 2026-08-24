#!/usr/bin/env python3
"""ahoy_training -- the entry point. Everything runs from here.

Pick one of three fine-tuning tracks, answer a couple of setup questions, and
this walks the whole loop: look at the untrained model, build or load the
training data, train a LoRA adapter, then measure and read the difference.
Each step explains what it is doing and why before it runs.
"""
import os, subprocess, sys, textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models
import genomic_train

W = 78


def rule(char="-"):
    print(char * W)


def para(text, indent="  "):
    """Print pre-written prose, wrapped, keeping blank-line paragraph breaks."""
    for block in textwrap.dedent(text).strip("\n").split("\n\n"):
        print(textwrap.fill(" ".join(block.split()), width=W - 2,
                            initial_indent=indent, subsequent_indent=indent))
        print()


def ask(prompt, choices, default):
    """Prompt for one of `choices` (dict of key -> meaning). Non-TTY -> default."""
    if not sys.stdin.isatty():
        return default
    while True:
        try:
            raw = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\naborted")
        if not raw:
            return default
        if raw in choices:
            return raw
        print(f"  pick one of: {', '.join(choices)}")


# --------------------------------------------------------------------------- setup questions
def setup_pirate():
    teacher = models.choose_model("teacher", models.DEFAULT_TEACHER, "gen_est", "gen")
    models.confirm(teacher, "gen_est")
    student = models.choose_model("student", models.DEFAULT_STUDENT)
    models.confirm(student, "train_est")
    return {"teacher": teacher["id"], "student": student["id"]}


def setup_student_only():
    student = models.choose_model("student", models.DEFAULT_STUDENT)
    models.confirm(student, "train_est")
    return {"student": student["id"]}


def setup_genomic():
    return {"task": genomic_train.choose_task()}


# --------------------------------------------------------------------------- step lists
def pirate_steps(c):
    return [
        dict(
            label="See what you are starting with",
            time="~3 min",
            argv=["scripts/baseline.py", "--track", "pirate", "--model", c["student"]],
            produces="out/pirate-baseline.json",
            what="""
                Loads the stock chat model, untouched, and hands you the
                prompt. Ask it anything you like -- your questions and its
                answers are written to a file.
            """,
            why="""
                Fine-tuning is a before-and-after claim, and without a
                "before" you are taking it on faith. This is also the last
                time this model will sound like every other assistant, which
                is worth hearing once. Step 5 replays these exact questions
                against the trained model.
            """,
        ),
        dict(
            label="Build the training data",
            time="~40 min",
            argv=["scripts/make_dataset.py", "--teacher", c["teacher"], "--yes"],
            produces="data/pirate_train.jsonl",
            what="""
                Takes ~1500 real user questions from the public
                databricks-dolly-15k dataset and has a larger "teacher" model
                answer each one in pirate dialect. Replies that do not
                actually sound piratey are thrown away.
            """,
            why="""
                LoRA learns from examples of the behaviour you want, and
                nobody has 1500 hand-written pirate answers lying around.
                Generating them from a bigger model is called
                self-distillation, and it is the usual answer to "I know what
                I want the model to do, but I have no dataset for it".

                The saved examples deliberately contain no system prompt. The
                student is never told to be a pirate -- it only ever sees
                questions and piratey answers, so the only place the voice can
                end up is in the weights.
            """,
        ),
        dict(
            label="Fine-tune",
            time="~5 min",
            argv=["scripts/train.py", "--model", c["student"], "--yes"],
            produces="out/pirate-lora",
            what="""
                LoRA (rank 16) on every linear layer, with the loss computed
                only on the assistant's tokens. Roughly 1% of the weights are
                trained; the other 99% stay frozen.
            """,
            why="""
                This is the training. Freezing the base model and learning a
                small adapter beside it is why this takes minutes and a few
                hundred megabytes instead of hours and tens of gigabytes --
                and why you can keep the original model and swap adapters in
                and out.
            """,
        ),
        dict(
            label="Measure the difference",
            time="~1 min",
            argv=["scripts/eval.py"],
            produces=None,
            what="""
                Runs the base model and the tuned model side by side on
                held-out prompts, then counts how often each reply contains
                pirate vocabulary.
            """,
            why="""
                Held-out means these prompts were never trained on. That is
                the check that the model learned a general behaviour rather
                than memorising the answers it was shown.
            """,
        ),
        dict(
            label="Read the difference, then chat",
            time="~5 min",
            argv=["scripts/chat.py", "out/pirate-lora-merged"],
            produces=None,
            what="""
                Replays your questions from step 1 -- the old answer and the
                new one printed together -- and then hands you the prompt
                again, this time talking to the fine-tuned model.
            """,
            why="""
                A metric tells you something changed. Reading the two answers
                tells you what changed. Try a question you did not ask the
                first time: if the voice holds up on unseen input, the
                behaviour generalised.
            """,
        ),
    ]


def ade_steps(c):
    return [
        dict(
            label="Build the dataset",
            time="~15 s, no GPU",
            argv=["scripts/ade_make_dataset.py"],
            produces="data/ade_train.jsonl",
            what="""
                Pairs ~5500 sentences from published PubMed case reports with
                their gold (drug, effect) annotations, and mixes in ~1500
                sentences that report no adverse event at all. Everything is
                reformatted as question/answer pairs where the answer is JSON.
            """,
            why="""
                This dataset already exists and is already annotated, so
                unlike the pirate track there is nothing to generate. The
                negative sentences matter as much as the positive ones:
                without them the model learns "always find something" and
                invents drug reactions that were never reported.

                Nothing here is patient data -- these are sentences from
                papers that have been published and de-identified for years.
            """,
        ),
        dict(
            label="See what you are starting with",
            time="~1 min",
            argv=["scripts/baseline.py", "--track", "ade", "--model", c["student"]],
            produces=None,
            what="""
                Shows the untrained model a handful of held-out sentences and
                prints what it extracts next to what the human annotators
                marked.
            """,
            why="""
                The failure is more interesting than a bad score. The base
                model follows the JSON format perfectly and then returns an
                empty list almost every time -- it has learned the shape of
                the answer from general instruction tuning, but nothing has
                ever taught it to find drug and effect spans in text. That is
                a precision and recall of zero, and it is the gap the next
                two steps close.
            """,
        ),
        dict(
            label="Fine-tune",
            time="~10 min",
            argv=["scripts/train.py", "--model", c["student"], "--yes",
                  "--data", "data/ade_train.jsonl", "--out", "out/ade-lora"],
            produces="out/ade-lora",
            what="""
                The same LoRA recipe and the same script as the pirate track,
                pointed at the ADE data instead.
            """,
            why="""
                Worth noticing that nothing about the method changed. The
                difference between a novelty and a working extraction tool is
                entirely in the dataset you point it at.
            """,
        ),
        dict(
            label="Measure the difference",
            time="~20 s",
            argv=["scripts/ade_eval.py"],
            produces=None,
            what="""
                Scores both models on 100 held-out sentences: how often the
                reply is valid JSON, and the precision, recall and F1 of the
                (drug, effect) pairs it extracted.
            """,
            why="""
                Precision is how much of what it found was real; recall is how
                much of what was there it found. Both start at zero because
                the base model proposes nothing, and both end up in the 0.7
                range -- the point at which a model stops being a demo and
                starts being a component you could put in a pipeline.
            """,
        ),
    ]


def genomic_steps(c):
    return [
        dict(
            label="See what you are starting with",
            time="~1 min",
            argv=["scripts/baseline.py", "--track", "genomic", "--task", c["task"]],
            produces=None,
            what="""
                Attaches a fresh, randomly initialised classification head to
                the DNA foundation model and asks it to label held-out
                sequences from the human reference genome.
            """,
            why="""
                The foundation model was pretrained only to fill in masked
                nucleotides -- the genomic equivalent of predicting the next
                word. It has never been told what a promoter is, and the head
                reading its embeddings starts out random, so it scores at
                chance. Everything that follows is teaching it to read
                embeddings it already had.
            """,
        ),
        dict(
            label="Fine-tune",
            time="~8 min",
            argv=["scripts/genomic_train.py", "--task", c["task"]],
            produces=f"out/genomic-{c['task']}-lora",
            what="""
                LoRA on the attention layers plus the classification head,
                trained on the labelled sequences from the published
                Nucleotide Transformer benchmark. Under 1% of the weights
                move.
            """,
            why="""
                Same technique as the other two tracks, but the output is a
                label rather than text. This is the standard way a general
                foundation model gets adapted to one specific scientific
                question without retraining it from scratch.
            """,
        ),
        dict(
            label="Measure the difference",
            time="~20 s",
            argv=["scripts/genomic_eval.py", "--adapter", f"out/genomic-{c['task']}-lora"],
            produces=None,
            what="""
                Scores the untrained head and the fine-tuned model on the same
                held-out test sequences, reporting accuracy and Matthews
                correlation (MCC).
            """,
            why="""
                MCC runs from -1 to 1 and sits at 0 for a model that is
                guessing, which makes it the honest metric here -- it cannot
                be flattered by class imbalance the way accuracy can. Watching
                it move off zero is watching the model actually pick up the
                sequence motifs that mark a real promoter.
            """,
        ),
    ]


# --------------------------------------------------------------------------- tracks
TRACKS = [
    dict(
        name="Pirate chat model",
        blurb="teach a chat model a new voice",
        time="~50 min",
        goal="""
            Take a small general-purpose chat model and change how it answers:
            every reply in pirate dialect, with no system prompt and no prompt
            engineering at inference time. The voice ends up in the weights.

            Changing how a model responds is the most common reason to
            fine-tune in practice -- a house style, a fixed output format, a
            domain persona, a tone. Pirate is a version of that you can hear
            in one sentence, which makes it the clearest first run. The method
            is identical to the other two tracks.
        """,
        setup=setup_pirate,
        steps=pirate_steps,
    ),
    dict(
        name="Medical ADE extractor",
        blurb="turn a chat model into a structured extraction tool",
        time="~11 min",
        goal="""
            Teach the same small model to read a sentence from a medical case
            report and return the adverse drug events it describes as JSON --
            which drug, which effect, and nothing when the sentence reports
            neither.

            This is the shape of most real LLM work: not conversation, but
            reading unstructured text and emitting a structured record that
            something downstream can rely on. Scanning published case reports
            for drug reactions is a real pharmacovigilance task, normally done
            by manual clinical review. The source data is published,
            de-identified literature -- no patient records anywhere in this.
        """,
        setup=setup_student_only,
        steps=ade_steps,
    ),
    dict(
        name="Genomic classifier",
        blurb="adapt a DNA foundation model to a specific biological question",
        time="~9 min",
        goal="""
            Take a 500M-parameter DNA language model and teach it to answer one
            question about a stretch of genome -- by default, does this window
            contain a gene promoter, the switch that turns transcription on.

            Finding regulatory elements in DNA normally means a wet-lab assay
            or a hand-built statistical model. This is the same LoRA technique
            as the other tracks with text swapped for nucleotides, and it is
            how foundation models actually get used in genomics research:
            pretrain once on raw sequence, adapt cheaply per question. The
            sequences are the public human reference genome, not anyone's
            personal data.
        """,
        setup=setup_genomic,
        steps=genomic_steps,
    ),
]


# --------------------------------------------------------------------------- driver
def intro():
    print()
    rule("=")
    print("  ahoy_training -- fine-tuning, three ways")
    rule("=")
    print()
    para("""
        Fine-tuning takes a model that already works and adjusts it until it
        does one thing the way you need it done. These tracks all use LoRA:
        the original weights are frozen and a small adapter is trained beside
        them, which is what makes each run finish in minutes on one GB10.
    """)
    para("""
        All three follow the same four beats -- look at the untrained model,
        get the data, train the adapter, measure and read the difference. The
        first is playful and the other two are real tasks, but the method
        underneath does not change between them.
    """)


def choose_track():
    print(f"  {'#':>2}  {'track':<24} {'~time':>7}  what it does")
    print("  " + "-" * (W - 4))
    for i, t in enumerate(TRACKS, 1):
        mark = "   <- start here" if i == 1 else ""
        print(f"  {i:>2}  {t['name']:<24} {t['time']:>7}  {t['blurb']}{mark}")

    if not sys.stdin.isatty():
        sys.exit("\nthis launcher is interactive -- open a terminal on the GB10 and "
                 "run it there (scripts/tune_remote.sh does that for you)")
    while True:
        try:
            raw = input("\nchoice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\naborted")
        if not raw:
            return TRACKS[0]
        if raw.isdigit() and 1 <= int(raw) <= len(TRACKS):
            return TRACKS[int(raw) - 1]
        print(f"  pick 1-{len(TRACKS)}")


def run_step(track, step, i, n):
    print()
    rule("=")
    print(f"  STEP {i} of {n}  ::  {step['label']}   ({step['time']})")
    rule("=")
    print()
    para(step["what"], indent="  ")
    print("  Why this step\n")
    para(step["why"], indent="    ")
    if step["produces"]:
        print(f"  Produces: {step['produces']}\n")

    done = step["produces"] and os.path.exists(step["produces"])
    if done:
        print(f"  {step['produces']} already exists from an earlier run.")
        choice = ask("  [Enter] skip it   [r] run it again   [q] quit: ",
                     {"r": 1, "q": 1}, default="")
        if choice == "q":
            sys.exit("\nstopped")
        if choice != "r":
            print("  skipped\n")
            return
    else:
        choice = ask("  [Enter] run this step   [s] skip   [q] quit: ",
                     {"s": 1, "q": 1}, default="")
        if choice == "q":
            sys.exit("\nstopped")
        if choice == "s":
            print("  skipped\n")
            return

    print()
    rule()
    rc = subprocess.call([sys.executable] + step["argv"])
    if rc != 0:
        sys.exit(f"\n[{track['name']}] {step['label']} failed (exit {rc})")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

    intro()
    track = choose_track()

    print()
    rule("=")
    print(f"  {track['name']}")
    rule("=")
    print()
    para(track["goal"])

    choices = track["setup"]()
    steps = track["steps"](choices)

    print()
    rule()
    print(f"  {len(steps)} steps:")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s['label']:<34} {s['time']}")
    rule()

    for i, step in enumerate(steps, 1):
        run_step(track, step, i, len(steps))

    print()
    rule("=")
    print(f"  {track['name']}: done")
    rule("=")
    print()
    para("""
        Run this again to try another track, or to redo a step with different
        settings -- anything already built is offered as a skip.
    """)


if __name__ == "__main__":
    main()
