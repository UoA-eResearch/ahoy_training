#!/usr/bin/env bash
# Run from your laptop: syncs scripts to the GB10 and runs the full pipeline there.
# Steps: setup -> dataset -> train -> eval. Each step is idempotent-ish; use STEPS to pick.
set -euo pipefail
source "$(dirname "$0")/common.sh"
STEPS="${STEPS:-setup dataset train eval}"
$SSH "mkdir -p $REMOTE_DIR/scripts"
scp -q -i "$SSH_KEY" "$(dirname "$0")"/*.py "$(dirname "$0")"/*.sh "$REMOTE_HOST:$REMOTE_DIR/scripts/"
for s in $STEPS; do
  echo "===== $s ====="
  case $s in
    setup)   $SSH "cd $REMOTE_DIR && bash scripts/setup.sh" ;;
    dataset) $SSH "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/make_dataset.py ${DATASET_ARGS:-}" ;;
    train)   $SSH "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/train.py ${TRAIN_ARGS:-}" ;;
    eval)    $SSH "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/eval.py ${EVAL_ARGS:-}" ;;
    ade-dataset)   $SSH "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/ade_make_dataset.py ${ADE_DATASET_ARGS:-}" ;;
    ade-train)     $SSH "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/train.py --data data/ade_train.jsonl --out out/ade-lora ${ADE_TRAIN_ARGS:-}" ;;
    ade-eval)      $SSH "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/ade_eval.py ${ADE_EVAL_ARGS:-}" ;;
    genomic-train) $SSH "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/genomic_train.py ${GENOMIC_TRAIN_ARGS:-}" ;;
    genomic-eval)  $SSH "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/genomic_eval.py ${GENOMIC_EVAL_ARGS:-}" ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
done
