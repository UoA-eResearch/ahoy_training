#!/usr/bin/env bash
# Run this from your laptop instead of SSH-ing in by hand.
# Copies the scripts to the GB10, makes sure the environment exists, then opens
# the same interactive launcher you would get by running tune.py on the box.
set -euo pipefail
source "$(dirname "$0")/common.sh"

$SSH "mkdir -p $REMOTE_DIR/scripts"
scp -q -i "$SSH_KEY" "$(dirname "$0")"/*.py "$(dirname "$0")"/*.sh "$REMOTE_HOST:$REMOTE_DIR/scripts/"

# setup.sh is a no-op once the venv exists, so this is safe to run every time
$SSH "cd $REMOTE_DIR && bash scripts/setup.sh"

# -t so the menus, prompts and chat sessions work over the connection
ssh -t -i "$SSH_KEY" -o ServerAliveInterval=30 "$REMOTE_HOST" \
  "cd $REMOTE_DIR && source .venv/bin/activate && python scripts/tune.py"
