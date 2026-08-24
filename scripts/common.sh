#!/usr/bin/env bash
# Shared config. Override any of these via environment variables.
REMOTE_HOST="${REMOTE_HOST:-smat924@lais01.cer.auckland.ac.nz}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
REMOTE_DIR="${REMOTE_DIR:-~/ahoy_training}"
SSH="ssh -i $SSH_KEY -o ServerAliveInterval=30 $REMOTE_HOST"
