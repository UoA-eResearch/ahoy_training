#!/usr/bin/env bash
# Run ON the GB10. Creates a uv venv with CUDA PyTorch (aarch64, cu130 -> supports GB10 sm_121)
# and the HuggingFace fine-tuning stack. No root/sudo needed.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "uv not found; install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -d .venv ] || uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install numpy "transformers>=5" "peft>=0.20" "trl>=1.0" datasets accelerate huggingface_hub bitsandbytes
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
echo "setup OK"
