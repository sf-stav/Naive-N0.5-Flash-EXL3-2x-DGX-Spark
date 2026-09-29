#!/usr/bin/env bash
# Build the serving environment on THIS node. Run it on both Sparks.
#   bash scripts/install.sh
# Creates $VENV with the pinned vLLM/torch set (requirements.lock.txt), builds vcruz305/vllm-exl3 and
# vcruz305/exllamav3 at the pinned revisions for sm_121a, and installs this repo's vLLM plugin.
# The CUDA extension builds take a while and use a lot of RAM: stop the server first.
source "$(dirname "$0")/common.sh"
set -e

command -v python3.12 >/dev/null || die "python3.12 not found"
[ -x "$CUDA_HOME/bin/nvcc" ] || die "nvcc not found under CUDA_HOME=$CUDA_HOME"
python3.12 - <<'EOF' || die "Python.h missing: sudo apt install python3.12-dev (or set EXTRA_CPATH)"
import os, sysconfig; assert os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))
EOF

if [ ! -x "$VENV/bin/python" ]; then
  say "creating venv $VENV"
  python3.12 -m venv "$VENV"
fi
PIP="$VENV/bin/python -m pip"
$PIP install -q --upgrade pip setuptools wheel ninja packaging

say "installing pinned packages (vLLM 0.29.0, torch 2.13.0+cu130, ...)"
$PIP install -q torch==2.13.0 --index-url "$TORCH_INDEX"
$PIP install -q -r "$REPO/requirements.lock.txt" --extra-index-url "$TORCH_INDEX"

mkdir -p "$SRC_DIR"
fetch() {  # fetch <repo> <rev> <dir>
  if [ ! -d "$3/.git" ]; then git clone -q "$1" "$3"; fi
  git -C "$3" fetch -q origin
  git -C "$3" checkout -q "$2"
}

export CUDA_HOME TORCH_CUDA_ARCH_LIST PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-8}

say "building vcruz305/exllamav3 @ $EXLLAMAV3_REV (CUDA kernels, sm_121a)"
fetch "$EXLLAMAV3_REPO" "$EXLLAMAV3_REV" "$SRC_DIR/exllamav3"
$PIP install --no-build-isolation --no-deps -e "$SRC_DIR/exllamav3"

say "building vcruz305/vllm-exl3 @ $VLLM_EXL3_REV"
fetch "$VLLM_EXL3_REPO" "$VLLM_EXL3_REV" "$SRC_DIR/vllm-exl3"
$PIP install --no-build-isolation --no-deps -e "$SRC_DIR/vllm-exl3"

say "installing the Naive-N0.5-Flash vLLM plugin"
$PIP install -q --no-build-isolation --no-deps -e "$REPO/plugin"

say "verifying"
"$VENV/bin/python" - <<'EOF'
import torch, vllm, vllm_exl3, exllamav3, vllm_naive_n05
from exllamav3.ext import exllamav3_ext  # compiled kernels
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "| vllm", vllm.__version__, "| cc", torch.cuda.get_device_capability())
assert hasattr(exllamav3_ext, "exl3_moe"), "exllamav3 extension incomplete"
print("ok")
EOF
say "install complete on $(hostname)"
