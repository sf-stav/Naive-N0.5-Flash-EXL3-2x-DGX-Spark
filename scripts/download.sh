#!/usr/bin/env bash
# Download the EXL3 pack (~128 GB) and the DSpark drafter (~1.3 GB) on THIS node. Run on both Sparks
# (or download once and rsync over the RoCE link).
#   bash scripts/download.sh
source "$(dirname "$0")/common.sh"
set -e
[ -x "$VENV/bin/python" ] || die "run scripts/install.sh first"

PACK_REPO=doth4580/Naive-N0.5-Flash-EXL3-3.5bpw
DRAFT_REPO=NaiveAI/Naive-N0.5-Flash-FP8-Draft

"$VENV/bin/python" - "$PACK_REPO" "$MODEL_DIR" "$DRAFT_REPO" "$DRAFT_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
pack_repo, pack_dir, draft_repo, draft_dir = sys.argv[1:]
for repo, d in ((pack_repo, pack_dir), (draft_repo, draft_dir)):
    print(f"==> {repo} -> {d}", flush=True)
    snapshot_download(repo_id=repo, local_dir=d, max_workers=8)
EOF

say "applying the drafter config change"
"$VENV/bin/python" "$REPO/scripts/fix_drafter_config.py" "$DRAFT_DIR"

say "checking the pack"
"$VENV/bin/python" - "$MODEL_DIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
q = json.load(open(os.path.join(d, "config.json")))["quantization_config"]
assert q.get("quant_method") == "exl3" and "non_routed_exl3" in q, "config.json is not the vLLM-rewritten pack config"
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
missing = sorted({f for f in idx.values() if not os.path.exists(os.path.join(d, f))})
assert not missing, f"missing shards: {missing}"
print(f"pack ok: {len(idx)} tensors in {len(set(idx.values()))} shards; non-routed k_words={q['non_routed_exl3'].get('k_words')}")
EOF
