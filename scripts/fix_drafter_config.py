"""Set dflash_config.causal = false in the DSpark drafter's config.json (idempotent).

The published NaiveAI/Naive-N0.5-Flash-FP8-Draft config omits the key. Every configuration in this
recipe was measured with it set to false (block-parallel drafting); the original file is kept as
config.json.orig.

    python scripts/fix_drafter_config.py <draft_dir>
"""
import json
import os
import shutil
import sys

d = sys.argv[1]
path = os.path.join(d, "config.json")
cfg = json.load(open(path))
if cfg.get("dflash_config", {}).get("causal") is False:
    print("drafter config already has dflash_config.causal=false")
    sys.exit(0)
if not os.path.exists(path + ".orig"):
    shutil.copyfile(path, path + ".orig")
cfg.setdefault("dflash_config", {})["causal"] = False
json.dump(cfg, open(path, "w"), indent=2)
print("set dflash_config.causal=false (original kept as config.json.orig)")
