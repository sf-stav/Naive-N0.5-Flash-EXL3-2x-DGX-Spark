"""Express a turboderp-native Naive-N0.5-Flash EXL3 pack in vllm-exl3's vocabulary.

Adapted from ``vllm-exl3/tools/exl3_pack_tools/qwen_pack_config.py``. The plugin
(``vllm_exl3.exl3.Exl3Config``) needs the pack's ``quantization_config`` to name
*every* EXL3-quantized module:

  * routed experts (``mlp.experts.*``) are served by ``Exl3MoEMethod`` with a
    base K plus per-layer overrides (``layer_bits``, keyed by layer index),
  * non-routed dense linears must appear under ``non_routed_exl3`` so
    ``Exl3Config.get_quant_method`` returns ``Exl3LinearMethod``: the split
    ``self_attn.{q,k,v,o}_proj``, ``self_attn.indexer.{wq,wk,weights_proj}``,
    the layer-0 dense MLP (``mlp.gate_up_proj`` for gate/up, ``mlp.down_proj``)
    and ``lm_head``.

The Qwen tool's fusion/root tables assume a nested language_model/visual tree
and a fused ``qkv_proj``; Naive keeps q/k/v split and roots everything at
``model.``, so this variant only fuses the dense-MLP ``gate_proj``/``up_proj``
into ``gate_up_proj``. Everything else follows the original: the native block
is preserved under ``native_quantization_config``, config.json is backed up to
``config.json.native``, and the safetensors index is rebuilt from the headers
actually present on disk (``model.safetensors.index.json.native`` backup) so
vLLM trusts the file list.

Fractional-K packs (e.g. 3.5bpw -> 56 int16 words per tile) keep the integer
``bits`` vllm-exl3 validates, and additionally carry the exact ``k_words`` per
dense prefix under ``non_routed_exl3`` (vllm_naive_n05.exl3_compat consumes
it).

usage: python3 naive_pack_config.py <pack_dir> [--scan scan.json] [--dry-run]
                                   [--no-index]
"""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def scan_pack(pack):
    """Header-only inventory: per-layer expert K and per-dense-linear K."""
    expert_re = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.(\w+)$")
    expert_k = collections.defaultdict(lambda: collections.defaultdict(set))
    dense_k = {}
    suffixes = collections.defaultdict(set)
    total = 0
    for shard in sorted(glob.glob(os.path.join(pack, "*.safetensors"))):
        header = read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            total += 1
            shape = meta.get("shape") or []
            m = expert_re.search(name)
            if m:
                layer, proj, kind = int(m.group(1)), m.group(3), m.group(4)
                suffixes[f"layers.N.experts.E.{proj}"].add(kind)
                if kind == "trellis" and shape:
                    expert_k[layer][proj].add(int(shape[-1]) // 16)
                continue
            if not name.endswith(".trellis") or not shape:
                continue
            words = int(shape[-1])
            prefix = name[: -len(".trellis")]
            # n-gram/PLE tables are row-wise 2-D trellises, not linear weights.
            if len(shape) == 2:
                continue
            dense_k[prefix] = words
            fam = re.sub(r"\.\d+\.", ".N.", prefix)
            suffixes[fam].add("trellis")

    per_layer = {}
    nonuniform = []
    for layer in sorted(expert_k):
        row = {}
        for proj in sorted(expert_k[layer]):
            ks = sorted(expert_k[layer][proj])
            row[proj] = ks
            if len(ks) != 1:
                nonuniform.append({"layer": layer, "proj": proj, "ks": ks})
        per_layer[layer] = row
    return {
        "pack": pack,
        "tensors": total,
        "expert_k_per_layer": {str(l): r for l, r in per_layer.items()},
        "expert_k_nonuniform": nonuniform,
        "dense_k_by_prefix": dense_k,
        "suffixes_by_family": {f: sorted(s) for f, s in sorted(suffixes.items())},
    }


def build_quant_config(scan, native_q):
    """Turn the scan into the plugin's ``quantization_config`` block."""
    if scan.get("expert_k_nonuniform"):
        print(
            "NOTE: experts within a layer have different K; the plugin keeps exact "
            "per-expert trellis shapes (python_loop for those layers):"
        )
        for r in scan["expert_k_nonuniform"][:10]:
            print(f"  layer {r['layer']} {r['proj']}: {r['ks']}")

    layer_k = {}
    disagree = []
    for layer, row in scan["expert_k_per_layer"].items():
        vals = []
        for proj in row:
            vals.extend(row[proj])
        if not vals:
            continue
        rep = collections.Counter(vals).most_common(1)[0][0]
        layer_k[int(layer)] = rep
        proj_reps = {p: collections.Counter(row[p]).most_common(1)[0][0] for p in row}
        if len(set(proj_reps.values())) != 1:
            disagree.append((layer, row))

    if not layer_k:
        raise SystemExit("REFUSE: no MoE expert K values found in scan")
    if disagree:
        print(
            "NOTE: gate/up/down K disagree within a layer (supported via "
            "python_loop; layer_bits uses the mode K):"
        )
        for layer, row in disagree[:10]:
            print(f"  layer {layer}: {row}")
    base = collections.Counter(layer_k.values()).most_common(1)[0][0]
    layer_bits = {str(l): k for l, k in sorted(layer_k.items()) if k != base}

    # Dense non-routed linears. Naive keeps q/k/v split; only the layer-0 dense
    # MLP is fused by vLLM (packed_modules_mapping: gate_up_proj <- gate/up).
    # ``dense_k_by_prefix`` holds exact k_words (trellis shape[-1]); keep the
    # integer bits for vllm-exl3 validation and carry k_words for fractional
    # packs (3.5bpw -> 56 words), consumed by vllm_naive_n05.exl3_compat.
    fused = {"gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}
    dense_layers = {}
    suffix_set = set()
    for prefix, words in sorted(scan["dense_k_by_prefix"].items()):
        variants = {prefix}
        head, _, leaf = prefix.rpartition(".")
        if leaf in fused:
            variants.add(f"{head}.{fused[leaf]}")
        for v in variants:
            dense_layers[v] = {"bits": int(words) // 16, "k_words": int(words)}
            lm = re.match(r"^model\.layers\.\d+\.(.*)$", v)
            suffix_set.add(lm.group(1) if lm else v)
    if not dense_layers:
        raise SystemExit("REFUSE: no dense EXL3 trellis tensors found in scan")

    word_counts = collections.Counter(
        int(w) for w in scan["dense_k_by_prefix"].values()
    )
    dense_default_words = word_counts.most_common(1)[0][0]
    codebook = str(native_q.get("codebook", "mcg"))
    new_q = {
        "quant_method": "exl3",
        "bits": int(base),
        "codebook": codebook,
        "head_bits": int(native_q.get("head_bits", 16)),
        "scope": "native_all_linears",
        "layer_bits": layer_bits,
        "non_routed_exl3": {
            "codebook": codebook,
            "bits": int(dense_default_words) // 16,
            "k_words": int(dense_default_words),
            "modules": sorted(suffix_set),
            "layers": dense_layers,
        },
        "native_quantization_config": native_q,
        "derived_from_headers": "naive_pack_scan",
    }
    for carry in ("mtp_experts", "mtp_experts_start_layer", "non_routed_dtype_policy"):
        if carry in native_q:
            new_q[carry] = native_q[carry]
    return new_q, layer_bits, dense_layers


def regenerate_index(pack, dry_run=False):
    shards = sorted(glob.glob(os.path.join(pack, "*.safetensors")))
    if not shards:
        print(f"WARNING: no *.safetensors under {pack}; index not regenerated")
        return
    weight_map = {}
    total_size = 0
    dupes = []
    for shard in shards:
        base = os.path.basename(shard)
        header = read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map:
                dupes.append((name, weight_map[name], base))
                continue
            weight_map[name] = base
            offsets = meta.get("data_offsets") or [0, 0]
            total_size += int(offsets[1]) - int(offsets[0])
    if dupes:
        raise SystemExit(
            f"REFUSE: {len(dupes)} tensor name(s) appear in more than one shard: {dupes[:3]}"
        )
    print(
        f"index: {len(shards)} shard(s), {len(weight_map)} tensor(s), "
        f"total_size={total_size}"
    )
    if dry_run:
        return
    index_path = os.path.join(pack, "model.safetensors.index.json")
    backup = index_path + ".native"
    if os.path.exists(index_path) and not os.path.exists(backup):
        shutil.copyfile(index_path, backup)
        print(f"backed up existing index to {backup}")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {index_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--scan", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args()

    pack = args.pack
    cfg_path = os.path.join(pack, "config.json")
    cfg = json.load(open(cfg_path))
    holder = cfg.get("text_config", cfg)
    q = holder.get("quantization_config") or cfg.get("quantization_config") or {}
    if q.get("native_quantization_config"):
        # config.json was already rewritten by this tool; start from the native block
        q = q["native_quantization_config"]

    if args.scan:
        scan = json.load(open(args.scan))
    else:
        scan = scan_pack(pack)

    new_q, layer_bits, dense_layers = build_quant_config(scan, q)
    print(
        f"experts: base K={new_q['bits']}, layer overrides="
        f"{len(layer_bits)} {layer_bits if len(layer_bits) < 20 else '(many)'}"
    )
    print(
        f"dense linears mapped: {len(dense_layers)}; K distribution:",
        dict(collections.Counter(v["bits"] for v in dense_layers.values())),
        "k_words:",
        dict(collections.Counter(v["k_words"] for v in dense_layers.values())),
    )
    print(f"codebook={new_q['codebook']} head_bits={new_q['head_bits']}")
    print("non_routed modules:", new_q["non_routed_exl3"]["modules"])
    if args.dry_run:
        print("dry run, config.json untouched")
        return 0

    backup = cfg_path + ".native"
    if not os.path.exists(backup):
        shutil.copyfile(cfg_path, backup)
    if "text_config" in cfg and "quantization_config" in cfg["text_config"]:
        cfg["text_config"]["quantization_config"] = new_q
    cfg["quantization_config"] = new_q
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"written {cfg_path} (backup {backup})")

    if not args.no_index:
        regenerate_index(pack, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
