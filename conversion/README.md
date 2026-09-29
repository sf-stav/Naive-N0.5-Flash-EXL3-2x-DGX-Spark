# How the EXL3 pack was made

You only need this to rebuild the pack yourself. The published pack
([doth4580/Naive-N0.5-Flash-EXL3-3.5bpw](https://huggingface.co/doth4580/Naive-N0.5-Flash-EXL3-3.5bpw))
is the output of these steps.

## 1. Quantize with ExLlamaV3

- **Source:** [NaiveAI/Naive-N0.5-Flash-FP8](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8), 49 shards, about 315 GB.
- **Converter:** [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) at `07572bd`, with
  `exllamav3-naive-n05.patch` applied. The patch adds the Naive-N0.5-Flash architecture:
  - 39 sliding-window layers with sinks and 9 GQA-DSA layers with the lightning indexer;
  - per-head V padding for `v_head_dim` 128 < `head_dim` 192;
  - the DSA indexer.

  It was validated against the HF reference on a tiny replica of the model: 100% argmax agreement,
  cosine 0.99999, on both the dense and the sparse path.

```bash
git clone https://github.com/vcruz305/exllamav3 && cd exllamav3
git checkout 07572bd && git apply /path/to/this/repo/conversion/exllamav3-naive-n05.patch
pip install --no-build-isolation -e .
python convert.py -i Naive-N0.5-Flash-FP8 -o Naive-N0.5-Flash-exl3-3.5bpw -w work \
  -b 3.5 -hb 6 -cr 250 -cc 2048 -cb mul1 -v
```

Parameters:
- Uniform 3.5 bpw with the `mul1` codebook: 56-word trellises, a fractional K.
- `lm_head` at 6 bpw.
- The embedding is not quantized.
- Calibration: 250 rows × 2048 columns.
- `attention_value_scale` (0.707) is folded into `o_proj` during conversion.

The run used 8×H100 80 GB and took a few hours; the MoE layers took about 170 s each. The pack is
about 128 GiB, so it does not fit on one Spark.

## 2. Rewrite the pack config for vLLM

vllm-exl3 needs the `quantization_config` to name every EXL3 module. It also needs the exact
trellis width for fractional K.

```bash
python conversion/naive_pack_config.py Naive-N0.5-Flash-exl3-3.5bpw
```

What the tool does:
- rewrites `config.json`: routed experts, plus `non_routed_exl3` with `k_words` for attention,
  the indexer, the dense layer-0 MLP and `lm_head`;
- rebuilds `model.safetensors.index.json` from the shard headers;
- keeps the originals as `*.native`.

The weight shards themselves are unchanged. Restore the `.native` files to get the plain
ExLlamaV3 pack back.
