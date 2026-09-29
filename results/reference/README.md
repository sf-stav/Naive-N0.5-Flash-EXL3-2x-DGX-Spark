# Reference results (2026-09-28)

These are the raw `scripts/bench.py` outputs behind the README tables. All runs used 2× DGX Spark,
TP2+EP2, 1 request, and 6 mixed chat prompts (greedy, natural EOS, at most 384 tokens).

| File | Config | Decode tok/s (weighted) |
|---|---|---|
| `*-baseline-spec6-eager.json` | eager, DSpark k=6 | 20.2 |
| `*-nospec-eager.json` | eager, no speculation | 21.5 |
| `*-nospec-graphs.json` | CUDA graphs, no speculation | 23.5 |
| `*-nospec-graphs-coop.json` | graphs, `VLLM_EXL3_COOP=1` | 22.6 |
| `*-nospec-graphs-hint4.json` | graphs, MoE launch hint 4 | 21.8 |
| `*-spec3-graphs.json` | graphs, DSpark k=3 (2048 context) | 24.4 |
| `*-spec6-adaptive-graphs.json` | graphs, k=6 + adaptive verification | 22.6 |
| `*-spec2-graphs-128k-parsers.json` | graphs, k=2, 128K context, parsers + corrected template | 24.1 |
| `*-recipe-k3-final.json` | **the final recipe** (k=3, 128K, parsers, template); also prefill to 33K and needle recall | 23.3 |

The first seven runs used `max_model_len` 2048, and their NLL figures were computed on a different
reference text. The last two use the current `scripts/bench_ref_text.txt` (NLL about 2.03). Decode
figures vary by about ±1 tok/s between runs.

The code, mixed and reasoning numbers in the README come from an external harness
(VeloBenchmark 0.1.0) run against the served recipe with 128K context.
