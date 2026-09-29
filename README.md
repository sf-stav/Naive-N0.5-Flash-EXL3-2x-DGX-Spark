# Naive-N0.5-Flash EXL3 (3.5 bpw) on 2× DGX Spark

Serving recipe for [NaiveAI/Naive-N0.5-Flash](https://huggingface.co/NaiveAI/Naive-N0.5-Flash)
(309B MoE, 15.5B active) on **two NVIDIA DGX Spark (GB10)** connected over their ConnectX link.
It uses vLLM with an EXL3 3.5 bpw pack, tensor parallelism plus expert parallelism across both
Sparks, and DSpark speculative decoding.

> **Not optimized.** This recipe exists so people can see what the model can do on 2× DGX Spark.
> It runs, and it was measured, but none of the performance work has been done yet; see
> [Limitations](#limitations). Updates are expected as the recipe is refined.

| | |
|---|---|
| Hardware | 2× DGX Spark (GB10, sm_121, 128 GB unified memory each), RoCE link between them |
| Weights | [doth4580/Naive-N0.5-Flash-EXL3-3.5bpw](https://huggingface.co/doth4580/Naive-N0.5-Flash-EXL3-3.5bpw): EXL3 3.5 bpw, lm_head 6 bpw, about 128 GiB |
| Drafter | [NaiveAI/Naive-N0.5-Flash-FP8-Draft](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8-Draft) (DSpark, 5 layers) |
| Engine | vLLM 0.29.0 + [vllm-exl3](https://github.com/vcruz305/vllm-exl3) + [exllamav3](https://github.com/vcruz305/exllamav3) kernels + the model plugin in `plugin/` |
| Parallelism | TP2 for attention, EP2 for the experts (128 whole experts per Spark), Ray over RoCE |
| Context | 131,072 tokens (a 64K prompt plus a 64K output) |
| Concurrency | 1 request |
| API | OpenAI-compatible, `http://<head-ip>:8000/v1`, model id `naive-n0.5-flash` |

## Performance

Measured 2026-09-28. Single request, greedy decoding, CUDA graphs on, DSpark with 3 draft tokens.

| Workload | Decode tok/s | Notes |
|---|---|---|
| Pure code (C sorting, 4,096 tokens) | **34.9** median | the drafter predicts code well (about 3.1 tokens per step) |
| Mixed code and story | **23.9** median | |
| Reasoning / logic | **19.5** median | about 28% draft acceptance |
| 6 mixed chat prompts (`scripts/bench.py`) | 23.3–24.4 weighted | two runs; code/math 27–35, prose 18–20 |
| Same, no speculative decoding | 23.5 | flat across content types |

| Prompt | Time to first token |
|---|---|
| 613 tokens | 1.1 s |
| 2,088 tokens | 2.7 s |
| 8,364 tokens | 9.6 s (875 tok/s) |
| 33,344 tokens | 41.8 s (800 tok/s) |
| 51,570 tokens | about 77 s |

In the decode table, the first three rows come from an external harness (VeloBenchmark 0.1.0). The
other figures come from
`scripts/bench.py`, with raw results in `results/reference/`. Prefix caching is on, so repeated
prompt prefixes return much faster than the cold times above.

- **Long-context recall:** a passphrase hidden at 10%, 50% and 90% depth of 14K- and 51K-token
  prompts was found every time (6 of 6).
- **Quality check:** teacher-forced NLL on `scripts/bench_ref_text.txt` is 2.03, stable across
  configurations. Use it as a regression check after changes.

How to read these numbers:
- **The step cost is fixed; acceptance varies.** Each speculative step costs about 94 ms, whatever
  the content. How much the drafter gets accepted decides whether speculation beats plain decoding
  (about 42 ms per token). Code gains about 50%; prose and reasoning are 15–20% slower than without
  speculation. Set `SPEC_TOKENS=0` in `config/cluster.env` for steady speed on prose-heavy work.
- **Where a decode step goes** (profiled):
  - routed-expert kernel: 55% of the step, running well below the memory-bandwidth limit;
  - cross-node all-reduces over RoCE: 17%;
  - attention projections: 14%.

  The weight-read floor is about 13 ms per token. The current 42 ms leaves a lot of headroom.
- **Loading** takes about 7.5 minutes: weights about 6.5 minutes, compile and CUDA graph capture
  about 1 minute.

## Requirements

- **Hardware:** two DGX Sparks with a working RoCE link between their ConnectX ports; NCCL must be
  able to use it (`ibv_devinfo` shows `PORT_ACTIVE`).
- **SSH:** passwordless SSH from either node to the other.
- **Software on both nodes:** Ubuntu with CUDA 13.0 (`/usr/local/cuda-13.0`), Python 3.12 with
  headers (`sudo apt install python3.12-dev`), `git`, `screen`.
- **Disk:** about 130 GB per node for the weights, plus about 15 GB for the venv and builds.
- **Memory headroom:** each Spark gives vLLM 80% of its memory (`GPU_MEM_UTIL=0.80`). Keep the
  Sparks otherwise idle while serving. Browsers, desktop sessions or large uploads on either node
  can push it into the out-of-memory killer (earlyoom), which takes down the whole server; this
  happened during testing. If you must run other things, lower `GPU_MEM_UTIL`.

## Quick start

Clone this repo **at the same path on both Sparks**, then:

```bash
# 1. Configure (both nodes use the same file)
cp config/cluster.env.example config/cluster.env
$EDITOR config/cluster.env        # HEAD_IP / WORKER_IP on the RoCE link, NET_IF, NCCL_IB_HCA, paths

# 2. Build the environment -- on BOTH nodes (compiles the EXL3 CUDA kernels for sm_121a)
bash scripts/install.sh

# 3. Download weights + drafter -- on BOTH nodes (or download once and rsync)
bash scripts/download.sh

# 4. Start (from either node): Ray head + worker, then the API server on the head
bash scripts/start.sh

# 5. Test
bash scripts/smoke.sh             # expects '323'
python3 scripts/test_chat.py --base http://<head-ip>:8000   # reasoning + tool-calling checks (24)
curl http://<head-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "naive-n0.5-flash",
  "messages": [{"role": "user", "content": "Write a C function that reverses a linked list."}],
  "max_tokens": 1024, "temperature": 0.6,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

## Operating

| Command | What it does |
|---|---|
| `bash scripts/start.sh` | Start Ray (if needed) and the server; waits until ready |
| `bash scripts/stop.sh [--all]` | Stop the server (`--all` also stops Ray on both nodes) |
| `bash scripts/status.sh [-f]` | Up/down, free memory on both nodes, recent throughput (`-f` follows the log) |
| `bash scripts/smoke.sh` | One short request |
| `python3 scripts/test_chat.py --base http://<head-ip>:8000` | Reasoning split and tool-calling checks (see below) |
| `python3 scripts/bench.py --base http://<head-ip>:8000 --tag X` | Quality (NLL) + decode + prefill benchmark; `--suites needle` for long context; `--compare` for runs |

The server runs in `screen` session `naive_n05` on the head node. Its log is `$RUN_DIR/serve.log`.

Settings in `config/cluster.env`:
- `MAX_MODEL_LEN`
- `MAX_NUM_SEQS`
- `SPEC_TOKENS` (0 = off)
- `GPU_MEM_UTIL`
- `PORT`
- `SERVED_NAME`
- `REASONING_PARSER`, `TOOL_PARSER`, `CHAT_TEMPLATE` (see below)

`scripts/serve.sh` is the complete `vllm serve` command line.

## Reasoning and tool calling

The model thinks in a `<think>...</think>` block and calls tools in Qwen3-Coder XML. The recipe
sets it up so the OpenAI API returns both properly:

| Request | Result |
|---|---|
| default | thinking on: `reasoning_content` holds the thinking, `content` the answer |
| `"chat_template_kwargs": {"enable_thinking": false}` or `"reasoning_effort": "none"` | no thinking, answer only |
| `"reasoning_effort"`: `minimal`/`low`, `medium`/`high`, `xhigh`/`max` | the model's Low / High / Max reasoning levels (default without it: Max) |
| `"tools": [...]` (auto, `required` or a named tool) | parsed `tool_calls` with JSON arguments typed per schema; parallel calls supported |

Three pieces make this work, each found by testing against the served model:

- **Reasoning parser `qwen3`.** `<think>` and `</think>` are ordinary tokens in this vocabulary.
  Without a parser, the whole thinking block ends up in `content`.
- **Corrected chat template** (`config/chat_template.jinja`). It changes two lines of NaiveAI's
  official template:
  - With thinking off, the official template pre-fills `<think></think>`, and the model then
    re-opens `<think>` about 86% of the time. Adding the blank line the model always writes after
    `</think>` makes it answer directly (96%).
  - OpenAI effort levels are mapped onto the model's three levels. The official template sends
    everything except `low`/`high` (including `medium`) to Max.
- **Tool parser `naive_n05`** (`plugin/vllm_naive_n05/tool_parser.py`).
  - The problem: at 3.5 bpw the model reliably starts a call with the `<tool_call>` token, but then
    often drifts into its own format. Examples: `<tool_call> function=getWeather city="Lisbon"`,
    or a parameter closed as `</parameter days>`. No stock parser can read these, and vLLM's
    built-in grammar never engages, because it waits for the exact `<tool_call>\n<function=` prefix.
  - The fix: this parser keeps vLLM's Qwen3-Coder parsing, and adds a grammar that engages on
    `<tool_call>`. It restricts the function name to the declared tools and each argument to its
    schema type: numbers, enums and booleans as such; strings as free text that may contain code
    or markup, but not a malformed parameter close.
  - Text before the call, including the thinking, is not constrained.

`scripts/test_chat.py` checks all of this: 24 cases, thinking on and off, streaming and
non-streaming, picking among several tools, code and HTML inside arguments, no call when none is needed,
parallel calls, and the tool-result round trip.

## How it works

| Component | Revision | Role |
|---|---|---|
| vLLM | 0.29.0 (PyPI, aarch64) | engine, V2 model runner, DSpark speculative decoding, Ray executor |
| torch | 2.13.0+cu130 | — |
| [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) | `08ed1bf` | EXL3 quantization plugin for vLLM (routed experts, dense linears) |
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) | `07572bd` | EXL3 CUDA kernels, including fractional-K (3.5 bpw) trellis kernels |
| `plugin/vllm_naive_n05` | this repo | Naive-N0.5-Flash model for vLLM, fractional-K compatibility shim, DSpark drafter loader, grammar-constrained tool parser |

The full package set is pinned in `requirements.lock.txt`. It is identical on both test nodes.

What the plugin does (it is loaded through vLLM's `vllm.general_plugins` entry point, so vLLM
itself is not patched):
- **Model:** `NaiveN05FlashForCausalLM`.
  - 39 sliding-window layers (window 128, attention sinks) and 9 DSA layers.
  - Split q/k/v projections. V is 128-dim against 192-dim QK, so it uses vLLM's DiffKV attention
    backend.
  - Experts go through vLLM's fused-MoE layer, and vllm-exl3 supplies the EXL3 kernels.
- **Fractional-K shim** (`exl3_compat.py`). vllm-exl3 assumes integer bits per weight; this pack is a
  uniform 3.5 bpw (56-word trellises). The shim carries the exact trellis width through config and
  loading.
- **DSpark drafter** (`dspark_draft.py`). It loads the drafter's learned mask embedding (the
  upstream loader drops it), and it keeps full KV for the drafter's sliding-window layers.
- **`method=dspark`, not `dflash`.** The drafter uses anchor sampling (hidden state at position p
  predicts token p+1). `dflash` semantics put every draft one position late, and acceptance drops
  to about 3%.

`scripts/download.sh` also sets `dflash_config.causal=false` in the drafter's `config.json`. All
measurements used that setting.

## Limitations

- **Beyond 2,048 tokens the DSA layers are approximate.** The model's 9 DSA layers attend to the
  top-2,048 tokens picked by a learned indexer. This recipe runs those layers as dense attention
  over the whole context. That is exact up to 2,048 tokens, but past that point they see more than
  the real model does. Long-context answers may differ from the reference model. The 1M native
  context needs a sparse attention backend, which is not implemented yet.
- **Speculative decoding helps code, costs prose.** See [Performance](#performance).
- **One request at a time.** The KV pool fits about two full 128K contexts. More concurrency needs
  work on the mid-batch EXL3 path.
- **Reasoning and tool-call details:**
  - With thinking on, `content` starts with the two newlines the model writes after `</think>`.
    It is returned as written, so multi-turn history round-trips exactly.
  - Without `reasoning_effort` the template's default is Max reasoning, which can be long. Pass
    `"reasoning_effort": "low"` for faster answers.
  - The tool grammar expects arguments in schema order.
  - A one-line string argument cannot contain a closing tag such as `</b>`. That is the price of
    blocking the model's malformed closes. Multi-line values (files, code, HTML) can contain them.
  - With `tool_choice="required"` the call comes first, with no thinking before it.
- **No fp8 KV cache.** The DiffKV Triton backend on GB10 supports only bf16 KV.
- **Start-up takes about 7.5 minutes,** and the log prints many harmless "trellis staging fallback"
  warnings while loading.

## Rebuilding the weights

See [conversion/README.md](conversion/README.md). It covers quantizing from the FP8 release with
ExLlamaV3 (the patch that adds this architecture is included) and rewriting the pack config for
vLLM.

## Repository layout

```
config/cluster.env.example   all site-specific settings (IPs, interfaces, paths, serving profile)
config/chat_template.jinja   NaiveAI's chat template with the two fixes described above
scripts/                     install, download, start/stop/status, serve (the recipe), smoke test,
                             test_chat.py (reasoning/tools checks), bench.py (performance)
plugin/                      vLLM model plugin (vllm_naive_n05), installed by install.sh
conversion/                  ExLlamaV3 patch for this architecture + pack-config tool + notes
requirements.lock.txt        pinned Python packages (vLLM 0.29.0 / torch 2.13.0+cu130 set)
results/reference/           raw benchmark JSON behind the tables above
```

## Credits

- **Model:** NaiveAI, [Naive-N0.5-Flash](https://huggingface.co/NaiveAI/Naive-N0.5-Flash), its
  DSpark drafter and chat template (MIT).
- **Kernels and quantization:** [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3),
  through the [vcruz305](https://github.com/vcruz305/exllamav3) fork with fractional-K kernels.
- **vLLM EXL3 plugin:** [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3). It builds on
  Mia's AI Lab's
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
  routed-expert work.
- **Engine:** [vLLM](https://github.com/vllm-project/vllm).

## License

The recipe's own code (scripts, plugin, patch, tools) is MIT; see `LICENSE`. Upstream components
keep their own licenses, and this repo does not redistribute them; `install.sh` fetches them:
- vLLM: Apache-2.0.
- exllamav3: MIT.
- vllm-exl3: **AGPL-3.0**, with parts under Apache-2.0 (see its `LICENSE` and
  `THIRD_PARTY_NOTICES.md`). Keep this in mind if you expose the server to others.
- Model and drafter weights: MIT (NaiveAI).
