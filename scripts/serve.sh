#!/usr/bin/env bash
# The serving recipe. Runs the vLLM API server on the HEAD node against the running Ray cluster
# (start.sh launches it inside screen; run it directly only for debugging).
source "$(dirname "$0")/common.sh"
node_env "$HEAD_IP"
cd "$RUN_DIR"   # neutral cwd: a local ./exllamav3 directory would shadow the package in the workers

args=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --quantization exl3
  --tensor-parallel-size 2 --enable-expert-parallel      # attention TP2, 128 whole experts per rank
  --distributed-executor-backend ray
  --host "$HOST" --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --dtype bfloat16
  --moe-backend triton
  --no-enable-flashinfer-autotune
  --kernel-config '{"enable_cutedsl_warmup": false, "enable_jit_warmup": false}'
)
# Reasoning (<think>...</think>) and tool calls (<tool_call><function=...><parameter=...>) use the
# Qwen3 / Qwen3-Coder formats; the parsers split them out of the content (see README).
[ -n "${CHAT_TEMPLATE:-}" ] && args+=(--chat-template "$CHAT_TEMPLATE")
[ -n "${REASONING_PARSER:-}" ] && args+=(--reasoning-parser "$REASONING_PARSER")
if [ -n "${TOOL_PARSER:-}" ]; then
  args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
  # naive_n05 = Qwen3-Coder parsing + a grammar that fires on <tool_call> (plugin/.../tool_parser.py)
  [ "$TOOL_PARSER" = naive_n05 ] && args+=(--tool-parser-plugin "$REPO/plugin/vllm_naive_n05/tool_parser.py")
fi
if [ "$SPEC_TOKENS" != 0 ]; then
  args+=(--speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFT_DIR\",\"num_speculative_tokens\":$SPEC_TOKENS}")
fi
echo "### $(date -Is) vllm serve ${args[*]}"
exec vllm serve "${args[@]}"
