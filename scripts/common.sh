# Sourced by the other scripts: loads config/cluster.env and defines helpers.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${CLUSTER_ENV:-$REPO/config/cluster.env}"
[ -f "$CFG" ] || { echo "error: $CFG missing (cp config/cluster.env.example config/cluster.env)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CFG"

# Pinned upstream revisions (see README "Components").
VLLM_EXL3_REPO=https://github.com/vcruz305/vllm-exl3
VLLM_EXL3_REV=08ed1bf
EXLLAMAV3_REPO=https://github.com/vcruz305/exllamav3
EXLLAMAV3_REV=07572bd
TORCH_INDEX=https://download.pytorch.org/whl/cu130
TORCH_CUDA_ARCH_LIST=12.1a

SCREEN_NAME=naive_n05
SERVE_LOG="$RUN_DIR/serve.log"

say() { echo "==> $*"; }
die() { echo "error: $*" >&2; exit 1; }
is_local() { hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "$1"; }
# run_on <ip> <command...>: local shell if <ip> is this host, ssh otherwise
run_on() {
  local ip=$1; shift
  if is_local "$ip"; then bash -lc "$*"; else ssh $SSH_OPTS "$ip" "$*"; fi
}

# Runtime environment for Ray, NCCL and vLLM. Called with this node's IP.
node_env() {
  local node_ip=$1
  export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
  export CUDA_HOME
  export RAY_TMPDIR=/dev/shm/ray
  export RAY_memory_monitor_refresh_ms=0
  export RAY_worker_register_timeout_seconds=120
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_SOCKET_IFNAME=$NET_IF GLOO_SOCKET_IFNAME=$NET_IF TP_SOCKET_IFNAME=$NET_IF
  export NCCL_IB_HCA NCCL_IB_GID_INDEX
  export VLLM_HOST_IP=$node_ip
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  export VLLM_DEEP_GEMM_WARMUP=skip
  export VLLM_EXL3_MOE_KERNEL=exllamav3
  export TRITON_CACHE_DIR=$RUN_DIR/triton VLLM_CACHE_ROOT=$RUN_DIR/vllm
  export RAY_ADDRESS="$HEAD_IP:6379"
  [ -n "${EXTRA_CPATH:-}" ] && export CPATH="${CPATH:+$CPATH:}$EXTRA_CPATH"
  mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
}
