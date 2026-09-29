#!/usr/bin/env bash
# Start the Ray head or worker on THIS node (start.sh calls it on both nodes over ssh).
#   bash scripts/ray_node.sh head|worker
source "$(dirname "$0")/common.sh"
role=${1:?usage: ray_node.sh head|worker}
case "$role" in
  head)   node_env "$HEAD_IP";   extra=(--head --port=6379 --node-ip-address="$HEAD_IP") ;;
  worker) node_env "$WORKER_IP"; extra=(--address="$HEAD_IP:6379" --node-ip-address="$WORKER_IP") ;;
  *) die "role must be head or worker" ;;
esac
ray stop --force >/dev/null 2>&1 || true
exec ray start "${extra[@]}" --temp-dir="$RAY_TMPDIR" --num-gpus=1 \
  --object-store-memory=1073741824 --disable-usage-stats
