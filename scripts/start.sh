#!/usr/bin/env bash
# Bring up Ray on both Sparks (if needed) and the API server on the head, then wait until ready.
#   bash scripts/start.sh          # from either node
source "$(dirname "$0")/common.sh"

url="http://$HEAD_IP:$PORT"
ready() { curl -s -m 5 "$url/v1/models" 2>/dev/null | grep -q '"object":"list"'; }
ray_ok() { run_on "$HEAD_IP" "source '$REPO/scripts/common.sh'; node_env $HEAD_IP; ray status 2>/dev/null" \
  | grep -qE '/2\.0 GPU'; }   # "used/total": an idle healthy cluster shows 0.0/2.0

run_on "$HEAD_IP" true || die "cannot reach head $HEAD_IP"
run_on "$WORKER_IP" true || die "cannot reach worker $WORKER_IP"
run_on "$WORKER_IP" "test -f '$REPO/scripts/common.sh'" || die "repo not found at $REPO on the worker"

if ready; then say "server already running at $url"; exit 0; fi

if ! ray_ok; then
  say "starting Ray (head $HEAD_IP, worker $WORKER_IP)"
  run_on "$HEAD_IP" "bash '$REPO/scripts/ray_node.sh' head" >/dev/null
  run_on "$WORKER_IP" "bash '$REPO/scripts/ray_node.sh' worker" >/dev/null
  for _ in $(seq 1 24); do ray_ok && break; sleep 5; done
  ray_ok || die "Ray cluster did not reach 2 GPUs"
fi

say "launching the API server (log: $HEAD_IP:$SERVE_LOG)"
run_on "$HEAD_IP" "mkdir -p '$RUN_DIR'; screen -S $SCREEN_NAME -X quit >/dev/null 2>&1; \
  screen -dmS $SCREEN_NAME bash -c \"bash '$REPO/scripts/serve.sh' > '$SERVE_LOG' 2>&1\""

say "waiting for readiness (weights load ~6.5 min, compile + graph capture ~1 min)"
for _ in $(seq 1 80); do
  sleep 15
  if ready; then
    say "READY  $url  (model id: $SERVED_NAME)"
    run_on "$HEAD_IP" "grep -oE 'GPU KV cache size: [0-9,]+ tokens' '$SERVE_LOG' | tail -1"
    exit 0
  fi
  if ! run_on "$HEAD_IP" "screen -ls | grep -q $SCREEN_NAME"; then
    run_on "$HEAD_IP" "grep -vE 'staging fallback' '$SERVE_LOG' | tail -30"
    die "server exited during startup (full log: $HEAD_IP:$SERVE_LOG)"
  fi
done
die "timed out waiting for $url"
