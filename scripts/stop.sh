#!/usr/bin/env bash
# Stop the API server; with --all also stop Ray on both nodes (frees the most memory).
#   bash scripts/stop.sh [--all]
source "$(dirname "$0")/common.sh"
run_on "$HEAD_IP" "screen -S $SCREEN_NAME -X quit >/dev/null 2>&1; pkill -f '[v]llm serve' >/dev/null 2>&1; true"
say "server stopped"
if [ "${1:-}" = --all ]; then
  for ip in "$HEAD_IP" "$WORKER_IP"; do
    run_on "$ip" "source '$REPO/scripts/common.sh'; node_env $ip; ray stop --force >/dev/null 2>&1; true"
  done
  say "Ray stopped on both nodes"
fi
