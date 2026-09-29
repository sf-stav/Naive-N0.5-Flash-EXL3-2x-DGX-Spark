#!/usr/bin/env bash
# Server status, memory headroom on both nodes, and the latest log lines.
#   bash scripts/status.sh [-f]      # -f follows the log
source "$(dirname "$0")/common.sh"
if curl -s -m 5 "http://$HEAD_IP:$PORT/v1/models" | grep -q '"object":"list"'; then
  say "server UP at http://$HEAD_IP:$PORT (model id: $SERVED_NAME)"
else
  say "server DOWN"
fi
for ip in "$HEAD_IP" "$WORKER_IP"; do
  echo "  $ip: $(run_on "$ip" "free -g | awk '/Mem:/{print \$7\" GiB available of \"\$2}'")"
done
if [ "${1:-}" = -f ]; then
  run_on "$HEAD_IP" "tail -f '$SERVE_LOG' | grep --line-buffered -v 'staging fallback'"
else
  run_on "$HEAD_IP" "grep -E 'Avg generation throughput|Mean acceptance length|Error|Traceback' '$SERVE_LOG' | tail -6"
fi
