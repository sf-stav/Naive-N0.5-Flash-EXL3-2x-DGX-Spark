#!/usr/bin/env bash
# One short chat request against the running server.
#   bash scripts/smoke.sh
source "$(dirname "$0")/common.sh"
curl -s "http://$HEAD_IP:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED_NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17*19? Reply with the integer only.\"}],
  \"max_tokens\": 16, \"temperature\": 0,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(repr(d["choices"][0]["message"]["content"].strip()), d["usage"])'
