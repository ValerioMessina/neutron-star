#!/bin/sh
set -eu

base="${NEUTRON_BASE_URL:-http://127.0.0.1:8080}"
auth=""
if [ -n "${NEUTRON_API_KEY:-}" ]; then auth="Authorization: Bearer ${NEUTRON_API_KEY}"; fi

curl -fsS "$base/healthz"
curl -fsS "$base/v1/models" ${auth:+-H "$auth"}
curl -fsS "$base/v1/chat/completions" ${auth:+-H "$auth"} \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b","messages":[{"role":"user","content":"Reply with exactly: pong"}],"temperature":0,"max_tokens":16}'
