#!/usr/bin/env bash
# Turn a bare rented pod into an enrolled FlashML worker.
#
# A pod has no browser, so the interactive device-code flow is useless to it.
# The token is minted by the owner BEFORE the pod exists and handed in via the
# environment; this script only has to place two files and start the agent.
#
# `--runner trusted`, not `argv`: a pod is itself a container and cannot run
# Docker-in-Docker, so it executes argv directly and installs the job's
# declared dependencies. That tier is only eligible for POOL jobs
# (`allowFallback` iff `pool`), which is why the machine is bound to a pool at
# enrolment time.
set -euo pipefail

: "${FLASHML_NODE_ID:?FLASHML_NODE_ID is required}"
: "${FLASHML_TOKEN:?FLASHML_TOKEN is required}"
: "${FLASHML_API:?FLASHML_API is required}"

API="${FLASHML_API%/}"
STATE="${FLASHNODE_STATE_DIR:-$HOME/.flashnode}"
mkdir -p "$STATE"

# The node_id is SEEDED, not generated: the machine row was approved against
# this exact id, and a self-generated one would present a token minted for an
# identity nobody approved.
printf '%s' "$FLASHML_NODE_ID" > "$STATE/node-id"

# credentials.json is keyed by normalised coordinator URL.
printf '{"%s": "%s"}' "$API" "$FLASHML_TOKEN" > "$STATE/credentials.json"
chmod 600 "$STATE/credentials.json"

python3 -m pip install --no-cache-dir --quiet "flashnode==0.4.0"

echo "flashnode $(flashnode --help 2>&1 | head -1)"
echo "node ${FLASHML_NODE_ID} -> ${API}"
exec flashnode work --runner trusted --coordinator "$API"
