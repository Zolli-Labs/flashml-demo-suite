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

# A VENV, not the system python. The rented image installs `cryptography`
# from Debian packages with no RECORD file, so pip cannot uninstall it to
# satisfy flashnode's dependency and exits — which, under `set -e`, crash-loops
# the container every ~17s. An isolated venv never touches the distro's copy.
# `--system-site-packages` so the image's CUDA-linked torch stays visible.
VENV=/opt/flashnode-venv
if [ ! -x "$VENV/bin/flashnode" ]; then
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install --no-cache-dir --quiet --upgrade pip
  "$VENV/bin/pip" install --no-cache-dir --quiet "flashnode==0.4.0"
fi

echo "node ${FLASHML_NODE_ID} -> ${API}"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
exec "$VENV/bin/flashnode" work --runner trusted --coordinator "$API"
