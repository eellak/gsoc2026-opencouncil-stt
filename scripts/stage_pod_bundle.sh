#!/usr/bin/env bash
# Resumably stage a verified training bundle onto a pod/network volume.
# Usage: stage_pod_bundle.sh POD_ID HOST SSH_PORT ARCHIVE [REMOTE_ROOT]
set -euo pipefail

POD_ID="${1:?pod id required}"
HOST="${2:?pod host required}"
PORT="${3:?ssh port required}"
ARCHIVE="${4:?bundle .tar.zst required}"
REMOTE_ROOT="${5:-/runpod/oc-bundles}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25)

[[ -f "$ARCHIVE" && -f "$ARCHIVE.sha256" ]] || {
  echo "FATAL: archive and $ARCHIVE.sha256 are required" >&2
  exit 2
}
pgrep -af "pod_hard_deadline.sh $POD_ID" >/dev/null || {
  echo "FATAL: no local hard-deadline watchdog found for pod $POD_ID" >&2
  exit 2
}

NAME="$(basename "$ARCHIVE")"
DIR="${NAME%.tar.zst}"
ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
  "command -v rsync >/dev/null && command -v zstd >/dev/null && mkdir -p '$REMOTE_ROOT/incoming'" || {
    echo "FATAL: pod needs rsync and zstd before staging" >&2
    exit 2
  }

rsync --partial --append-verify --human-readable --progress \
  -e "ssh ${SSH_OPTS[*]} -p $PORT" \
  "$ARCHIVE" "$ARCHIVE.sha256" "root@$HOST:$REMOTE_ROOT/incoming/"

ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "
  set -e
  cd '$REMOTE_ROOT/incoming'
  sha256sum -c '$NAME.sha256'
  mkdir -p '$REMOTE_ROOT'
  tar -I zstd -xf '$NAME' -C '$REMOTE_ROOT'
  python3 '$REMOTE_ROOT/$DIR/code/scripts/pod_bundle.py' verify '$REMOTE_ROOT/$DIR'
  printf '%s\n' '$REMOTE_ROOT/$DIR' > '$REMOTE_ROOT/CURRENT_BUNDLE'
"

echo "ready: $REMOTE_ROOT/$DIR"
