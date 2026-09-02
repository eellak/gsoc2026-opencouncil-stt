#!/usr/bin/env bash
# Prepare an already-created pod, stage the bundle, and hand off the unattended run.
set -euo pipefail

POD_ID="${1:?pod id required}"
BUNDLE_ARCHIVE="${2:?bundle archive required}"
HOURS="${HARD_DEADLINE_HOURS:-18}"
CACHE="$HOME/.cache/oc-public/dense-screen-launch-$POD_ID"
mkdir -p "$CACHE"
handed_off=0

cleanup() {
  rc=$?
  if [[ "$rc" != 0 && "$handed_off" == 0 ]]; then
    runpodctl remove pod "$POD_ID" || true
  fi
}
trap cleanup EXIT

nohup scripts/pod_hard_deadline.sh "$POD_ID" "$HOURS" dense-screen \
  > "$CACHE/hard-deadline.log" 2>&1 &
sleep 1
pgrep -af "pod_hard_deadline.sh $POD_ID" >/dev/null || {
  echo "FATAL: hard deadline did not arm" >&2; exit 2;
}

key="$(sed -n "s/.*apikey *= *'\([^']*\)'.*/\1/p" "$HOME/.runpod/config.toml")"
[[ -n "$key" ]] || { echo "FATAL: no RunPod API key" >&2; exit 2; }

host=""; port=""
for _ in $(seq 1 60); do
  runtime="$(curl -fsS -X POST "https://api.runpod.io/graphql?api_key=$key" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"query{pod(input:{podId:\\\"$POD_ID\\\"}){runtime{ports{ip publicPort privatePort type isIpPublic}}}}\"}")"
  host="$(jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | .ip' <<<"$runtime" | head -1)"
  port="$(jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | .publicPort' <<<"$runtime" | head -1)"
  [[ -n "$host" && -n "$port" ]] && break
  sleep 10
done
[[ -n "$host" && -n "$port" ]] || { echo "FATAL: SSH endpoint unavailable" >&2; exit 3; }
printf '%s %s\n' "$host" "$port" > "$CACHE/ssh-endpoint"

ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25)
ssh "${ssh_opts[@]}" -p "$port" "root@$host" \
  'apt-get update -qq && apt-get install -y -qq ffmpeg rsync zstd'
scripts/stage_pod_bundle.sh "$POD_ID" "$host" "$port" "$BUNDLE_ARCHIVE"
BUNDLE="$(ssh "${ssh_opts[@]}" -p "$port" "root@$host" 'cat /runpod/oc-bundles/CURRENT_BUNDLE')"
ssh "${ssh_opts[@]}" -p "$port" "root@$host" \
  "pip install --break-system-packages -r '$BUNDLE/code/notebooks/requirements-runpod.txt'"
printf '%s' "$key" | ssh "${ssh_opts[@]}" -p "$port" "root@$host" \
  'umask 077; dd of=/root/.runpod_api_key status=none'
ssh "${ssh_opts[@]}" -p "$port" "root@$host" \
  "mkdir -p /runpod/runs && BUNDLE='$BUNDLE' RUNPOD_POD_ID='$POD_ID' nohup bash '$BUNDLE/code/notebooks/run_dense_screen.sh' > /runpod/runs/dense-screen-driver.log 2>&1 < /dev/null &"
handed_off=1
printf '%s\n' "handed off pod=$POD_ID ssh=$host:$port bundle=$BUNDLE" | tee "$CACHE/HANDOFF"
