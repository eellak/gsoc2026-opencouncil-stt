#!/usr/bin/env bash
# Prepare an already-created pod, stage the bundle, and hand off the unattended run.
#
# Three staging attempts died before this rewrite, each differently: a caller timeout, a
# stray file inside the verified bundle, and a broken SSH pipe during pip install. The
# common cause was fragility, not the pod: five separate SSH sessions, any of which
# tearing the pod down through the trap. So setup is now ONE ssh session fed from stdin,
# and the trap stops killing the pod once the bundle is verified on the volume - past
# that point a dropped connection means reconnect, not burn the work down.
set -euo pipefail

POD_ID="${1:?pod id required}"
staged=0

cleanup() {
  rc=$?
  if [[ "$rc" != 0 && "$staged" == 0 ]]; then
    runpodctl remove pod "$POD_ID" || true
    sleep 5
    k="$(sed -n "s/.*apikey *= *'\([^']*\)'.*/\1/p" "$HOME/.runpod/config.toml")"
    if curl -fsS --max-time 30 -X POST "https://api.runpod.io/graphql?api_key=$k" \
         -H 'Content-Type: application/json' \
         -d '{"query":"query{myself{pods{id}}}"}' 2>/dev/null | grep -q "$POD_ID"; then
      echo "ALERT: pod $POD_ID still listed after remove - it is BILLING. Terminate by hand." >&2
    fi
  elif [[ "$rc" != 0 ]]; then
    echo "staging done, handoff failed (rc=$rc): pod $POD_ID LEFT RUNNING on purpose." >&2
    echo "re-run this script to retry; the hard deadline watchdog still owns the pod." >&2
  fi
}
trap cleanup EXIT

# Everything that can fail now happens with the trap already installed: a missing
# archive or an unwritable cache used to abort with the pod created and nothing
# watching it.
BUNDLE_ARCHIVE="${2:?bundle archive required}"
HOURS="${HARD_DEADLINE_HOURS:-8}"
CACHE="$HOME/.cache/oc-public/clean-pack-launch-$POD_ID"
mkdir -p "$CACHE"

setsid nohup scripts/pod_hard_deadline.sh "$POD_ID" "$HOURS" clean-pack-screen \
  > "$CACHE/hard-deadline.log" 2>&1 &
sleep 1
pgrep -af "pod_hard_deadline.sh $POD_ID" >/dev/null || {
  echo "FATAL: hard deadline did not arm" >&2; exit 2;
}

key="$(sed -n "s/.*apikey *= *'\([^']*\)'.*/\1/p" "$HOME/.runpod/config.toml")"
[[ -n "$key" ]] || { echo "FATAL: no RunPod API key" >&2; exit 2; }

host=""; port=""
for _ in $(seq 1 90); do
  runtime="$(curl -fsS --connect-timeout 15 --max-time 30 -X POST "https://api.runpod.io/graphql?api_key=$key" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"query{pod(input:{podId:\\\"$POD_ID\\\"}){runtime{ports{ip publicPort privatePort type isIpPublic}}}}\"}" || true)"
  host="$(jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | .ip' <<<"$runtime" 2>/dev/null | head -1 || true)"
  port="$(jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | .publicPort' <<<"$runtime" 2>/dev/null | head -1 || true)"
  [[ -n "$host" && -n "$port" ]] && break
  sleep 10
done
[[ -n "$host" && -n "$port" ]] || { echo "FATAL: SSH endpoint unavailable" >&2; exit 3; }
printf '%s %s\n' "$host" "$port" > "$CACHE/ssh-endpoint"

# ServerAliveInterval is what a 20-minute pip install needs; without it the session is
# idle from the client's point of view and the far end hangs up.
ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=25 -o ServerAliveInterval=20 -o ServerAliveCountMax=15)

ssh "${ssh_opts[@]}" -p "$port" "root@$host" \
  'apt-get update -qq && apt-get install -y -qq ffmpeg rsync zstd'
scripts/stage_pod_bundle.sh "$POD_ID" "$host" "$port" "$BUNDLE_ARCHIVE"
staged=1
BUNDLE="$(ssh "${ssh_opts[@]}" -p "$port" "root@$host" 'cat /runpod/oc-bundles/CURRENT_BUNDLE')"
[[ -n "$BUNDLE" ]] || { echo "FATAL: CURRENT_BUNDLE empty" >&2; exit 4; }

# The driver lands OUTSIDE the bundle: stage_pod_bundle.sh verifies the bundle's file set
# on every run, and an extra file inside it fails that check.
scp "${ssh_opts[@]}" -P "$port" notebooks/run_clean_pack_screen.sh \
  "root@$host:/runpod/run_clean_pack_screen.sh"

# One session for the rest. Idempotent: pip is skipped when the marker exists, and the
# driver itself skips any arm that already wrote COMPLETE, so a retry resumes.
ssh "${ssh_opts[@]}" -p "$port" "root@$host" "BUNDLE='$BUNDLE' POD='$POD_ID' KEY='$key' bash -s" <<'REMOTE'
set -euo pipefail
mkdir -p /runpod/runs
# __pycache__ written beside bundle modules would fail the next file-set verification,
# and that failure deletes the pod. Keep bytecode out of the bundle entirely.
export PYTHONDONTWRITEBYTECODE=1
# The image ships /usr/local/cuda/compat (libcuda 575) so it can run on newer hosts.
# On a host whose driver is OLDER than that (seen: 570.195 on an RTX 4090) the loader
# takes the compat lib anyway and every CUDA call dies with error 804, "forward
# compatibility was attempted on non supported HW". torch then reports no GPU at all.
# Moving compat aside makes the loader use the host driver, which is the one that works.
if [[ -d /usr/local/cuda/compat ]]; then
  python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    || { echo "CUDA unavailable with compat layer; disabling it"; \
         mv /usr/local/cuda/compat /usr/local/cuda/compat.disabled; ldconfig; }
fi
if [[ ! -f /runpod/.deps-installed ]]; then
  pip install --break-system-packages -r "$BUNDLE/code/notebooks/requirements-runpod.txt"
  touch /runpod/.deps-installed
fi
umask 077; printf '%s' "$KEY" > /root/.runpod_api_key
if pgrep -f run_clean_pack_screen.sh >/dev/null; then
  echo "driver already running"
else
  BUNDLE="$BUNDLE" RUNPOD_POD_ID="$POD" PYTHONDONTWRITEBYTECODE=1 \
    nohup bash /runpod/run_clean_pack_screen.sh \
    > /runpod/runs/clean-pack-driver.log 2>&1 < /dev/null &
  disown
  sleep 5
fi
pgrep -f run_clean_pack_screen.sh >/dev/null || { echo "FATAL: driver not running" >&2; exit 5; }
echo "driver up"
REMOTE

printf '%s\n' "handed off pod=$POD_ID ssh=$host:$port bundle=$BUNDLE" | tee "$CACHE/HANDOFF"
