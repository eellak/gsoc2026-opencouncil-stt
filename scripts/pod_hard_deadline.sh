#!/bin/bash
# Terminate a RunPod pod at a fixed wall-clock deadline, no matter what.
#
# CLAUDE.md: "A GPU pod bills from creation. Arm a watchdog with a hard deadline
# BEFORE uploading anything." This is that watchdog, and it is deliberately the
# stupidest possible one: it does not check whether the run is healthy, whether
# training is still going, or whether anyone is watching. It sleeps and then kills.
#
# scripts/train_watchdog.sh is the *other* kind - it syncs checkpoints and notices
# crashes. That one can be fooled by a frozen pod or a dead session. This one cannot,
# which is why it is armed first and separately.
#
# Usage: pod_hard_deadline.sh <pod_id> <hours> [reason]
#   nohup scripts/pod_hard_deadline.sh ul4z0drp5owiac 10 "correction-only" &
set -u
POD="$1"; HOURS="$2"; REASON="${3:-unspecified}"
KEY=$(grep apikey ~/.runpod/config.toml | sed "s/.*'\(.*\)'.*/\1/")
[ -n "$KEY" ] || { echo "FATAL: no RunPod api key in ~/.runpod/config.toml"; exit 2; }

DEADLINE=$(( $(date +%s) + $(printf '%.0f' "$(echo "$HOURS * 3600" | bc)") ))
echo "armed: pod $POD will be terminated at $(date -d @$DEADLINE -Is) ($REASON)"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 60
  # Already gone? Then there is nothing to guard and idling costs nothing.
  if ! runpodctl get pod "$POD" >/dev/null 2>&1; then
    echo "$(date -Is) pod $POD no longer exists - watchdog standing down"
    exit 0
  fi
done

echo "$(date -Is) DEADLINE REACHED - terminating $POD"
runpodctl remove pod "$POD" || \
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"$POD\\\"})}\"}"
sleep 20
runpodctl get pod "$POD" >/dev/null 2>&1 \
  && echo "!! pod $POD STILL ALIVE AFTER TERMINATE - STOP IT BY HAND" \
  || echo "$(date -Is) pod $POD terminated"
