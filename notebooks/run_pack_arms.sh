#!/usr/bin/env bash
# Four runs: packed training with per-utterance timestamps (P) and without (Pn), two seeds.
# docs/runbooks/2026-08-10-unattended-packing-run.md
#
# Arm A is NOT retrained. The corrected 2026-08-02 adapter already exists and is the frozen
# reference; a fresh A run would mostly measure whether the pipeline reproduces itself.
# The money buys a second seed instead, because a one-seed ranking against a measured
# 2.1-point seed spread is noise wearing a number.
#
# Seed order puts both seed-13 arms first on purpose: if anything eats the night, the run
# still leaves one complete P/Pn pair rather than two halves of two comparisons.
set -uo pipefail

WORK="${WORK_DIR:-/workspace/whisper-run}"
PACKS="${PACKS_DIR:-/workspace/packs}"
STEPS="${MAX_STEPS:-2600}"
mkdir -p "$WORK"

[ -f "$PACKS/manifest.jsonl" ] || { echo "FATAL: no $PACKS/manifest.jsonl"; exit 2; }
N_AUDIO=$(ls "$PACKS/audio" | wc -l)
N_MAN=$(wc -l < "$PACKS/manifest.jsonl")
[ "$N_AUDIO" = "$N_MAN" ] || { echo "FATAL: $N_AUDIO audio files vs $N_MAN manifest rows"; exit 2; }
echo "== $N_MAN packs, $STEPS steps per arm"

# Stop the pod on every exit path. A run that dies at 04:00 must not bill until someone
# notices. A local watchdog was armed at pod creation for the paths where this script is
# not alive to run at all.
finish() {
  rc=$?
  echo "== exiting with status $rc"
  # A file, not an env var. `KEEP_POD=1 pkill -f run_pack_arms` sets the variable in
  # pkill's environment and not in the driver's, so the trap never sees it and the pod
  # dies anyway. That happened once and cost a pod plus a 1.5 GB re-upload. `touch
  # /workspace/KEEP_POD` works from any shell, at any time, including after the driver
  # has already started.
  if [ -f /workspace/KEEP_POD ] || [ "${KEEP_POD:-0}" != "0" ]; then
    echo "== KEEP_POD present, leaving pod up"; return
  fi
  [ -n "${RUNPOD_POD_ID:-}" ] || { echo "!! RUNPOD_POD_ID unset — STOP THE POD BY HAND"; return; }
  if [ "$rc" -ne 0 ]; then
    echo "== FAILED (status $rc) — leaving the pod up for 2h so the log survives the diagnosis."
    setsid nohup bash -c '
      sleep 7200
      pgrep -f train_runpod >/dev/null && exit 0
      curl -s -X POST "https://api.runpod.io/graphql?api_key='"${RUNPOD_API_KEY:-}"'" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"'"$RUNPOD_POD_ID"'\\\"})}\"}"
    ' >/dev/null 2>&1 < /dev/null &
    return
  fi
  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl remove pod "$RUNPOD_POD_ID" && return
  fi
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"})}\"}" \
      && echo " <- pod terminate requested" && return
  fi
  echo "!! could not stop pod $RUNPOD_POD_ID — STOP IT BY HAND"
}
trap finish EXIT

for spec in p:13 pn:13 p:37 pn:37; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  OUT="$WORK/adapter_${ARM}_s${SEED}"
  if [ -f "$OUT/COMPLETE" ]; then echo "== $ARM s$SEED already complete"; continue; fi
  echo "== $(date -Is) arm=$ARM seed=$SEED -> $OUT"
  SMOKE=0 SEED="$SEED" MAX_STEPS="$STEPS" EVAL_STRATEGY=no \
    PACK_MANIFEST="$PACKS/manifest.jsonl" PACK_ARM="$ARM" \
    WORK_DIR="$OUT" \
    python notebooks/train_runpod.py 2>&1 | tee "$WORK/${ARM}_s${SEED}.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!! $ARM s$SEED failed with $rc"
    # One arm failing is a result, not a reason to burn the rest of the budget on three
    # more runs of the same broken thing.
    exit "$rc"
  fi
  touch "$OUT/COMPLETE"
  echo "== $(date -Is) $ARM s$SEED done"
done

echo "== all four arms complete"
