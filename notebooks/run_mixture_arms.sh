#!/usr/bin/env bash
# Seven training runs for the mixture-ratio experiment, on one pod, sequentially.
# Spec: docs/specs/mixture-ratio-preregistration.md
#
# Runs unattended. A review of the first version found four ways it could burn a paid
# pod overnight and produce nothing, or worse, produce the wrong arm quietly:
#   - it exited after building clips instead of continuing
#   - it never passed --nb2-ids, so arm C silently became arm C1
#   - it never ran C1 at all
#   - it left the pod billing on every terminal path
# All four are fixed below. Do not "simplify" them back out.
#
# DATA_DIR must contain, under these exact names:
#   train.parquet           the EXPANDED superset (this is what gets cut into clips)
#   train_control.parquet   the ORIGINAL train parquet (arm A is exactly its rows)
#   train_expanded.parquet  a copy of the superset, for mix_arms
#   validation.parquet      unchanged
#   nb2_ids.json            algorithm-selected correction ids
#
# Usage on the pod:
#   bash run_mixture_arms.sh 2>&1 | tee /workspace/mixture.log
set -euo pipefail

WORK=${WORK_DIR:-/workspace/whisper-run}
OC=${OC_DIR:-/workspace/oc}
DATA=${DATA_DIR:-$OC/data}
export SMOKE=0 MAX_STEPS=7242 WORK_DIR="$WORK" DATA_DIR="$DATA"

cd "$OC"
mkdir -p "$WORK"

for f in train.parquet train_control.parquet train_expanded.parquet \
         validation.parquet nb2_ids.json; do
  [ -f "$DATA/$f" ] || { echo "FATAL: missing $DATA/$f"; exit 2; }
done

# Stop the pod on every exit path, including failure and interrupt. A run that dies at
# 02:00 must not bill until someone notices. RUNPOD_POD_ID is set inside the pod.
finish() {
  rc=$?
  echo "== exiting with status $rc"
  # NB: this fires on SIGTERM too, so `pkill -f run_mixture_arms` terminates the POD,
  # not just the run. That is correct for an unattended failure and surprising when
  # you are only trying to stop the script — export KEEP_POD=1 first, or kill the
  # python child instead of the driver.
  [ "${KEEP_POD:-0}" = "0" ] || { echo "== KEEP_POD set, leaving pod up"; return; }
  [ -n "${RUNPOD_POD_ID:-}" ] || { echo "!! RUNPOD_POD_ID unset — STOP THE POD BY HAND"; return; }

  # A FAILED run keeps the pod alive. The third pod died mid-build for a reason that
  # is now unknowable, because the log lived on the machine the trap destroyed —
  # terminating on failure buys a few cents of idle billing and costs the diagnosis.
  # A watchdog bounds the exposure: if nothing is training three hours from now, the
  # pod goes regardless, so a dead session cannot leave it billing forever.
  if [ "$rc" -ne 0 ]; then
    echo "== FAILED (status $rc) — leaving the pod up so the log can be read."
    echo "== watchdog: terminating in 3h unless training resumes."
    setsid nohup bash -c '
      sleep 10800
      pgrep -f train_runpod >/dev/null && exit 0
      curl -s -X POST "https://api.runpod.io/graphql?api_key='"$RUNPOD_API_KEY"'" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"'"$RUNPOD_POD_ID"'\\\"})}\"}"
    ' >/dev/null 2>&1 < /dev/null &
    return
  fi
  # The image ships neither runpodctl nor RUNPOD_POD_ID, so both are supplied by the
  # launcher and the API call is the fallback that actually works. A pod that outlives
  # its run bills silently until someone notices, which is the expensive failure here.
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

# 1. Build every clip once, from the superset. Both arms are lists over these files.
if [ ! -f "$WORK/manifest.json" ]; then
  echo "== building clips from the superset (~2h)"
  BUILD_AND_EXIT=1 SEED=13 python notebooks/train_runpod.py
  [ -f "$WORK/manifest.json" ] || { echo "FATAL: build produced no manifest"; exit 3; }
fi

# 2. Derive the arms. This hard-fails if any arm drifts outside the preregistered
#    tolerances on duration, correction share, or city mixture.
if [ ! -f "$WORK/arm_report.json" ]; then
  echo "== deriving arm manifests"
  python notebooks/mix_arms.py --work "$WORK" \
    --parquet "$DATA/train_expanded.parquet" \
    --control "$DATA/train_control.parquet" \
    --nb2-ids "$DATA/nb2_ids.json"
fi

# 3. The seven runs. C1 is the one-seed diagnostic and only exists at the first seed.
run_one() {
  TAG=$1; MAN=$2; SEED=$3
  if [ -f "$WORK/adapter_$TAG/COMPLETE" ]; then echo "== skip $TAG (complete)"; return 0; fi
  [ -f "$MAN" ] || { echo "FATAL: $MAN missing"; exit 4; }
  # A previous attempt may have died after moving the adapter but before marking it.
  rm -rf "$WORK/adapter" "$WORK/adapter_$TAG.partial"
  echo "== run $TAG  manifest=$(basename "$MAN")  seed=$SEED  steps=$MAX_STEPS"
  SEED=$SEED TRAIN_MANIFEST="$MAN" python notebooks/train_runpod.py
  mv "$WORK/adapter" "$WORK/adapter_$TAG.partial"
  # COMPLETE is written last and is the only thing the skip check trusts, so a crash
  # anywhere earlier leaves the run repeatable instead of half-recorded.
  mv "$WORK/adapter_$TAG.partial" "$WORK/adapter_$TAG"
  date -Is > "$WORK/adapter_$TAG/COMPLETE"
}

for SEED in 13 29 47; do
  run_one "A_s$SEED"  "$WORK/manifest_A.json"       "$SEED"
  run_one "C_s$SEED"  "$WORK/manifest_C_s$SEED.json" "$SEED"
  if [ "$SEED" = 13 ]; then
    run_one "C1_s13" "$WORK/manifest_C1_s13.json" 13
  fi
done

echo "== all seven runs complete; adapters in $WORK/adapter_*"
echo "== next: transcribe the 32 dev windows with each adapter, then score locally:"
echo "   python -m eval.controlled_eval.score_audio_faithful --paired \\"
echo "     --pairs A_s13:C_s13 A_s29:C_s29 A_s47:C_s47 --dir hyps/"
