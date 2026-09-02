#!/usr/bin/env bash
# Paired 300-step dense-window screen, followed by one-stack GPU decode and scoring.
set -euo pipefail

BUNDLE="${BUNDLE:?set BUNDLE to the extracted bundle root}"
ROOT="${RUN_ROOT:-/runpod/runs/dense-screen}"
STEPS="${MAX_STEPS:-300}"
mkdir -p "$ROOT"

finish() {
  rc=$?
  printf '%s\n' "$rc" > "$ROOT/EXIT_STATUS"
  if [[ "${KEEP_POD:-0}" != 0 ]]; then return; fi
  if [[ -n "${RUNPOD_POD_ID:-}" && -f /root/.runpod_api_key ]]; then
    key="$(cat /root/.runpod_api_key)"
    curl -fsS -X POST "https://api.runpod.io/graphql?api_key=$key" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"})}\"}" || true
  fi
}
trap finish EXIT

run_one() {
  arm="$1"; seed="$2"; out="$ROOT/${arm}_s${seed}"
  [[ -f "$out/COMPLETE" ]] && { echo "skip ${arm}_s${seed}"; return; }
  mkdir -p "$out"
  common=(SMOKE=0 SEED="$seed" MAX_STEPS="$STEPS" EVAL_STRATEGY=no SAVE_STEPS="$STEPS"
          TRAIN_BS=1 GRAD_ACC=8 WORK_DIR="$out" HF_HOME=/runpod/hf-cache)
  if [[ "$arm" == A ]]; then
    env "${common[@]}" DATA_DIR="$BUNDLE/data/public" \
      PREP_CACHE_DIR=/runpod/prep/control FEATURE_CACHE_DIR=/runpod/features/control \
      python "$BUNDLE/code/notebooks/train_runpod.py" 2>&1 | tee "$out/train.log"
  else
    env "${common[@]}" PACK_MANIFEST="$BUNDLE/packs/manifest.jsonl" PACK_ARM=pn \
      PREP_CACHE_DIR=/runpod/prep/dense FEATURE_CACHE_DIR=/runpod/features/dense \
      python "$BUNDLE/code/notebooks/train_runpod.py" 2>&1 | tee "$out/train.log"
  fi
  [[ "${PIPESTATUS[0]}" == 0 ]] || exit 1
  date -Is > "$out/COMPLETE"
}

for seed in 13 29 47; do
  run_one A "$seed"
  run_one B "$seed"
done

PYTHONPATH="$BUNDLE/code" HF_HOME=/runpod/hf-cache python -m \
  eval.controlled_eval.dense_screen_eval all \
  --runs "$ROOT" \
  --validation "$BUNDLE/eval/validation/manifest.json" \
  --training "$BUNDLE/eval/training/sample.json" \
  --out "$ROOT/hypotheses" --ct2 "$ROOT/ct2-work" \
  --result "$ROOT/results.json" 2>&1 | tee "$ROOT/eval.log"
date -Is > "$ROOT/ALL_COMPLETE"
