#!/bin/bash
# Arm B at a 2-epoch budget, checkpointed. No arm A: the comparison baseline is the
# published incumbent, re-scored on this same stack. Steps are matched across datasets
# (619) so the contiguous-vs-spliced contrast changes data and nothing else.
set -o pipefail
B=/workspace/oc/bundle
ROOT=/workspace/oc/runs
STEPS=619
SAVE=155
NTFY=https://ntfy.sh/MvtrfeSLN3jvt50N
say(){ curl -s -X POST -H "Title: 🤖 OpenCouncil" -H "Priority: 3" -d "$1" "$NTFY" >/dev/null 2>&1; }
mkdir -p "$ROOT"

train(){  # $1=tag  $2=manifest  $3=seed
  local out="$ROOT/$1_s$3"
  [[ -f "$out/COMPLETE" ]] && { echo "skip $1_s$3"; return 0; }
  mkdir -p "$out"
  local log="/workspace/$1_s$3.log"
  local -a st
  set +e
  env SMOKE=0 SEED="$3" MAX_STEPS="$STEPS" SAVE_STEPS="$SAVE" EVAL_STRATEGY=no \
      TRAIN_BS=1 GRAD_ACC=8 WORK_DIR="$out" HF_HOME=/workspace/hf-cache \
      PACK_MANIFEST="$2" PACK_ARM=pn \
      PYTHONPATH="$B/code" python "$B/code/notebooks/train_runpod.py" 2>&1 | tee "$log"
  st=("${PIPESTATUS[@]}")
  set -e
  cp -f "$log" "$out/train.log" 2>/dev/null || true
  if [[ "${st[0]}" != 0 ]]; then
    echo "FATAL: $1_s$3 exited ${st[0]}" >&2; say "❌ $1_s$3 απέτυχε (rc=${st[0]})"; return 1
  fi
  [[ -s "$out/adapter/adapter_model.safetensors" ]] || { echo "FATAL: no adapter for $1_s$3" >&2; say "❌ $1_s$3 χωρίς adapter"; return 1; }
  date -Is > "$out/COMPLETE"
  local n=$(ls -d "$out"/adapter/checkpoint-* 2>/dev/null | wc -l)
  echo "done $1_s$3 ($n checkpoints)"
  say "✅ $1_s$3 τελείωσε — $n checkpoints"
}

say "🚀 Ξεκίνησε arm B, 619 βήματα (2 epochs), 3 seeds contiguous"
for s in 13 29 47; do
  train cont "$B/packs/manifest.jsonl" "$s" || exit 1
done
date -Is > "$ROOT/CONT_COMPLETE"
say "🏁 Και τα 3 contiguous seeds τελείωσαν"

# combined arm, only if the spliced set finished staging
if [[ -f /workspace/oc/combined/manifest.jsonl ]]; then
  for s in 13 29 47; do
    train comb /workspace/oc/combined/manifest.jsonl "$s" || exit 1
  done
  date -Is > "$ROOT/COMB_COMPLETE"
  say "🏁 Και τα 3 combined seeds τελείωσαν"
else
  say "⚠️ combined manifest λείπει — μόνο contiguous έτρεξε"
fi
date -Is > "$ROOT/ALL_COMPLETE"
