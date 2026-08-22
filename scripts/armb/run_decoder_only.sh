#!/bin/bash
# Fallback / third arm: identical to the contiguous runs in every respect except that
# LoRA is restricted to the decoder. The control is already on disk, so this is a
# genuine one-variable comparison - the cleanest this project has managed.
set -o pipefail
B=/workspace/oc/bundle
ROOT=/workspace/oc/runs
NTFY=https://ntfy.sh/MvtrfeSLN3jvt50N
say(){ curl -s -X POST -H "Title: 🤖 OpenCouncil" -H "Priority: 3" -d "$1" "$NTFY" >/dev/null 2>&1; }
for s in 13 29 47; do
  out="$ROOT/dec_s$s"
  [[ -f "$out/COMPLETE" ]] && continue
  mkdir -p "$out"; log="/workspace/dec_s$s.log"
  set +e
  env SMOKE=0 SEED="$s" MAX_STEPS=619 SAVE_STEPS=155 EVAL_STRATEGY=no \
      TRAIN_BS=1 GRAD_ACC=8 WORK_DIR="$out" HF_HOME=/workspace/hf-cache \
      PACK_MANIFEST="$B/packs/manifest.jsonl" PACK_ARM=pn LORA_SCOPE=decoder \
      PYTHONPATH="$B/code" python "$B/code/notebooks/train_runpod.py" 2>&1 | tee "$log"
  st=("${PIPESTATUS[@]}"); set -e
  cp -f "$log" "$out/train.log" 2>/dev/null || true
  [[ "${st[0]}" == 0 ]] || { say "❌ dec_s$s rc=${st[0]}"; exit 1; }
  [[ -s "$out/adapter/adapter_model.safetensors" ]] || { say "❌ dec_s$s χωρίς adapter"; exit 1; }
  date -Is > "$out/COMPLETE"
  say "✅ dec_s$s (decoder-only) τελείωσε"
done
date -Is > "$ROOT/DEC_COMPLETE"
say "🏁 Και τα 3 decoder-only seeds τελείωσαν"
