#!/usr/bin/env bash
# Paired 300-step clean-pack screen (overlap-free contiguous packs vs the incumbent recipe), followed by one-stack GPU decode and scoring.
set -euo pipefail

BUNDLE="${BUNDLE:?set BUNDLE to the extracted bundle root}"
ROOT="${RUN_ROOT:-/runpod/runs/clean-pack-screen}"
STEPS="${MAX_STEPS:-300}"
mkdir -p "$ROOT"

preflight() {
  local trainer="$BUNDLE/code/notebooks/train_runpod.py"
  local empty
  [[ -s "$trainer" ]] || { echo "FATAL: trainer is empty or missing: $trainer" >&2; exit 10; }
  # __init__.py files are legitimately empty; everything else must not be.
  empty="$(find "$BUNDLE" -type f -size 0 ! -name '__init__.py' | wc -l)"
  [[ "$empty" == 0 ]] || {
    echo "FATAL: $empty zero-byte files in the bundle - it extracted incompletely" >&2
    find "$BUNDLE" -type f -size 0 ! -name '__init__.py' | head -5 >&2
    exit 11
  }
  [[ -s "$BUNDLE/packs/manifest.jsonl" ]] || { echo "FATAL: pack manifest empty" >&2; exit 12; }
  [[ -s "$BUNDLE/data/public/train.parquet" ]] || { echo "FATAL: control parquet empty" >&2; exit 13; }
  local packs audio
  packs="$(wc -l < "$BUNDLE/packs/manifest.jsonl")"
  audio="$(find "$BUNDLE/packs/audio" -type f | wc -l)"
  [[ "$packs" == "$audio" ]] || { echo "FATAL: $packs manifest rows but $audio audio files" >&2; exit 14; }
  # Arm B reads five manifest keys. One of them was only ever used in a log line, and a
  # typo there killed a run 25s into arm B -- after 110 minutes of arm A prep. Check the
  # contract here, where failing costs nothing.
  python -c "
import json,sys
r=json.loads(open('$BUNDLE/packs/manifest.jsonl').readline())
need={'audio','dur_sec','n_utterances','text_p','text_pn'}-set(r)
sys.exit('pack manifest missing %s' % sorted(need) if need else 0)" \
    || { echo "FATAL: pack manifest schema mismatch" >&2; exit 16; }
  python -c "import torch,transformers,peft; assert torch.cuda.is_available()" \
    || { echo "FATAL: no CUDA or missing deps" >&2; exit 15; }
  # The scorer runs at hour ~16, after every training run is paid for. Import it and
  # load its inputs now, so a missing dependency or a moved manifest costs seconds
  # instead of the whole lease.
  PYTHONPATH="$BUNDLE/code" python - "$BUNDLE" <<'PYSCORE' || { echo "FATAL: scorer cannot start" >&2; exit 17; }
import json, pathlib, sys
from eval.controlled_eval import dense_screen_eval as ds
b = pathlib.Path(sys.argv[1])
v = ds.load_rows(b / "eval/validation/manifest.json", "eval_windows", "audio_local")
t = ds.load_rows(b / "eval/training/sample.json", "rows", "clip")
missing = [r["audio_local"] for r in v if not pathlib.Path(r["audio_local"]).is_file()][:3]
if missing:
    sys.exit(f"FATAL: validation audio missing, e.g. {missing}")
print(f"  scorer ok: {len(v)} validation windows, {len(t)} training rows")
PYSCORE
  echo "preflight ok: $packs packs, no empty files, CUDA present"
}

finish() {
  rc=$?
  set +e
  # Every line here must tolerate a dead volume. Recording the status is a courtesy;
  # terminating the pod is the part that costs money if it does not happen, and a
  # failed write to /runpod must never abort the trap before the terminate call.
  printf '%s\n' "$rc" > "$ROOT/EXIT_STATUS" 2>/dev/null || true
  if [[ "${KEEP_POD:-0}" != 0 ]]; then return; fi
  if [[ -n "${RUNPOD_POD_ID:-}" && -f /root/.runpod_api_key ]]; then
    key="$(cat /root/.runpod_api_key)"
    curl -fsS --connect-timeout 15 --max-time 60 -X POST "https://api.runpod.io/graphql?api_key=$key" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"})}\"}" || true
  fi
}
# Installed BEFORE preflight on purpose. Preflight exists to catch exactly the
# failures that should not cost a full lease, and without the trap a preflight
# rejection left the pod idle until the watchdog fired hours later.
trap finish EXIT

preflight

run_one() {
  local arm="$1" seed="$2" out="$ROOT/${1}_s${2}"
  [[ -f "$out/COMPLETE" ]] && { echo "skip ${arm}_s${seed}"; return; }
  mkdir -p "$out"
  local log="/tmp/${arm}_s${seed}.log"
  # An adapter plus its checkpoint is ~260 MB, and a volume filling mid-run is how the
  # 2026-08-19 bundle ended up with 332 zero-byte files. `df` cannot answer this: the
  # volume is MooseFS and df reports the whole cluster (1.8P free), not our 100 GB
  # quota, so a df threshold is a guard that can never fail. Write and delete a real
  # 300 MB file instead - the quota answers a write, not a stat.
  local probe="$out/.space-probe"
  if ! dd if=/dev/zero of="$probe" bs=1M count=300 status=none 2>/dev/null; then
    rm -f "$probe"
    echo "FATAL: cannot write 300MB to $out - the volume quota is full" >&2
    exit 20
  fi
  rm -f "$probe"
  common=(SMOKE=0 SEED="$seed" MAX_STEPS="$STEPS" EVAL_STRATEGY=no SAVE_STEPS="$STEPS"
          TRAIN_BS=1 GRAD_ACC=8 WORK_DIR="$out" HF_HOME=/runpod/hf-cache)
  local -a status
  set +e
  if [[ "$arm" == A ]]; then
    env "${common[@]}" DATA_DIR="$BUNDLE/data/public" \
      PREP_CACHE_DIR=/runpod/prep/control FEATURE_CACHE_DIR=/runpod/features/control \
      python "$BUNDLE/code/notebooks/train_runpod.py" 2>&1 | tee "$log"
  else
    env "${common[@]}" PACK_MANIFEST="$BUNDLE/packs/manifest.jsonl" PACK_ARM=pn \
      PREP_CACHE_DIR=/runpod/prep/cleanpack FEATURE_CACHE_DIR=/runpod/features/cleanpack \
      python "$BUNDLE/code/notebooks/train_runpod.py" 2>&1 | tee "$log"
  fi
  status=("${PIPESTATUS[@]}")
  set -e
  local rc="${status[0]}" tee_rc="${status[1]}"
  [[ "$tee_rc" == 0 ]] || echo "WARN: tee exited $tee_rc for ${arm}_s${seed}" >&2
  # Copy the log to the volume only after the process is done. B_s29 on 2026-08-20
  # trained all 300 steps, saved its adapter, and was then killed because `tee` hit
  # "No such device or address" writing to the network volume: the broken pipe took
  # python down, the driver saw a non-zero status and terminated the pod. The log is
  # a diagnostic; it must never be able to destroy a finished run.
  cp -f "$log" "$out/train.log" 2>/dev/null || echo "WARN: could not copy log to volume" >&2
  [[ "$rc" == 0 ]] || exit 1
  # An exit code of 0 is not proof of work: the empty-trainer bug produced exactly that.
  [[ -s "$log" ]] || { echo "FATAL: ${arm}_s${seed} wrote an empty log" >&2; exit 2; }
  [[ -s "$out/adapter/adapter_model.safetensors" ]] \
    || { echo "FATAL: ${arm}_s${seed} wrote no adapter weights" >&2; exit 3; }
  # Non-zero length is not evidence. Open the tensors and confirm the trainer really
  # reached the last step: an interrupted save and a 6-second phantom resume both
  # produce a plausible-looking file.
  python - "$out" "$STEPS" <<'PYCHECK' || exit 4
import json, pathlib, sys
out, steps = pathlib.Path(sys.argv[1]), int(sys.argv[2])
from safetensors.torch import load_file
t = load_file(out / "adapter" / "adapter_model.safetensors")
if not t:
    sys.exit("FATAL: adapter has no tensors")
st = out / "adapter" / f"checkpoint-{steps}" / "trainer_state.json"
if not st.is_file():
    sys.exit(f"FATAL: no trainer_state at step {steps}")
gs = json.loads(st.read_text())["global_step"]
if gs != steps:
    sys.exit(f"FATAL: global_step {gs}, expected {steps}")
print(f"  verified: {len(t)} tensors, global_step {gs}")
PYCHECK
  date -Is > "$out/COMPLETE"
}

for seed in 13 29 47; do
  run_one A "$seed"
  run_one B "$seed"
done
date -Is > "$ROOT/ALL_TRAINING_COMPLETE"
echo "all six adapters trained and verified; scoring next"

PYTHONPATH="$BUNDLE/code" HF_HOME=/runpod/hf-cache python -m \
  eval.controlled_eval.dense_screen_eval all \
  --runs "$ROOT" \
  --validation "$BUNDLE/eval/validation/manifest.json" \
  --training "$BUNDLE/eval/training/sample.json" \
  --out "$ROOT/hypotheses" --ct2 "$ROOT/ct2-work" \
  --result "$ROOT/results.json" 2>&1 | tee /tmp/eval.log
score_rc="${PIPESTATUS[0]}"
cp -f /tmp/eval.log "$ROOT/eval.log" 2>/dev/null || echo "WARN: could not copy eval log" >&2
# Scoring runs at hour ~16 of a 20-hour lease, after every training run is already paid
# for. Its log must not be able to kill it, and a failure here must say so loudly
# instead of leaving an ALL_COMPLETE that means nothing.
[[ "$score_rc" == 0 ]] || { echo "FATAL: scoring failed rc=$score_rc; adapters are safe on the volume" >&2; exit 30; }
[[ -s "$ROOT/results.json" ]] || { echo "FATAL: scoring wrote no results.json" >&2; exit 31; }
date -Is > "$ROOT/ALL_COMPLETE"
