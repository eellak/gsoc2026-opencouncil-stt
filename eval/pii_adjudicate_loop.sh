#!/bin/bash
# Auto-resume wrapper for the PII adjudication (overnight-robust). Each
# `run` is resumable; this loop restarts it if it dies mid-way, and exits
# once no candidates remain to adjudicate. Max 80 iterations as a backstop.
cd /home/harold/opencouncil-fine-tuning || exit 1
PY=.venv-eval/bin/python
LOG=data/pii/adjudicate.log
for i in $(seq 1 80); do
  echo "=== wrapper iteration $i $(date -u +%H:%M:%S) ===" >> "$LOG"
  $PY -m eval.pii_adjudicate run --model sonnet >> "$LOG" 2>&1
  remaining=$($PY - <<'PYEOF'
import json, os
SV="3"; AV="2"
flagged=set()
for l in open("data/pii/scan.jsonl"):
    l=l.strip()
    if not l: continue
    d=json.loads(l)
    if d.get("scan_version")==SV and d.get("flag"):
        flagged.add(d["utterance_id"])
done=set()
if os.path.exists("data/pii/adjudicated.jsonl"):
    for l in open("data/pii/adjudicated.jsonl"):
        l=l.strip()
        if not l: continue
        try:
            d=json.loads(l)
            if d.get("adj_version")==AV: done.add(d["utterance_id"])
        except Exception: pass
print(len(flagged-done))
PYEOF
)
  echo "=== after iteration $i: $remaining candidates remaining ===" >> "$LOG"
  if [ "$remaining" = "0" ]; then
    echo "ADJUDICATION COMPLETE at iteration $i" >> "$LOG"
    break
  fi
  sleep 10
done
