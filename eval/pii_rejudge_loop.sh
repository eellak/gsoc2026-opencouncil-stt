#!/bin/bash
# Auto-resume wrapper for the before-only after-text re-judge (overnight-robust).
cd /home/harold/opencouncil-fine-tuning || exit 1
PY=.venv-eval/bin/python
LOG=data/pii/rejudge.log
for i in $(seq 1 60); do
  echo "=== rejudge iteration $i $(date -u +%H:%M:%S) ===" >> "$LOG"
  $PY -m eval.pii_rejudge_after >> "$LOG" 2>&1
  remaining=$($PY - <<'PYEOF'
import json
scan={}
for l in open("data/pii/scan.jsonl"):
    l=l.strip()
    if not l: continue
    d=json.loads(l)
    if d.get("scan_version")=="3" and d.get("flag") and d.get("hit_field")=="before":
        scan[d["utterance_id"]]=1
done=set()
for l in open("data/pii/adjudicated.jsonl"):
    l=l.strip()
    if not l: continue
    try:
        d=json.loads(l)
        if d.get("rejudged_after"): done.add(d["utterance_id"])
    except Exception: pass
print(len(set(scan)-done))
PYEOF
)
  echo "=== after iteration $i: $remaining before-only rows remaining ===" >> "$LOG"
  if [ "$remaining" = "0" ]; then
    echo "REJUDGE COMPLETE at iteration $i" >> "$LOG"
    break
  fi
  sleep 10
done
