#!/usr/bin/env python3
"""Fail when the ledger and the repo have drifted apart.

The ledger is only worth reading if it cannot quietly go stale. This checks the
things that actually went wrong: dangling references, conclusions with no evidence,
artifacts nobody can identify, and dated reports that exist on disk but appear in no
experiment record.

    python3 scripts/check-research-state.py

Exit 0 clean, 1 with findings. Run it before declaring research work done.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "research/ledger.json"

EXP_STATUS = {"OPEN", "CLOSED", "SUPERSEDED"}
ART_STATUS = {"ACTIVE", "HISTORICAL", "KNOWN_BROKEN", "SUPERSEDED", "MISSING"}
CAP_STATUS = {"AVAILABLE", "DEGRADED", "UNKNOWN", "RETIRED"}

# Narrative pieces that predate the ledger and are deliberately not experiments.
# Advice is not evidence. A report that records what an outside reviewer *recommended*
# has no experiment behind it by construction, and inventing a record to satisfy this
# check would make the ledger claim a result that was never measured.
REPORT_EXEMPT = {
    "README.md", "month-1-2026-06.md", "research-findings-simple.md",
    "minipc-slides-prompt.md", "diarization-conditioned-asr-review.md",
    "2026-08-16-advisory-what-remains.md",
}


def main() -> int:
    bad: list[str] = []

    if not LEDGER.exists():
        print(f"FAIL {LEDGER.relative_to(ROOT)} is missing — the repo has no authoritative state")
        return 1
    try:
        led = json.loads(LEDGER.read_text())
    except json.JSONDecodeError as e:
        print(f"FAIL ledger is not valid JSON: {e}")
        return 1

    exps = led.get("experiments", [])
    arts = led.get("artifacts", [])
    caps = led.get("capabilities", [])

    exp_ids = {e["id"] for e in exps}
    art_ids = {a["id"] for a in arts}
    cap_ids = {c["id"] for c in caps}

    for label, items, ids in (("experiment", exps, exp_ids),
                              ("artifact", arts, art_ids),
                              ("capability", caps, cap_ids)):
        seen: set[str] = set()
        for it in items:
            i = it.get("id", "<missing id>")
            if i in seen:
                bad.append(f"duplicate {label} id: {i}")
            seen.add(i)

    # ---- experiments
    for e in exps:
        i = e.get("id", "?")
        st = e.get("status")
        if st not in EXP_STATUS:
            bad.append(f"{i}: status {st!r} not one of {sorted(EXP_STATUS)}")
        if st == "OPEN" and not e.get("next_action"):
            bad.append(f"{i}: OPEN with no next_action — nobody knows what to do")
        if st == "CLOSED":
            if not e.get("conclusion"):
                bad.append(f"{i}: CLOSED with no conclusion")
            if not e.get("decision"):
                bad.append(f"{i}: CLOSED with no decision — the next agent will rerun it")
            if not e.get("reports"):
                bad.append(f"{i}: CLOSED with no report — the conclusion has no evidence")
        if st == "SUPERSEDED" and not e.get("superseded_by"):
            bad.append(f"{i}: SUPERSEDED with no superseded_by")
        for r in e.get("reports", []):
            if not (ROOT / r).exists():
                bad.append(f"{i}: report does not exist: {r}")
        for s in e.get("specs", []):
            if not (ROOT / s).exists():
                bad.append(f"{i}: spec does not exist: {s}")
        for a in e.get("artifact_ids", []):
            if a not in art_ids:
                bad.append(f"{i}: unknown artifact_id {a}")
        for c in e.get("capability_ids", []):
            if c not in cap_ids:
                bad.append(f"{i}: unknown capability_id {c}")
        sb = e.get("superseded_by")
        if sb and sb not in exp_ids:
            bad.append(f"{i}: superseded_by points at unknown experiment {sb}")

    # ---- artifacts
    for a in arts:
        i = a.get("id", "?")
        st = a.get("status")
        if st not in ART_STATUS:
            bad.append(f"{i}: status {st!r} not one of {sorted(ART_STATUS)}")
        h = a.get("hash") or {}
        if st != "MISSING" and not h.get("value") and h.get("algorithm") != "none":
            bad.append(f"{i}: no hash and not marked MISSING — this artifact is unidentifiable")
        rb = a.get("replaced_by")
        if rb and rb not in art_ids:
            bad.append(f"{i}: replaced_by points at unknown artifact {rb}")
        if st == "KNOWN_BROKEN" and not a.get("caveats"):
            bad.append(f"{i}: KNOWN_BROKEN with no caveats saying why")
        for loc in a.get("locations", []):
            p = loc.get("path")
            if st != "MISSING" and loc.get("host") == "minipc" and p and not Path(p).exists():
                bad.append(f"{i}: recorded path is gone: {p} (mark it MISSING or fix it)")

    # ---- capabilities
    for c in caps:
        i = c.get("id", "?")
        st = c.get("status")
        if st not in CAP_STATUS:
            bad.append(f"{i}: status {st!r} not one of {sorted(CAP_STATUS)}")
        if st == "AVAILABLE":
            if not c.get("last_verified_at"):
                bad.append(f"{i}: AVAILABLE with no last_verified_at")
            rb = c.get("runbook")
            if not rb:
                bad.append(f"{i}: AVAILABLE with no runbook")
            elif not (ROOT / rb).exists():
                bad.append(f"{i}: runbook does not exist: {rb}")

    # ---- every dated report belongs to some experiment
    claimed = {r for e in exps for r in e.get("reports", [])}
    for p in sorted((ROOT / "docs/reports").glob("*.md")):
        if p.name in REPORT_EXEMPT:
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}-", p.name):
            continue
        rel = str(p.relative_to(ROOT))
        if rel not in claimed:
            bad.append(f"report in no experiment record: {rel}")

    if bad:
        print(f"{len(bad)} finding(s):\n")
        for b in bad:
            print(f"  - {b}")
        return 1
    print(f"ledger clean: {len(exps)} experiments, {len(arts)} artifacts, {len(caps)} capabilities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
