#!/usr/bin/env python3
"""Score a diarization arm as a merge/split pair, per meeting.

Purity alone cannot be the outcome: a system that gives every utterance its own
voice scores perfect purity and is useless. So every arm is reported as B-cubed
**precision and recall together**:

- **precision** falls when one predicted voice covers several real people —
  that is *merging*, the error this whole experiment is about.
- **recall** falls when one real person is spread over several predicted voices
  — that is *splitting*, the price a fix usually pays.

Reported both unweighted (one vote per utterance) and duration-weighted, because
a chair who holds the floor must not drown out fifteen brief speakers.

Identity policy, frozen: a reviewed utterance's identity is its **`personId`**
when the reviewers resolved one, otherwise its `speakerTagId` prefixed with
``tag:``. This matters — a speaker tag is partly a pyannote artefact (one person
in this cohort carries up to 20 tags), so scoring by tag would fold the previous
run's splitting into the answer key. Utterances whose identity is an unresolved
tag are counted, and reported separately, because excluding them would quietly
shrink the denominator.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

CACHE = Path(os.environ.get("OC_CACHE", Path.home() / ".cache/oc-public"))
RUN_DIR = CACHE / "diarization-clustering-2026-09"

# Arms whose voice count is one pyannote job's own cluster inventory, and so can
# speak to the ceiling gate. Sliced arms namespace speakers per slice, which
# inflates the count by construction — see G1 in README.md.
WHOLE_JOB_ARMS = frozenset({"a0", "a1", "a2"})


# --- reference -------------------------------------------------------------

def reference_turns(snapshot: dict) -> list[dict]:
    """Human-reviewed speaker turns, identity resolved by the frozen policy."""
    tags = {t["id"]: t for t in (snapshot.get("speakerTags") or [])}
    turns = []
    for seg in snapshot.get("transcript") or []:
        tid = seg.get("speakerTagId")
        tag = tags.get(tid) or {}
        person = tag.get("personId")
        identity = person if person else f"tag:{tid}"
        for u in seg.get("utterances") or []:
            turns.append({
                "start": float(u["startTimestamp"]),
                "end": float(u["endTimestamp"]),
                "identity": identity,
                "resolved": bool(person),
            })
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


# --- assignment ------------------------------------------------------------

def assign(turn: dict, segments: list[dict]) -> str | None:
    """Give a turn the speaker covering most of it; None when nothing overlaps.

    Max-overlap, not midpoint: a midpoint rule silently hands a long turn to
    whoever happens to sit at its centre.

    Overlap is **summed per speaker**. Diarizers emit one segment per continuous
    stretch of voice, so a speaker who dominates a turn across three short
    segments would otherwise lose it to someone with one marginally longer
    segment — and that error grows exactly where turns are contested.
    """
    totals: dict[str, float] = {}
    for s in segments:
        ov = min(turn["end"], s["end"]) - max(turn["start"], s["start"])
        if ov > 0:
            totals[s["speaker"]] = totals.get(s["speaker"], 0.0) + ov
    if not totals:
        return None
    return max(totals, key=totals.get)


# --- metrics ---------------------------------------------------------------

def bcubed(pairs: list[tuple[str, str | None, float]]) -> dict:
    """B-cubed precision/recall over (reference, hypothesis, weight) triples.

    Unassigned hypotheses are kept as singleton clusters rather than dropped:
    a system that refuses to answer must not score better for it.
    """
    items = [(r, h if h is not None else f"__unassigned_{i}", w)
             for i, (r, h, w) in enumerate(pairs)]
    if not items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "items": 0}

    by_hyp: dict[str, float] = defaultdict(float)
    by_ref: dict[str, float] = defaultdict(float)
    joint: dict[tuple[str, str], float] = defaultdict(float)
    for r, h, w in items:
        by_hyp[h] += w
        by_ref[r] += w
        joint[(r, h)] += w

    total = sum(w for _, _, w in items)
    p = sum(w * joint[(r, h)] / by_hyp[h] for r, h, w in items) / total
    rec = sum(w * joint[(r, h)] / by_ref[r] for r, h, w in items) / total
    f1 = 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)
    return {"precision": round(p, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "items": len(items)}


def score(turns: list[dict], segments: list[dict]) -> dict:
    pairs_u, pairs_d = [], []
    unassigned = 0
    for t in turns:
        h = assign(t, segments)
        if h is None:
            unassigned += 1
        pairs_u.append((t["identity"], h, 1.0))
        pairs_d.append((t["identity"], h, max(t["end"] - t["start"], 1e-6)))

    resolved = [t for t in turns if t["resolved"]]
    pairs_r = [(t["identity"], assign(t, segments), 1.0) for t in resolved]

    hyp_voices = len({s["speaker"] for s in segments})
    return {
        "turns": len(turns),
        "unassigned": unassigned,
        "unassigned_percent": round(100 * unassigned / max(len(turns), 1), 2),
        "hypothesis_voices": hyp_voices,
        "reference_identities": len({t["identity"] for t in turns}),
        "reference_resolved_people": len({t["identity"] for t in resolved}),
        "bcubed": bcubed(pairs_u),
        "bcubed_duration_weighted": bcubed(pairs_d),
        "bcubed_resolved_only": bcubed(pairs_r),
    }


def load_arm(mdir: Path, arm: str) -> list[dict] | None:
    p = mdir / f"diarization_{arm}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("exclusiveDiarization") or d["diarization"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["a0"])
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    a = ap.parse_args()

    cohort = json.loads((a.run_dir / "cohort.json").read_text())
    snap_dir = a.run_dir / "reference-snapshot"
    out: dict[str, dict] = {}

    for arm in a.arms:
        print(f"\n=== {arm} ===")
        print(f'{"meeting":24} {"voices":>6} {"refIDs":>6} {"B3-P":>7} {"B3-R":>7} '
              f'{"B3-F":>7} {"dwF":>7} {"unasg%":>7}')
        rows = {}
        for rec in cohort:
            key = f'{rec["city"]}__{rec["meeting"]}'
            mdir = a.run_dir / key
            segs = load_arm(mdir, arm)
            if segs is None:
                print(f"{rec['city']+'/'+rec['meeting']:24} (not run)")
                continue
            snap = json.loads((snap_dir / f"{key}.json").read_text())
            r = score(reference_turns(snap), segs)
            rows[key] = r
            b, d = r["bcubed"], r["bcubed_duration_weighted"]
            print(f'{rec["city"]+"/"+rec["meeting"]:24} {r["hypothesis_voices"]:6} '
                  f'{r["reference_identities"]:6} {b["precision"]:7.4f} {b["recall"]:7.4f} '
                  f'{b["f1"]:7.4f} {d["f1"]:7.4f} {r["unassigned_percent"]:7.2f}')
        out[arm] = rows
        if rows:
            mf = sum(v["bcubed"]["f1"] for v in rows.values()) / len(rows)
            mp = sum(v["bcubed"]["precision"] for v in rows.values()) / len(rows)
            mr = sum(v["bcubed"]["recall"] for v in rows.values()) / len(rows)
            mx = max(v["hypothesis_voices"] for v in rows.values())
            print(f'{"MEAN (equal meetings)":24} {"":6} {"":6} {mp:7.4f} {mr:7.4f} {mf:7.4f}')
            # G1 is about what ONE pyannote job returns under automatic counting.
            # Sliced arms namespace speakers per slice, so their voice count is an
            # artefact of the partition, not a model output, and cannot falsify it.
            if arm in WHOLE_JOB_ARMS:
                verdict = "FALSIFIED" if mx >= 35 else "not falsified"
                print(f'G1 ceiling: max voices in one job = {mx} -> {verdict} '
                      f'(threshold 35, frozen before the run)')
            else:
                print(f'G1 not applicable to {arm}: {mx} voices is the namespaced '
                      f'partition count, not one job\'s cluster inventory')

    (a.run_dir / "scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwritten to {a.run_dir / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
