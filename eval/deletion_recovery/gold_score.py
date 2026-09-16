# -*- coding: utf-8 -*-
"""Step 3 of `exp-2026-09-16-deletion-recovery`: does putting the words back make
the transcript closer to what a human heard?

The 391-window screen said a blind restore is catastrophic on
agreement-with-OpenCouncil (0.09056 -> 0.12908) while removing almost all of the
deletion gap, and that only about one donor-only word in five appears in the
published reference. That reference is itself a clean read, so it cannot answer
whether the other four were spoken. This asks the only thing that can: the frozen
gold set, 30 cells in 6 meetings, transcribed by a human who listened.

THE ARMS ARE FROZEN AND ALL OF THEM ARE REPORTED. THRESHOLD_GRID below is the
whole search, declared before any score was computed, and no arm is selected after
the fact -- the same discipline gold_set_score.py states for its regions. With 30
cells this set can falsify the idea; it cannot certify it.

Both arms come from ElevenLabs Scribe v2 on the same audio, differing only in
`no_verbatim`, so a restored word carries its own timestamp and goes back at that
instant. No cross-vendor alignment is involved anywhere in this file.

Aggregates only. Transcript text never leaves the cache.

    .venv-eval/bin/python -m eval.deletion_recovery.gold_score
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.scoring import wtoks                       # noqa: E402
from eval.deletion_recovery.screen import _backtrace, is_filler      # noqa: E402
from eval.gold_set_score import (                                    # noqa: E402
    GOLD, REGIONS, gold_view, in_intervals, load, masked_intervals, region_span,
)
from eval.soniox_confidence_probe import _align, word_units          # noqa: E402

BASE_DIR = GOLD / "hyp" / "scribe-nv"    # clean read: what production would ship
DONOR_DIR = GOLD / "hyp" / "scribe"      # verbatim: the donor, same vendor
OUT = ROOT / "eval" / "results_deletion_recovery_gold.json"
PRIMARY_REGION = "core_envelope"

# The entire search, frozen. Names are quantiles OF THE DONOR'S OWN emitted-word
# logprobs, computed per run over every donor word on the set -- a property of the
# donor, never of the score. "none" restores every non-filler candidate, which is
# the arm the 391-window screen already priced on agreement.
THRESHOLD_GRID = (("none", None), ("p25", 0.25), ("median", 0.50), ("p75", 0.75))


def read_words(path: Path) -> list[dict]:
    """Raw Scribe response -> every emitted word, in time order.

    A word with no logprob is still a word the model said. Dropping it here would
    delete it from the transcript being scored and charge the arm a deletion it
    never made. The confidence gate belongs at candidate selection, where a missing
    logprob simply means the word cannot clear a threshold. Measured 2026-09-16:
    coverage on this set is 2378/2378 verbatim and 2126/2126 clean, so this changes
    no number here -- it stops the next set from lying quietly.
    """
    body = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for w in body.get("words", []):
        if w.get("type") != "word":
            continue
        lp = w.get("logprob")
        out.append({"raw": w.get("text", ""), "start": w.get("start"),
                    "end": w.get("end"), "logprob": lp,
                    "conf": None if lp is None else max(0.0, min(1.0, math.exp(lp)))})
    return out


def _key(word: dict) -> str:
    """The token space the alignment runs in: the scorer's, not the vendor's."""
    ts = wtoks(word.get("raw", ""))
    return ts[0] if ts else ""


def candidates(base: list[dict], donor: list[dict]) -> list[dict]:
    """Donor words sitting where the base emitted nothing, fillers excluded.

    A substitution is not a candidate: both arms spoke in that slot, so the
    question there is which word is right, not which word is missing.
    """
    bk = [_key(w) for w in base]
    dk = [_key(w) for w in donor]
    keep_b = [i for i, t in enumerate(bk) if t]
    keep_d = [j for j, t in enumerate(dk) if t]
    ops = _backtrace([bk[i] for i in keep_b], [dk[j] for j in keep_d])
    out = []
    for op, _i, j in ops:
        if op != "donor_only":
            continue
        word = donor[keep_d[j]]
        if is_filler(_key(word)):
            continue
        out.append(word)
    return out


def restore(base: list[dict], donor: list[dict], min_logprob: float | None) -> list[dict]:
    """The base stream with the donor's confident additions put back, by time."""
    picked = [w for w in candidates(base, donor)
              if min_logprob is None
              or (w["logprob"] is not None and w["logprob"] >= min_logprob)]
    merged = list(base) + picked
    merged.sort(key=lambda w: (w["start"] if w["start"] is not None else 0.0))
    return merged


def edits(ref_tokens: list[str], words: list[dict]) -> tuple[int, int, int, int]:
    """(sub, del, ins, ref_len) of a word stream against the gold token sequence."""
    units, _dropped, _split = word_units(words)
    ops = _align(ref_tokens, [u["tok"] for u in units], ("S", "D", "I"), False)
    s = d = i = 0
    for op, _a, _b in ops:
        if op == "S":
            s += 1
        elif op == "D":
            d += 1
        elif op == "I":
            i += 1
    return s, d, i, len(ref_tokens)


def cell_streams(cid: str, ans: dict, region: str):
    """Base and donor words inside the region's scored intervals, or None."""
    seq, _by, _ov, blocks, _kept = gold_view(ans[cid], region)
    span = region_span(blocks, region)
    if span is None or not seq:
        return None
    outside = [b for b in ans[cid]["b"]["blocks"]
               if b["id"] not in {x["id"] for x in blocks}]
    ivs = masked_intervals(blocks, span, outside)

    def clip(words):
        return [w for w in words
                if w["start"] is not None and in_intervals(w["start"], ivs)]

    base = clip(read_words(BASE_DIR / f"{cid}.json"))
    donor = clip(read_words(DONOR_DIR / f"{cid}.json"))
    return seq, base, donor


def main() -> int:
    man, ans, _cells, sel = load()
    scored = [c for c in man["scored_cell_ids"]
              if (BASE_DIR / f"{c}.json").exists() and (DONOR_DIR / f"{c}.json").exists()]
    if not scored:
        print(f"no cells decoded in both modes; run eval/scribe_gold_decode.py")
        return 2

    # The donor's own logprob distribution, over every emitted word on the set.
    # Computed before any arm is scored, and it does not look at the gold labels.
    pool = [w["logprob"] for c in scored for w in read_words(DONOR_DIR / f"{c}.json")
            if w["logprob"] is not None]
    cuts = {"none": None}
    for name, q in THRESHOLD_GRID:
        if q is not None:
            cuts[name] = statistics.quantiles(pool, n=100)[int(q * 100) - 1]

    result = {
        "protocol": "deletion-recovery-gold-2026-09-16",
        "experiment": "exp-2026-09-16-deletion-recovery",
        "cells": len(scored),
        "meetings": len({sel[c]["meeting_id"] for c in scored}),
        "donor_logprob_cuts": cuts,
        "regions": {},
    }

    for region in REGIONS:
        arms: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        restored_counts: dict[str, int] = defaultdict(int)
        per_meeting: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0]))
        cand_total = 0
        used: list[str] = []
        for cid in scored:
            got = cell_streams(cid, ans, region)
            if got is None:
                continue
            used.append(cid)
            seq, base, donor = got
            cand_total += len(candidates(base, donor))
            meeting = sel[cid]["meeting_id"]
            streams = {"clean": base}
            for name, _q in THRESHOLD_GRID:
                streams[f"restored_{name}"] = restore(base, donor, cuts[name])
                restored_counts[name] += len(restore(base, donor, cuts[name])) - len(base)
            streams["donor_verbatim"] = donor
            for name, stream in streams.items():
                s, d, i, n = edits(seq, stream)
                a = arms[name]
                a[0] += s; a[1] += d; a[2] += i; a[3] += n
                pm = per_meeting[meeting][name]
                pm[0] += s + d + i
                pm[1] += n

        # Per region, not the global decode count: a cell with no block inside the
        # region contributes nothing, and reporting the global figure beside a
        # region's WER overstates what that WER rests on.
        region_out = {"cells": len(used),
                      "meetings": len({sel[c]["meeting_id"] for c in used}),
                      "candidates": cand_total, "restored_words": dict(restored_counts),
                      "arms": {}, "per_meeting": {}}
        for name, (s, d, i, n) in arms.items():
            region_out["arms"][name] = {
                "wer": (s + d + i) / n if n else None, "sub": s / n if n else None,
                "del": d / n if n else None, "ins": i / n if n else None,
                "ref_words": n,
            }
        base_arm = arms["clean"]
        base_wer = (sum(base_arm[:3]) / base_arm[3]) if base_arm[3] else None
        for name, (s, d, i, n) in arms.items():
            if base_wer is not None and n:
                region_out["arms"][name]["delta_vs_clean"] = (s + d + i) / n - base_wer
        for meeting, by_arm in sorted(per_meeting.items()):
            region_out["per_meeting"][meeting] = {
                name: (v[0] / v[1] if v[1] else None) for name, v in by_arm.items()}
        result["regions"][region] = region_out

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    r = result["regions"][PRIMARY_REGION]
    print(f"{r['cells']} cells / {r['meetings']} meetings IN THIS REGION "
          f"(of {result['cells']} decoded in both modes), "
          f"region {PRIMARY_REGION} (primary; all three are in the json)")
    print(f"non-filler donor-only candidates: {r['candidates']}")
    print(f"donor logprob cuts: "
          + "  ".join(f"{k}={v:.4f}" if v is not None else f"{k}=none"
                      for k, v in result["donor_logprob_cuts"].items()))
    print()
    order = ["clean"] + [f"restored_{n}" for n, _ in THRESHOLD_GRID] + ["donor_verbatim"]
    for name in order:
        a = r["arms"].get(name)
        if not a:
            continue
        put = ""
        if name.startswith("restored_"):
            put = f"  (+{r['restored_words'].get(name.split('_', 1)[1], 0)} words)"
        print(f"{name:18s} WER {a['wer']:.5f}  sub {a['sub']:.5f}  del {a['del']:.5f}  "
              f"ins {a['ins']:.5f}  Δ {a.get('delta_vs_clean', 0):+.5f}{put}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
