#!/usr/bin/env python3
"""Stage 3: build the cell grid and the two independent candidate signals.

A CELL is a fixed slot on a 15 s grid over a sampled meeting. Only the core
(15 s) is scored; the page also plays LEAD seconds before and TAIL after, as
listening context that is never scored.

Two signals, computed over the WHOLE of every sampled meeting so that every
cell's stream membership - and therefore its inclusion probability - is known:

  P  pyannoteAI precision-2 `diarization`: >= MIN_OV seconds of two-or-more
     simultaneous turns inside the core.
  I  the published OpenCouncil transcript (a different pipeline, human-reviewed
     for these meetings): a speaker segment shorter than SHORT_TURN seconds
     whose neighbours on both sides are a different speaker - an interjection,
     which in this domain is where people talk over each other. Independent of
     pyannote by construction. Counted only when the cell is NOT P, so the two
     streams are mutually exclusive.
  H  handover control: a pyannote speaker change in the core, no P, no I.
  M  mono control: exactly one pyannote speaker in the core.
  R  everything eligible, for the probability anchor.

No text is read out of the published transcript here - only timestamps and
speaker-tag labels. Nothing written by this script contains speech.
"""
import csv
import json
import os
from pathlib import Path

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
ROOT = Path(__file__).resolve().parents[2]

CORE = 15.0          # scored seconds per cell
LEAD = 10.0          # unscored context before
TAIL = 10.0          # unscored context after
MIN_OV = 0.40        # seconds of simultaneous speech to call a core P
SHORT_TURN = 3.0     # published-segment duration that counts as an interjection
MIN_SPEECH = 0.55    # fraction of the core that must be speech to be eligible


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_spans(turns):
    """Intervals where two or more diarization turns are simultaneous."""
    ev = []
    for t in turns:
        ev.append((t["start"], 1))
        ev.append((t["end"], -1))
    ev.sort()
    cur, prev, out = 0, None, []
    for tm, d in ev:
        if cur >= 2 and prev is not None and tm > prev:
            out.append([prev, tm])
        cur += d
        prev = tm
    return merge(out)


def speech_spans(turns):
    return merge([[t["start"], t["end"]] for t in turns])


def isect(spans, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in spans)


def build(city, mid, duration):
    diar = json.loads((SC / "gold-set/diar" / f"{city}__{mid}.json").read_text())["output"]
    turns = diar["diarization"]
    ov = overlap_spans(turns)
    sp = speech_spans(turns)

    pub = json.loads((SC / "meetings" / f"{city}__{mid}.json").read_text())["transcript"]
    segs = sorted(((s["startTimestamp"], s["endTimestamp"], s["speakerTag"]["label"])
                   for s in pub), key=lambda x: x[0])
    interj = []
    for i in range(1, len(segs) - 1):
        a, b, c = segs[i - 1], segs[i], segs[i + 1]
        if (b[1] - b[0]) <= SHORT_TURN and b[2] != a[2] and b[2] != c[2]:
            interj.append((b[0] + b[1]) / 2.0)

    cells = []
    n = int((duration - LEAD - TAIL) // CORE)
    for i in range(n):
        a = LEAD + i * CORE
        b = a + CORE
        speech = isect(sp, a, b)
        if speech / CORE < MIN_SPEECH:
            continue
        ov_sec = isect(ov, a, b)
        spk = {t["speaker"] for t in turns if t["end"] > a and t["start"] < b}
        n_turns = sum(1 for t in turns if t["end"] > a and t["start"] < b)
        n_interj = sum(1 for m in interj if a <= m < b)
        if ov_sec >= MIN_OV:
            stream = "P"
        elif n_interj > 0:
            stream = "I"
        elif len(spk) >= 2:
            stream = "H"
        else:
            stream = "M"
        cells.append({
            "cell_id": f"gc_{city}_{mid}_{int(round(a * 1000))}",
            "city_id": city, "meeting_id": mid,
            "core_start": round(a, 3), "core_end": round(b, 3),
            "clip_start": round(a - LEAD, 3), "clip_end": round(b + TAIL, 3),
            "speech_frac": round(speech / CORE, 3),
            "pyannote_overlap_sec": round(ov_sec, 3),
            "pyannote_speakers": len(spk),
            "pyannote_turns": n_turns,
            "pub_interjections": n_interj,
            "stream": stream,
        })
    return cells


def main():
    with open(ROOT / "research/gold-set-2026-08/meetings.csv", newline="", encoding="utf-8") as fh:
        meets = list(csv.DictReader(fh))
    allc = []
    for m in meets:
        c = build(m["city_id"], m["meeting_id"], float(m["duration_sec"]))
        allc += c
        from collections import Counter
        print(f"{m['city_id']:12s} cells={len(c):5d} " + str(dict(sorted(Counter(x['stream'] for x in c).items()))))
    out = ROOT / "research/gold-set-2026-08/cells.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allc[0].keys()))
        w.writeheader()
        w.writerows(allc)
    from collections import Counter
    print("TOTAL", len(allc), dict(sorted(Counter(x["stream"] for x in allc).items())))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
