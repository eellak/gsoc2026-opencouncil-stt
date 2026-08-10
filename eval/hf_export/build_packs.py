#!/usr/bin/env python3
"""Pack consecutive utterances into 20-30s training examples with per-utterance timestamps.

Phase 0 of the [unattended run](../../docs/runbooks/2026-08-10-unattended-packing-run.md).

The problem it attacks, measured rather than assumed: the current training set is 28,967
clips averaging 2.79s inside Whisper's fixed 30-second encoder window, so **91% of every
step is padding**, and 7,242 steps at 8 clips per step is exactly 2.00 epochs over 22.47
hours. The optimizer pays for 483 hours of window to be shown 45 hours of speech. It also
never sees a speaker change inside one example, and never sees a timestamp.

Packing fixes all three at once, and the gaps are where it can go wrong.

**Gaps are silence, never the original audio between utterances.** The published transcript
disagrees with a careful listener by 17.5%, mostly short interjections nobody wrote down.
The audio sitting between two transcribed utterances is exactly where those live. Splicing
it in with no label would train the model to hear speech and emit nothing, which is the
failure this whole line of work exists to remove. Silence is artificial; unlabelled speech
is poison.

Two arms come out of one build, sharing byte-identical audio and pack ids:
  P   native timestamps, <|0.00|> text <|2.34|> <|2.90|> text <|5.10|> ...
  Pn  the same text concatenated, no timestamps

  .venv-eval/bin/python -m eval.hf_export.build_packs --out data/hf-dataset/packs
Env: SRC MIN_SEC MAX_SEC MAX_GAP_FILL SPLIT_GAP
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
SRC = os.environ.get("SRC", "data/hf-dataset/public/train.parquet")
MIN_SEC = float(os.environ.get("MIN_SEC", "20"))
MAX_SEC = float(os.environ.get("MAX_SEC", "29.5"))   # headroom under Whisper's 30s window
GAP_SIL = float(os.environ.get("GAP_SIL", "0.4"))    # fixed silence between utterances
GRID = 0.02          # Whisper's timestamp resolution


def log(*a):
    print(*a, flush=True)


def span(r):
    """The boundary-corrected span when it exists, exactly as the trainer picks it."""
    sa, ea = r.get("start_adj"), r.get("end_adj")
    ok = lambda x: x is not None and x == x
    return (float(sa), float(ea)) if ok(sa) and ok(ea) else (float(r["start"]), float(r["end"]))


def floor_grid(x):
    # math.floor/ceil, not int(): int() truncates toward zero, and -int(-x/GRID) turned
    # 2.34 into 2.32 whenever the division landed a hair under the integer. That produced
    # 107 packs with non-monotonic timestamps on the first build.
    return round(math.floor(x / GRID + 1e-9) * GRID, 2)


def ceil_grid(x):
    return round(math.ceil(x / GRID - 1e-9) * GRID, 2)


def build(rows) -> list[dict]:
    """Greedy packs over consecutive selected utterances of one meeting, in time order.

    The real gap between two rows is ignored, and a fixed silence goes in instead. That is
    not a shortcut: the training set is a *selection* (corrections plus sampled clean rows),
    so two consecutive rows in one meeting are routinely minutes apart. Honouring real gaps
    made the mean pack 1.3 utterances long, which is the status quo with extra steps. The
    pack is a synthetic construction either way; what matters is that every second of audio
    inside it carries a label.
    """
    by_mtg = collections.defaultdict(list)
    for r in rows:
        by_mtg[(r["city_id"], r["meeting_id"])].append(r)

    packs, pid = [], 0
    for key, items in sorted(by_mtg.items()):
        items.sort(key=lambda r: span(r)[0])
        cur, t = [], 0.0
        for r in items:
            s, e = span(r)
            dur = e - s
            if dur < GRID or dur > MAX_SEC:
                continue
            fill = 0.0 if not cur else GAP_SIL
            if cur and t + fill + dur > MAX_SEC:
                packs.append(finish(cur, key, pid)); pid += 1
                cur, t, fill = [], 0.0, 0.0
            t0 = floor_grid(t + fill)
            t1 = ceil_grid(t0 + dur)
            if t1 <= t0:
                t1 = round(t0 + GRID, 2)
            cur.append({"utterance_id": r["utterance_id"], "text": (r["text"] or "").strip(),
                        "src_start": s, "src_end": e, "gap_fill": round(fill, 3),
                        "t_start": t0, "t_end": t1})
            t = t1
        if cur:
            packs.append(finish(cur, key, pid)); pid += 1
    return packs


def finish(items, key, pid) -> dict:
    return {"pack_id": f"pk_{pid:06d}", "city_id": key[0], "meeting_id": key[1],
            "dur_sec": items[-1]["t_end"], "n_utt": len(items),
            "speech_sec": round(sum(i["src_end"] - i["src_start"] for i in items), 3),
            "silence_sec": round(sum(i["gap_fill"] for i in items), 3),
            "items": items}


def gates(packs, rows) -> list[str]:
    """Every accounting and timing invariant the runbook demands. Returns failures."""
    bad = []
    src = {r["utterance_id"]: r for r in rows}
    used = collections.Counter(i["utterance_id"] for p in packs for i in p["items"])

    dup = [u for u, n in used.items() if n > 1]
    if dup:
        bad.append(f"{len(dup)} utterances in more than one pack, e.g. {dup[:3]}")

    # Dropped utterances are legitimate (zero/over-long spans) but must be counted, never
    # silently absorbed: a packing bug and a filter look identical in the totals.
    missing = set(src) - set(used)
    drop_h = sum(max(0.0, span(src[u])[1] - span(src[u])[0]) for u in missing) / 3600
    log(f"  dropped {len(missing)} utterances ({drop_h:.2f} h) as sub-grid or >{MAX_SEC}s")
    if drop_h > 0.5:
        bad.append(f"dropped {drop_h:.2f} h of source speech, over the 0.5 h limit")

    speech_src = sum(max(0.0, span(src[u])[1] - span(src[u])[0]) for u in used) / 3600
    speech_pack = sum(p["speech_sec"] for p in packs) / 3600
    if abs(speech_src - speech_pack) > 0.01:
        bad.append(f"speech hours {speech_pack:.3f} != source {speech_src:.3f}")

    for p in packs:
        if p["dur_sec"] > 30.0:
            bad.append(f"{p['pack_id']} is {p['dur_sec']}s, over the encoder window")
        last = -1.0
        for i in p["items"]:
            if i["t_start"] < last or i["t_end"] <= i["t_start"] or i["t_start"] < 0:
                bad.append(f"{p['pack_id']} timestamps not monotonic/positive")
                break
            if abs((i["t_end"] - i["t_start"]) - (i["src_end"] - i["src_start"])) > GRID + 1e-9:
                bad.append(f"{p['pack_id']} timestamp span off by more than one grid step")
                break
            if not i["text"]:
                bad.append(f"{p['pack_id']} has an empty target segment")
                break
            last = i["t_end"]
        if p["items"][0]["t_start"] > 1.0:
            bad.append(f"{p['pack_id']} starts at {p['items'][0]['t_start']}s; Whisper's "
                       f"decoder cannot emit a first timestamp that late")

    # City balance, measured on speech and on composite duration, since inserted silence
    # can shift the apparent mix without a single word changing.
    def shares(f):
        tot = collections.Counter()
        for p in packs:
            tot[p["city_id"]] += f(p)
        s = sum(tot.values())
        return {c: 100 * v / s for c, v in tot.items()}
    src_sh = collections.Counter()
    for u in used:
        r = src[u]
        src_sh[r["city_id"]] += span(r)[1] - span(r)[0]
    tot = sum(src_sh.values())
    src_sh = {c: 100 * v / tot for c, v in src_sh.items()}
    for label, got in (("speech", shares(lambda p: p["speech_sec"])),
                       ("composite", shares(lambda p: p["dur_sec"]))):
        for c, v in got.items():
            if abs(v - src_sh.get(c, 0)) > 2.0:
                bad.append(f"city {c} {label} share {v:.1f}% vs source "
                           f"{src_sh.get(c, 0):.1f}%, over 2 points")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hf-dataset/packs")
    a = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(ROOT / SRC)
    rows = df.to_dict("records")
    log(f"{len(rows)} source utterances from {SRC}")

    packs = build(rows)
    dur = [p["dur_sec"] for p in packs]
    nutt = [p["n_utt"] for p in packs]
    log(f"{len(packs)} packs, mean {sum(dur)/len(dur):.1f}s, "
        f"mean {sum(nutt)/len(nutt):.1f} utterances, "
        f"{sum(dur)/3600:.2f} h composite, {sum(p['speech_sec'] for p in packs)/3600:.2f} h speech, "
        f"{sum(p['silence_sec'] for p in packs)/3600:.2f} h inserted silence")
    log(f"  under {MIN_SEC:.0f}s: {sum(1 for d in dur if d < MIN_SEC)} packs "
        f"(tails of meetings, kept)")

    bad = gates(packs, rows)
    out = Path(ROOT / a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "packs.json").write_text(json.dumps(packs, ensure_ascii=False))
    if bad:
        log(f"\nGATES FAILED ({len(bad)}):")
        for b in bad[:20]:
            log("  " + b)
        raise SystemExit(1)
    log("\nall accounting and timing gates passed")

    # Padding, which is the whole point of doing this.
    steps, per_step = 7242, 8
    log(f"\nat {steps} steps x {per_step}: single-utterance = "
        f"{steps*per_step/len(rows):.2f} epochs, packs = "
        f"{steps*per_step/len(packs):.2f} epochs over the same audio")
    log(f"padding: single-utterance {100*(1-2.79/30):.0f}%, "
        f"packs {100*(1-sum(dur)/len(dur)/30):.0f}%")


if __name__ == "__main__":
    main()
