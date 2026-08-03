#!/usr/bin/env python3
"""Does cutting the audio at speaker changes help? Build the three arms and decode them.

Design frozen in `docs/specs/segmentation-experiment-preregistration.md`.

The whole experiment turns on arm 3. Arm 2 cuts at speaker changes; arm 3 cuts the audio
into chunks of the SAME lengths in the SAME order but places every boundary away from any
speaker change. Without it, "shorter chunks decode better" and "cutting at speaker changes
decodes better" are the same number, and only the second one is a finding.

Phase 1 (local, free): build the chunk plans from precision-2 turns.
Phase 2 (GPU pod): decode every chunk of every arm and stitch.

Usage:
  SC=~/.cache/oc-overlap python eval/controlled_eval/exp_segmentation.py plan
  MIX=... MODELS=... python eval/controlled_eval/exp_segmentation.py run
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("/home/harold/opencouncil-fine-tuning")
AUDIO = ROOT / "data/asr/audio"
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
PLAN = SC / "segmentation_plan.json"

MIN_TURN = 0.25        # frozen: turns shorter than this are not speaker changes
COLLAPSE = 2.0         # frozen: changes closer than this collapse to one boundary
MIN_CHUNK, MAX_CHUNK = 5.0, 30.0
PAD = 0.75             # frozen: context padding on each side
AWAY = 1.0             # frozen: arm 3 boundaries stay this far from any speaker change


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------------------------------------------------------ phase 1: planning
def change_points(turns, dur):
    """Times where the active speaker changes, after the frozen conventions."""
    t = sorted((x for x in turns if float(x["end"]) - float(x["start"]) >= MIN_TURN),
               key=lambda z: float(z["start"]))
    raw = []
    for a, b in zip(t, t[1:]):
        if a["speaker"] != b["speaker"]:
            # the change is somewhere between a's end and b's start; take the midpoint
            raw.append(max(0.0, min(dur, (float(a["end"]) + float(b["start"])) / 2)))
    raw.sort()
    out = []
    for x in raw:
        if not out or x - out[-1] >= COLLAPSE:
            out.append(x)
    return out


def chunks_from_boundaries(bounds, dur):
    """Boundaries -> chunks, enforcing the frozen min/max length."""
    pts = [0.0] + [b for b in bounds if 0 < b < dur] + [dur]
    out = []
    start = pts[0]
    for p in pts[1:]:
        if p - start < MIN_CHUNK and p != dur:
            continue                                   # too short, keep accumulating
        while p - start > MAX_CHUNK:
            out.append((start, start + MAX_CHUNK))
            start += MAX_CHUNK
        if p - start >= MIN_CHUNK or not out:
            out.append((start, p))
            start = p
    if start < dur - 0.01:
        if out and dur - start < MIN_CHUNK:
            out[-1] = (out[-1][0], dur)
        else:
            out.append((start, dur))
    return out


def shifted_chunks(n_chunks, changes, dur, grid=0.25):
    """Same NUMBER of chunks, boundaries placed as far from speaker changes as possible.

    The first attempt kept arm 2's chunk lengths exactly and searched a global offset.
    That fails: with six boundaries in two minutes and changes scattered through, no
    single offset avoids them all, and it left 182 of 232 windows with a margin of zero —
    a control that cuts at speaker changes too, which controls nothing.

    So the constraint that is preserved is the chunk COUNT and the legal length range,
    and the objective is the largest achievable minimum distance from any change.
    Binary search on that margin, feasibility by dynamic programming over a 0.25 s grid.
    Deterministic, and the achieved margin is recorded per window so a window where the
    control is weak can be identified rather than hidden.
    """
    import numpy as np
    n_pts = int(dur / grid) + 1
    t = np.arange(n_pts) * grid
    if changes:
        dist = np.min(np.abs(t[:, None] - np.array(changes)[None, :]), axis=1)
    else:
        dist = np.full(n_pts, 1e9)
    lo_i, hi_i = int(MIN_CHUNK / grid), int(MAX_CHUNK / grid)

    def feasible(margin):
        """Can [0,dur] be split into exactly n_chunks legal chunks, all boundaries clear?"""
        ok = dist >= margin
        reach = np.zeros((n_chunks + 1, n_pts), dtype=bool)
        reach[0, 0] = True
        for k in range(1, n_chunks + 1):
            cs = np.concatenate([[0], np.cumsum(reach[k - 1].astype(np.int32))])
            for i in range(n_pts):
                a, b = max(0, i - hi_i), i - lo_i
                if b < a:
                    continue
                if cs[b + 1] - cs[a] > 0:
                    reach[k, i] = True
            if k < n_chunks:
                reach[k] &= ok            # interior boundaries must clear the margin
        return reach[n_chunks, n_pts - 1], reach

    lo, hi, best = 0.0, float(min(MAX_CHUNK, dur)), None
    for _ in range(18):
        mid = (lo + hi) / 2
        f, reach = feasible(mid)
        if f:
            lo, best = mid, (mid, reach)
        else:
            hi = mid
    if best is None:
        f, reach = feasible(0.0)
        if not f:
            return [[0.0, dur]], 0.0
        best = (0.0, reach)
    margin, reach = best

    # walk the DP back to an actual segmentation
    bounds, i = [], n_pts - 1
    for k in range(n_chunks, 0, -1):
        if k == 1:
            bounds.append(0)
            break
        found = None
        for j in range(max(0, i - hi_i), i - lo_i + 1):
            if reach[k - 1, j] and dist[j] >= margin:
                found = j
                break
        if found is None:
            found = max(0, i - hi_i)
        bounds.append(found)
        i = found
    bounds = sorted(set(bounds + [0, n_pts - 1]))
    out = [[float(t[a]), float(t[b])] for a, b in zip(bounds, bounds[1:])]
    achieved = min((float(dist[b]) for b in bounds[1:-1]), default=1e9)
    return out, achieved


def plan():
    import bench_data as B
    p2 = {k: v for k, v in json.loads((SC / "precision2_corpus.json").read_text()).items()
          if "feat" in v}
    report = B.load_report()
    by_id = {it["itemId"]: it for it in report["items"]}
    items = B.common_items(report, B.provider_ids(report))

    out, skipped = [], 0
    for it in items:
        iid = it["item_id"]
        src = AUDIO / f"{it['city_id']}__{it['meeting_id']}.mp3"
        if iid not in p2 or not src.exists():
            skipped += 1
            continue
        raw = by_id[iid]
        dur = p2[iid]["measured_dur"]
        ch = change_points(p2[iid]["turns"], dur)
        a2 = chunks_from_boundaries(ch, dur)
        a3, margin = shifted_chunks(len(a2), ch, dur)
        out.append({
            "item_id": iid, "city_id": it["city_id"], "meeting_id": it["meeting_id"],
            "ref": it["ref"], "audio": str(src),
            "abs_start": raw["startSec"], "dur": dur,
            "n_changes": len(ch), "turn_rate_per_min": len(ch) / (dur / 60),
            "arm1": [[0.0, dur]], "arm2": [list(x) for x in a2],
            "arm3": [list(x) for x in a3], "arm3_margin_sec": round(margin, 3),
        })
    n2 = sum(len(x["arm2"]) for x in out)
    n3 = sum(len(x["arm3"]) for x in out)
    bad = sum(1 for x in out if x["arm3_margin_sec"] < AWAY)
    log(f"{len(out)} windows planned, {skipped} skipped")
    log(f"chunks: arm1 {len(out)}, arm2 {n2}, arm3 {n3}")
    log(f"windows where arm3 could not keep {AWAY}s from every change: {bad} "
        f"(reported, not dropped)")
    PLAN.write_text(json.dumps({"pad": PAD, "min_chunk": MIN_CHUNK,
                                "max_chunk": MAX_CHUNK, "away": AWAY,
                                "items": out}, ensure_ascii=False))
    log(f"-> {PLAN}")


# ------------------------------------------------------------------ phase 2: decoding
def dedupe(prev: str, nxt: str, max_overlap: int = 12) -> str:
    """Drop the repeated words the padding creates at a seam."""
    a, b = prev.split(), nxt.split()
    best = 0
    for k in range(min(max_overlap, len(a), len(b)), 0, -1):
        if [w.lower() for w in a[-k:]] == [w.lower() for w in b[:k]]:
            best = k
            break
    return " ".join(b[best:])


def run():
    import numpy as np
    plan_ = json.loads(Path(os.environ.get("PLAN", PLAN)).read_text())
    models_dir = Path(os.environ.get("MODELS", "/workspace/models"))
    out_json = Path(os.environ.get("OUT_JSON", "/workspace/segmentation_hyps.json"))
    audio_dir = Path(os.environ.get("AUDIO_DIR", "/workspace/audio"))
    systems = {"finetune": str(models_dir / "ct2"), "whisper-large-v3": "large-v3"}
    arms = [a for a in os.environ.get("ARMS", "arm1,arm2,arm3").split(",") if a]

    jobs = [(s, it["item_id"], a) for s in systems for it in plan_["items"] for a in arms]
    random.Random(20260803).shuffle(jobs)
    by_id = {it["item_id"]: it for it in plan_["items"]}
    done = json.loads(out_json.read_text()) if out_json.exists() else {}
    log(f"{len(jobs)} window-decodes, {len(done)} cached")

    from faster_whisper import WhisperModel
    DEC = dict(language="el", beam_size=1, temperature=0.0, best_of=1,
               condition_on_previous_text=False, vad_filter=False)

    for sysname, spec in systems.items():
        todo = [j for j in jobs if j[0] == sysname and f"{j[0]}|{j[1]}|{j[2]}" not in done]
        if not todo:
            continue
        log(f"loading {sysname}")
        model = WhisperModel(spec, device="cuda", compute_type="float16")
        t0 = time.time()
        for i, (_, iid, arm) in enumerate(todo, 1):
            it = by_id[iid]
            wav = audio_dir / f"{iid}.wav"
            pieces = []
            for s, e in it[arm]:
                s0, e0 = max(0.0, s - plan_["pad"]), min(it["dur"], e + plan_["pad"])
                tmp = Path("/tmp") / f"chunk_{os.getpid()}.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.3f}",
                                "-t", f"{e0 - s0:.3f}", "-i", str(wav), "-ac", "1",
                                "-ar", "16000", str(tmp)], check=True)
                segs, _ = model.transcribe(str(tmp), **DEC)
                pieces.append(" ".join(x.text.strip() for x in segs).strip())
            text = pieces[0] if pieces else ""
            for p in pieces[1:]:
                text = (text + " " + dedupe(text, p)).strip()
            done[f"{sysname}|{iid}|{arm}"] = text
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"  {sysname} {i}/{len(todo)} ({el / i:.1f}s each, "
                    f"eta {(len(todo) - i) * el / i / 60:.0f}min)")
                tmp = out_json.with_suffix(".part")
                tmp.write_text(json.dumps(done, ensure_ascii=False))
                tmp.replace(out_json)
        del model
    tmp = out_json.with_suffix(".part")
    tmp.write_text(json.dumps(done, ensure_ascii=False))
    tmp.replace(out_json)
    log(f"{len(done)} decodes -> {out_json}")


if __name__ == "__main__":
    {"plan": plan, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "plan"]()
