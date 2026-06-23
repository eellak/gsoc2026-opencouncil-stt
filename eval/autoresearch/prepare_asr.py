"""Fixed data-prep for the tiny-Whisper CPU auto-research loop (Track 2).

Analog of karpathy/autoresearch `prepare.py`: one-time, NOT edited by the loop.
Builds a *capped* (1-hour-budget) frozen manifest + 16 kHz mono clip cache from:

  1. /api/export             -> ~1963 included corrections (target = final_after_text)
  2. meeting JSON            -> no-edit utts (lastModifiedBy is null) for the
                               regression set + optional train backbone
  3. data/eval/meeting_meta.json -> humanReview gate for trustworthy no-edit

Splits (city-disjoint, never random):
  val_corr  = corrections in VAL_CITIES (orestiada, argos)
  val_reg   = no-edit utts from the SAME val-city humanReview meetings (regression guard)
  train     = corrections in other cities
  train_noedit = no-edit utts from train-city humanReview meetings (backbone, for a sweep axis)

Correctness (folded in from Codex review, trimmed to the 1h budget):
  - decode each mp3 to PCM ONCE, slice in memory (no compressed seeks)
  - half-open padded slices floor/ceil, clamped to actual decoded frames
  - filter on UNPADDED duration; drop non-positive / out-of-range spans
  - composite identity (city, meeting, utterance_id); corrections excluded from
    the no-edit pool; no utterance crosses the train/val boundary
  - keep source mp3s; validate every written clip (mono/16k/finite/frames)
  - frozen manifest.jsonl + dataset_stats.json; source mp3 sha recorded

Caps are CLI-tunable so the whole build fits the wall-clock budget. What gets
dropped is logged, never silent.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent.parent
ASR = ROOT / "data" / "asr"
EXPORT = ASR / "export.jsonl"
META = ROOT / "data" / "eval" / "meeting_meta.json"
AUDIO = ASR / "audio"
CLIPS = ASR / "clips"
MJSON = ASR / "meeting_json"
MANIFEST = ASR / "manifest.jsonl"
STATS = ASR / "dataset_stats.json"
LOG = ROOT / "data" / "reports" / "finetune-research" / "prepare.log"

VAL_CITIES = {"orestiada", "argos"}
MEETING_API = "https://opencouncil.gr/api/cities/{city}/meetings/{meeting}"

SR = 16000
PAD_S = 0.2
MIN_DUR = 1.0
MAX_DUR = 30.0
SEED = 1234


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_meeting_json(city: str, meeting: str) -> dict | None:
    cache = MJSON / f"{city}__{meeting}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = MEETING_API.format(city=city, meeting=meeting)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "oc-asr/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.load(r)
        cache.write_text(json.dumps(d, ensure_ascii=False))
        return d
    except Exception as e:  # noqa: BLE001
        log(f"  meeting JSON FAIL {city}/{meeting}: {type(e).__name__} {str(e)[:60]}")
        return None


def noedit_utts(d: dict):
    """Yield (utterance_id, start, end, text) for no-edit utts in a meeting JSON."""
    for seg in d.get("transcript") or []:
        for u in seg.get("utterances") or []:
            if u.get("lastModifiedBy") is not None:
                continue
            try:
                s = float(u.get("startTimestamp")); e = float(u.get("endTimestamp"))
            except (TypeError, ValueError):
                continue
            txt = (u.get("text") or "").strip()
            if txt:
                yield u.get("id"), s, e, txt


def download_mp3(url: str, city: str, meeting: str) -> Path | None:
    dst = AUDIO / f"{city}__{meeting}.mp3"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    try:
        t0 = time.time()
        subprocess.run(["curl", "-sS", "-m", "600", "-o", str(dst), url], check=True)
        log(f"  downloaded {city}/{meeting} {dst.stat().st_size/1e6:.0f}MB in {time.time()-t0:.0f}s")
        return dst
    except Exception as e:  # noqa: BLE001
        log(f"  download FAIL {city}/{meeting}: {type(e).__name__} {str(e)[:60]}")
        return None


def decode_pcm(mp3: Path) -> np.ndarray | None:
    """Decode whole mp3 -> 16 kHz mono float32 PCM, once."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(mp3),
             "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
            check=True, capture_output=True).stdout
        return np.frombuffer(out, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        log(f"  decode FAIL {mp3.name}: {type(e).__name__} {str(e)[:60]}")
        return None


def slice_clip(pcm: np.ndarray, start: float, end: float, total: int):
    """Half-open padded slice clamped to decoded frames. Returns (arr, raw_dur)."""
    raw_dur = end - start
    a = max(0, int(np.floor(start * SR)) - int(PAD_S * SR))
    b = min(total, int(np.ceil(end * SR)) + int(PAD_S * SR))
    if b <= a:
        return None, raw_dur
    return pcm[a:b], raw_dur


def write_clip(arr: np.ndarray, city: str, meeting: str, uid: str) -> str | None:
    d = CLIPS / city / meeting
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{uid}.wav"
    tmp = d / f".{uid}.tmp.wav"
    sf.write(tmp, arr, SR, subtype="PCM_16")
    # validate
    info = sf.info(tmp)
    if info.samplerate != SR or info.channels != 1 or info.frames != len(arr):
        log(f"  clip VALIDATE FAIL {uid}: sr={info.samplerate} ch={info.channels} frames={info.frames}")
        tmp.unlink(missing_ok=True)
        return None
    if not np.all(np.isfinite(arr)) or float(np.abs(arr).max()) < 1e-6:
        log(f"  clip SIGNAL FAIL {uid}: nonfinite or silent")
        tmp.unlink(missing_ok=True)
        return None
    tmp.rename(p)
    return str(p.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-meetings", type=int, default=14, help="# high-yield train meetings")
    ap.add_argument("--val-meetings-per-city", type=int, default=3)
    ap.add_argument("--val-reg-per-city", type=int, default=40)
    ap.add_argument("--train-noedit-cap", type=int, default=1500)
    args = ap.parse_args()

    LOG.write_text("")  # fresh
    rng = np.random.default_rng(SEED)
    t_start = time.time()
    for d in (AUDIO, CLIPS, MJSON):
        d.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in EXPORT.open()]
    meta = json.loads(META.read_text())
    hr = {k for k, v in meta.items() if v.get("humanReview")}

    # ---- assert correction utterance_id uniqueness (composite) ----
    seen = set()
    corr = []
    for r in rows:
        key = (r["city_id"], r["meeting_id"], r["utterance_id"])
        if key in seen:
            continue
        seen.add(key)
        corr.append(r)
    log(f"export rows={len(rows)} unique corrections={len(corr)}")

    # ---- drop denylisted (unreviewed, <5% human-edit) meetings ----
    sys.path.insert(0, str(ROOT))
    from eval.exclusions import load_excluded_keys  # noqa: E402
    excluded = load_excluded_keys()
    if excluded:
        before = len(corr)
        corr = [r for r in corr if (r["city_id"], r["meeting_id"]) not in excluded]
        log(f"denylist: dropped {before - len(corr)} correction rows from "
            f"{len(excluded)} excluded meetings")

    # ---- pick meetings under the cap ----
    by_mtg = collections.defaultdict(list)
    for r in corr:
        by_mtg[(r["city_id"], r["meeting_id"])].append(r)

    val_mtgs, train_mtgs = [], []
    for city in sorted(VAL_CITIES):
        ms = sorted([m for m in by_mtg if m[0] == city], key=lambda m: -len(by_mtg[m]))
        val_mtgs += ms[: args.val_meetings_per_city]
    train_cands = sorted(
        [m for m in by_mtg if m[0] not in VAL_CITIES], key=lambda m: -len(by_mtg[m]))
    train_mtgs = train_cands[: args.train_meetings]
    chosen = val_mtgs + train_mtgs
    log(f"chosen meetings: {len(val_mtgs)} val + {len(train_mtgs)} train = {len(chosen)}")
    log(f"  val={val_mtgs}")
    log(f"  train cities={collections.Counter(c for c,_ in train_mtgs)}")

    # ---- build per-meeting work plan ----
    manifest = []
    corr_ids_by_mtg = {m: {r['utterance_id'] for r in by_mtg[m]} for m in chosen}
    train_noedit_budget = args.train_noedit_cap

    for mi, m in enumerate(chosen, 1):
        city, meeting = m
        key = f"{city} {meeting}"
        is_val = city in VAL_CITIES
        is_hr = key in hr
        # plan utts: corrections (always) + no-edit (only if humanReview)
        plan = []  # (uid, start, end, text, source, error_categories, split)
        for r in by_mtg[m]:
            split = "val_corr" if is_val else "train"
            plan.append((r["utterance_id"], float(r["start"]), float(r["end"]),
                         r["final_after_text"], "correction",
                         r.get("error_categories") or [], split))
        if is_hr:
            d = fetch_meeting_json(city, meeting)
            if d is not None:
                cand = [t for t in noedit_utts(d) if t[0] not in corr_ids_by_mtg[m]]
                rng.shuffle(cand)
                if is_val:
                    take = cand[: args.val_reg_per_city]
                    for uid, s, e, txt in take:
                        plan.append((uid, s, e, txt, "no_edit", [], "val_reg"))
                else:
                    per = max(0, min(len(cand), train_noedit_budget // max(1, len(train_mtgs))))
                    take = cand[:per]
                    train_noedit_budget -= len(take)
                    for uid, s, e, txt in take:
                        plan.append((uid, s, e, txt, "no_edit", [], "train_noedit"))
        else:
            log(f"  {key}: NOT humanReview -> corrections only (no trustworthy no-edit)")

        # download + decode once
        url = by_mtg[m][0]["audio_url"]
        mp3 = download_mp3(url, city, meeting)
        if mp3 is None:
            continue
        audio_sha = sha256_file(mp3)[:16]
        pcm = decode_pcm(mp3)
        if pcm is None:
            continue
        total = len(pcm)
        dec_dur = total / SR
        kept = dropped = 0
        for uid, s, e, txt, source, ecats, split in plan:
            raw_dur = e - s
            if raw_dur < MIN_DUR or raw_dur > MAX_DUR or s < 0 or e > dec_dur + 1.0:
                dropped += 1
                continue
            arr, _ = slice_clip(pcm, s, e, total)
            if arr is None:
                dropped += 1
                continue
            rel = write_clip(arr, city, meeting, uid)
            if rel is None:
                dropped += 1
                continue
            manifest.append({
                "utterance_id": uid, "split": split, "city": city, "meeting": meeting,
                "start": round(s, 3), "end": round(e, 3), "dur": round(raw_dur, 3),
                "audio_sha": audio_sha, "text": txt, "source": source,
                "error_categories": ecats, "clip_path": rel,
            })
            kept += 1
        del pcm
        log(f"[{mi}/{len(chosen)}] {key} hr={is_hr} kept={kept} dropped={dropped} "
            f"decoded={dec_dur/60:.0f}min elapsed={time.time()-t_start:.0f}s")

    # ---- freeze manifest ----
    with MANIFEST.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- stats ----
    by_split = collections.Counter(r["split"] for r in manifest)
    dur_by_split = collections.defaultdict(float)
    for r in manifest:
        dur_by_split[r["split"]] += r["dur"]
    cat = collections.Counter()
    for r in manifest:
        if r["split"] == "train":
            for c in r["error_categories"]:
                cat[c] += 1
    stats = {
        "n_clips": len(manifest),
        "by_split": dict(by_split),
        "minutes_by_split": {k: round(v / 60, 1) for k, v in dur_by_split.items()},
        "train_error_categories": dict(cat.most_common()),
        "val_cities": sorted(VAL_CITIES),
        "params": vars(args),
        "build_seconds": round(time.time() - t_start, 1),
        "sr": SR, "pad_s": PAD_S, "min_dur": MIN_DUR, "max_dur": MAX_DUR, "seed": SEED,
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    log(f"DONE clips={len(manifest)} by_split={dict(by_split)} "
        f"minutes={stats['minutes_by_split']} in {stats['build_seconds']:.0f}s")
    log(f"  -> {MANIFEST}\n  -> {STATS}")


if __name__ == "__main__":
    main()
