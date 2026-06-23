"""Enlarged, frozen VAL set for the error-division experiment (Design A).

Separate from the train build: writes data/asr/val_manifest.jsonl and does NOT
touch data/asr/manifest.jsonl. Reuses the decode-once-then-slice + clip-validate
path from prepare_asr.py.

Two disjoint slices, never mixed into one denominator:
  val_corr_cat : ALL /api/export corrections in VAL_CITIES (orestiada, argos),
                 reference = final_after_text, error_categories kept. This is the
                 ONLY slice with categories -> the per-category matrix val.
  val_reg      : no-edit utts (lastModifiedBy is null) from the humanReview val
                 meetings, capped per meeting to spread across meetings, deduped
                 vs corrections. Regression / overall-WER sanity only.

Leakage guard: VAL_CITIES are disjoint from the train cities, so no clip can be
shared; we still assert (city, meeting, utterance_id) disjointness against the
frozen train manifest and log any reference-text-hash collisions for the record.

Why only ~160 categorized: the /api/export UI-included corrections are the only
source with error_categories. The larger ~9.9k meeting-JSON corrected pool has no
categories, so it cannot feed the per-category matrix. See
docs/specs/error-division.md.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time

import prepare_asr as P  # reuse fetch_meeting_json/noedit_utts/download_mp3/decode_pcm/slice_clip/write_clip

VAL_MANIFEST = P.ASR / "val_manifest.jsonl"
VAL_STATS = P.ASR / "val_stats.json"
VAL_LOG = P.ROOT / "data" / "reports" / "error-division" / "prepare_val.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    VAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with VAL_LOG.open("a") as f:
        f.write(line + "\n")


def text_hash(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-reg-target", type=int, default=400,
                    help="global cap on no-edit regression clips")
    ap.add_argument("--val-reg-per-meeting", type=int, default=20,
                    help="cap no-edit clips taken per meeting (spread across meetings)")
    args = ap.parse_args()

    VAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    VAL_LOG.write_text("")
    for d in (P.AUDIO, P.CLIPS, P.MJSON):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rng = __import__("numpy").random.default_rng(P.SEED)

    # ---- export corrections in the val cities (unique by composite id) ----
    rows = [json.loads(l) for l in P.EXPORT.open()]
    seen, corr = set(), []
    for r in rows:
        key = (r["city_id"], r["meeting_id"], r["utterance_id"])
        if key in seen:
            continue
        seen.add(key)
        if r["city_id"] in P.VAL_CITIES:
            corr.append(r)
    log(f"export corrections in {sorted(P.VAL_CITIES)}: {len(corr)}")

    meta = json.loads(P.META.read_text())
    hr = {k for k, v in meta.items() if v.get("humanReview")}

    by_mtg = collections.defaultdict(list)
    for r in corr:
        by_mtg[(r["city_id"], r["meeting_id"])].append(r)
    log(f"val meetings with corrections: {len(by_mtg)} "
        f"({sum(1 for m in by_mtg if f'{m[0]} {m[1]}' in hr)} humanReview)")

    # ---- TRAIN-SPLIT keys for the leakage guard ----
    # The old manifest.jsonl holds every split (train, train_noedit, val_corr,
    # val_reg). Only the actual training rows matter for leakage; the val rows in
    # it are the same clips we are rebuilding here, not contamination.
    TRAIN_SPLITS = {"train", "train_noedit"}
    train_keys, train_texthashes, train_cities = set(), set(), set()
    if P.MANIFEST.exists():
        for l in P.MANIFEST.open():
            t = json.loads(l)
            if t["split"] not in TRAIN_SPLITS:
                continue
            train_keys.add((t["city"], t["meeting"], t["utterance_id"]))
            train_texthashes.add(text_hash(t["text"]))
            train_cities.add(t["city"])
    overlap_cities = P.VAL_CITIES & train_cities
    if overlap_cities:
        raise SystemExit(f"LEAKAGE: val cities also in the TRAIN split: {overlap_cities}")
    log(f"train-split keys={len(train_keys)} across {len(train_cities)} cities "
        f"(disjoint from val: OK)")

    manifest = []
    val_reg_budget = args.val_reg_target

    for mi, m in enumerate(sorted(by_mtg), 1):
        city, meeting = m
        key = f"{city} {meeting}"
        is_hr = key in hr
        corr_ids = {r["utterance_id"] for r in by_mtg[m]}

        plan = []  # (uid, start, end, text, source, error_categories, split)
        for r in by_mtg[m]:
            plan.append((r["utterance_id"], float(r["start"]), float(r["end"]),
                         r["final_after_text"], "correction",
                         r.get("error_categories") or [], "val_corr_cat"))

        if is_hr and val_reg_budget > 0:
            d = P.fetch_meeting_json(city, meeting)
            if d is not None:
                cand = [t for t in P.noedit_utts(d) if t[0] not in corr_ids]
                rng.shuffle(cand)
                take = cand[: min(args.val_reg_per_meeting, val_reg_budget)]
                for uid, s, e, txt in take:
                    plan.append((uid, s, e, txt, "no_edit", [], "val_reg"))
        elif not is_hr:
            log(f"  {key}: NOT humanReview -> corrections only (no trustworthy no-edit)")

        url = by_mtg[m][0]["audio_url"]
        mp3 = P.download_mp3(url, city, meeting)
        if mp3 is None:
            log(f"  {key}: download FAILED -> skipped (corrections lost: {len(corr_ids)})")
            continue
        audio_sha = P.sha256_file(mp3)[:16]
        pcm = P.decode_pcm(mp3)
        if pcm is None:
            log(f"  {key}: decode FAILED -> skipped")
            continue
        total = len(pcm)
        dec_dur = total / P.SR
        kept = dropped = reg_kept = 0
        for uid, s, e, txt, source, ecats, split in plan:
            raw_dur = e - s
            if raw_dur < P.MIN_DUR or raw_dur > P.MAX_DUR or s < 0 or e > dec_dur + 1.0:
                dropped += 1
                continue
            arr, _ = P.slice_clip(pcm, s, e, total)
            if arr is None:
                dropped += 1
                continue
            rel = P.write_clip(arr, city, meeting, uid)
            if rel is None:
                dropped += 1
                continue
            # leakage assertion (defensive: cities already disjoint)
            if (city, meeting, uid) in train_keys:
                raise SystemExit(f"LEAKAGE: val clip {city}/{meeting}/{uid} is in train manifest")
            if split == "val_reg":
                reg_kept += 1
            manifest.append({
                "utterance_id": uid, "split": split, "city": city, "meeting": meeting,
                "start": round(s, 3), "end": round(e, 3), "dur": round(raw_dur, 3),
                "audio_sha": audio_sha, "text": txt, "source": source,
                "error_categories": ecats, "clip_path": rel,
            })
            kept += 1
        val_reg_budget -= reg_kept
        del pcm
        log(f"[{mi}/{len(by_mtg)}] {key} hr={is_hr} kept={kept} (reg={reg_kept}) "
            f"dropped={dropped} reg_budget_left={val_reg_budget} elapsed={time.time()-t_start:.0f}s")

    # ---- text-hash collisions vs train (informational, not leakage) ----
    val_corr_texthashes = {text_hash(r["text"]) for r in manifest if r["split"] == "val_corr_cat"}
    collisions = val_corr_texthashes & train_texthashes
    if collisions:
        log(f"NOTE: {len(collisions)} val_corr reference texts also appear verbatim in train "
            f"(short/shared strings; cities are disjoint so not clip leakage)")

    with VAL_MANIFEST.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_split = collections.Counter(r["split"] for r in manifest)
    n_meet = collections.defaultdict(set)
    cat_clips = collections.Counter()
    cat_meet = collections.defaultdict(set)
    for r in manifest:
        n_meet[r["split"]].add((r["city"], r["meeting"]))
        if r["split"] == "val_corr_cat":
            for c in r["error_categories"]:
                cat_clips[c] += 1
                cat_meet[c].add((r["city"], r["meeting"]))
    stats = {
        "n_clips": len(manifest),
        "by_split": dict(by_split),
        "meetings_by_split": {k: len(v) for k, v in n_meet.items()},
        "val_corr_per_category": {
            c: {"n_clips": cat_clips[c], "n_meetings": len(cat_meet[c]),
                "directional": cat_clips[c] < 30 or len(cat_meet[c]) < 5}
            for c in sorted(cat_clips, key=lambda x: -cat_clips[x])
        },
        "val_cities": sorted(P.VAL_CITIES),
        "params": vars(args),
        "build_seconds": round(time.time() - t_start, 1),
        "sr": P.SR, "pad_s": P.PAD_S, "seed": P.SEED,
    }
    VAL_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    log(f"DONE clips={len(manifest)} by_split={dict(by_split)} "
        f"meetings={stats['meetings_by_split']} in {stats['build_seconds']:.0f}s")
    log(f"  -> {VAL_MANIFEST}\n  -> {VAL_STATS}")


if __name__ == "__main__":
    main()
