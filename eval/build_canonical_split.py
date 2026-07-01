"""Build the canonical train/val split (manifest-only) for the Greek ASR set.

Self-contained, deterministic, text-only: audio stays referenced by `audio_url`
+ `start`/`end` (no clip download/decode). No test split (test = temporal
June-2026+ hold-out, currently empty). See
docs/specs/dataset-split-and-publish-plan.md and the 2026-06-23 mentor sync.

Locked decisions (do not re-litigate here):
  - VAL = Argos + Orestiada whole cities        -> stratum unseen_city
  - VAL top-up = md5-ordered whole-speaker hold-out from the mixed cities until
    total val >= 20% of trusted minutes          -> stratum unseen_speaker
    (only speakers with >= 10 trusted min eligible; tiny speakers stay in train)
  - TRAIN = everything else trusted
  - speaker-exclusive: every person_id lives in exactly one split
  - trust gate: drop denylisted meetings; corrections = all export.jsonl includes
    (tier human_verified); no-edit backbone = lastModifiedBy-null utts from
    humanReview=true meetings only (tier no_edit), minus any correction uid
  - duration gate: keep only 1.0s <= end-start <= 30.0s
  - determinism: SEED=1234, md5(person_id) only (no RNG). Same inputs -> same out.

Reuses fetch_meeting_json / noedit_utts from eval/autoresearch/prepare_asr.py.
Run: .venv-eval/bin/python eval/build_canonical_split.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.exclusions import load_excluded_keys  # noqa: E402

# prepare_asr imports soundfile at module top for its clip-writing path, which we
# do not use (manifest-only). Stub it so we can reuse its proven, pure-stdlib
# fetch_meeting_json / noedit_utts without pulling in libsndfile.
import types  # noqa: E402
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
from eval.autoresearch.prepare_asr import (  # noqa: E402
    fetch_meeting_json,
    noedit_utts,
)

ASR = ROOT / "data" / "asr"
EXPORT = ASR / "export.jsonl"
META = ROOT / "data" / "eval" / "meeting_meta.json"
SPEAKERS = ROOT / "data" / "eval" / "speakers.parquet"

SPLIT_CSV = ASR / "canonical_split.csv"
TRAIN_JSONL = ASR / "train.jsonl"
VAL_JSONL = ASR / "val.jsonl"
STATS_JSON = ASR / "split_stats.json"

VAL_CITIES = {"argos", "orestiada"}
SEED = 1234
MIN_DUR = 1.0
MAX_DUR = 30.0
VAL_TARGET_FRAC = 0.20
VAL_BAND = (0.18, 0.22)
MIN_SPEAKER_MIN = 10.0


def log(msg: str) -> None:
    print(msg, flush=True)


def span_ok(start, end) -> bool:
    """True iff start/end finite, start >= 0, and 1.0s <= (end-start) <= 30.0s."""
    try:
        s = float(start)
        e = float(end)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(s) and math.isfinite(e)):
        return False
    if s < 0:
        return False
    d = e - s
    return MIN_DUR <= d <= MAX_DUR


def md5_hex(person_id: str) -> str:
    return hashlib.md5(person_id.encode("utf-8")).hexdigest()


def load_person_map() -> dict[str, str | None]:
    """utterance_id -> normalized person_id (None if missing / null / blank)."""
    pdf = pd.read_parquet(SPEAKERS, columns=["utterance_id", "person_id"])
    out: dict[str, str | None] = {}
    for uid, pid in zip(pdf["utterance_id"], pdf["person_id"]):
        if pid is None or (isinstance(pid, float) and math.isnan(pid)):
            out[str(uid)] = None
        else:
            s = str(pid).strip()
            out[str(uid)] = s or None
    return out


def load_corrections(excluded: set[tuple[str, str]]) -> list[dict]:
    """Corrections (tier human_verified) after denylist + duration gate."""
    by_uid: dict[str, dict] = {}
    with EXPORT.open() as f:
        for line in f:
            r = json.loads(line)
            uid = r["utterance_id"]
            if uid in by_uid:
                prev = by_uid[uid]
                same = (
                    prev["final_after_text"] == r["final_after_text"]
                    and (prev["city_id"], prev["meeting_id"])
                    == (r["city_id"], r["meeting_id"])
                )
                if not same:
                    raise SystemExit(
                        f"conflicting duplicate correction utterance_id {uid}"
                    )
                continue
            by_uid[uid] = r

    kept: list[dict] = []
    dropped_denylist = dropped_span = 0
    for uid in sorted(by_uid):
        r = by_uid[uid]
        key = (r["city_id"], r["meeting_id"])
        if key in excluded:
            dropped_denylist += 1
            continue
        if not span_ok(r["start"], r["end"]):
            dropped_span += 1
            continue
        text = (r.get("final_after_text") or "").strip()
        if not text:
            dropped_span += 1
            continue
        kept.append({
            "utterance_id": uid,
            "city_id": r["city_id"],
            "meeting_id": r["meeting_id"],
            "start": round(float(r["start"]), 3),
            "end": round(float(r["end"]), 3),
            "dur": round(float(r["end"]) - float(r["start"]), 3),
            "audio_url": r.get("audio_url"),
            "text": text,
            "tier": "human_verified",
            "source": "correction",
            "error_categories": r.get("error_categories") or [],
        })
    log(f"corrections: {len(by_uid)} unique -> {len(kept)} kept "
        f"(dropped denylist={dropped_denylist} span/empty={dropped_span})")
    return kept


def load_noedit(
    hr_meetings: list[tuple[str, str]],
    correction_uids: set[str],
) -> tuple[list[dict], int]:
    """No-edit backbone (tier no_edit) from humanReview meeting JSON."""
    kept: list[dict] = []
    fetched = 0
    dropped_span = dropped_dup = 0
    for city, meeting in hr_meetings:
        cache = ASR / "meeting_json" / f"{city}__{meeting}.json"
        was_cached = cache.exists()
        d = fetch_meeting_json(city, meeting)
        if not was_cached and cache.exists():
            fetched += 1
        if d is None:
            log(f"  no-edit: meeting JSON unavailable {city}/{meeting}")
            continue
        audio_url = ((d.get("meeting") or {}).get("audioUrl"))
        for uid, s, e, txt in noedit_utts(d):
            if uid in correction_uids:
                dropped_dup += 1
                continue
            if not span_ok(s, e):
                dropped_span += 1
                continue
            text = (txt or "").strip()
            if not text:
                dropped_span += 1
                continue
            kept.append({
                "utterance_id": uid,
                "city_id": city,
                "meeting_id": meeting,
                "start": round(float(s), 3),
                "end": round(float(e), 3),
                "dur": round(float(e) - float(s), 3),
                "audio_url": audio_url,
                "text": text,
                "tier": "no_edit",
                "source": "no_edit",
                "error_categories": [],
            })
    log(f"no-edit: {len(kept)} kept (fetched {fetched} uncached; "
        f"dropped span/empty={dropped_span} already-correction={dropped_dup})")
    return kept, fetched


def choose_holdout_speakers(
    pool: list[dict],
    total_min: float,
) -> tuple[set[str], dict]:
    """md5-ordered whole-speaker hold-out from mixed cities until val>=20%."""
    unseen_city_min = sum(
        r["dur"] for r in pool if r["city_id"] in VAL_CITIES) / 60.0
    target_min = VAL_TARGET_FRAC * total_min

    mixed_speaker_min: dict[str, float] = collections.defaultdict(float)
    for r in pool:
        if r["city_id"] in VAL_CITIES:
            continue
        pid = r["person_id"]
        if pid is not None:
            mixed_speaker_min[pid] += r["dur"] / 60.0

    eligible = sorted(
        (pid for pid, m in mixed_speaker_min.items() if m >= MIN_SPEAKER_MIN),
        key=lambda pid: (md5_hex(pid), pid),
    )

    holdout: set[str] = set()
    val_min = unseen_city_min
    ran_out = True
    for pid in eligible:
        if val_min >= target_min:
            ran_out = False
            break
        holdout.add(pid)
        val_min += mixed_speaker_min[pid]
    else:
        ran_out = val_min < target_min

    diag = {
        "target_minutes": round(target_min, 1),
        "unseen_city_minutes": round(unseen_city_min, 1),
        "unseen_speaker_minutes": round(val_min - unseen_city_min, 1),
        "eligible_speakers": len(eligible),
        "held_out_speakers": len(holdout),
        "min_speaker_minutes": MIN_SPEAKER_MIN,
        "eligible_speakers_exhausted": ran_out,
    }
    return holdout, diag


def assign_split(pool: list[dict], holdout: set[str]) -> None:
    """Mutate rows in place with split + stratum."""
    for r in pool:
        if r["city_id"] in VAL_CITIES:
            r["split"], r["stratum"] = "val", "unseen_city"
        elif r["person_id"] is not None and r["person_id"] in holdout:
            r["split"], r["stratum"] = "val", "unseen_speaker"
        else:
            r["split"], r["stratum"] = "train", "train"


def validate(pool: list[dict], diag: dict) -> dict:
    train = [r for r in pool if r["split"] == "train"]
    val = [r for r in pool if r["split"] == "val"]

    train_pids = {r["person_id"] for r in train if r["person_id"] is not None}
    val_pids = {r["person_id"] for r in val if r["person_id"] is not None}
    overlap = train_pids & val_pids
    assert not overlap, f"speaker leakage: {len(overlap)} person_id in both splits"

    val_city_in_train = [r for r in train if r["city_id"] in VAL_CITIES]
    assert not val_city_in_train, (
        f"{len(val_city_in_train)} val-city utts leaked into train")

    uids = [r["utterance_id"] for r in pool]
    assert len(uids) == len(set(uids)), "duplicate utterance_id across pool"

    for r in pool:
        assert r["text"], f"empty text {r['utterance_id']}"
        assert isinstance(r["dur"], float) and r["dur"] > 0, (
            f"bad dur {r['utterance_id']}")

    total_min = sum(r["dur"] for r in pool) / 60.0
    val_min = sum(r["dur"] for r in val) / 60.0
    achieved = val_min / total_min if total_min else 0.0
    if not (VAL_BAND[0] <= achieved <= VAL_BAND[1]):
        log(f"WARNING: achieved val fraction {achieved:.3f} outside "
            f"{VAL_BAND} (eligible speakers exhausted="
            f"{diag['eligible_speakers_exhausted']})")

    # straddling-meeting diagnostic (mixed cities are speaker- not meeting-exclusive)
    mtg_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for r in pool:
        mtg_splits[(r["city_id"], r["meeting_id"])].add(r["split"])
    straddling = {m for m, s in mtg_splits.items() if len(s) > 1}
    straddle_val_min = sum(
        r["dur"] for r in val
        if (r["city_id"], r["meeting_id"]) in straddling) / 60.0

    return {
        "achieved_val_fraction": round(achieved, 4),
        "total_minutes": round(total_min, 1),
        "val_minutes": round(val_min, 1),
        "train_minutes": round(total_min - val_min, 1),
        "straddling_meetings": len(straddling),
        "straddling_val_minutes": round(straddle_val_min, 1),
    }


def summarize(pool: list[dict]) -> dict:
    def counts(rows):
        return {
            "n_utts": len(rows),
            "minutes": round(sum(r["dur"] for r in rows) / 60.0, 1),
        }

    by_split = {s: counts([r for r in pool if r["split"] == s])
                for s in sorted({r["split"] for r in pool})}
    by_stratum = {s: counts([r for r in pool if r["stratum"] == s])
                  for s in sorted({r["stratum"] for r in pool})}
    by_tier = {t: counts([r for r in pool if r["tier"] == t])
               for t in sorted({r["tier"] for r in pool})}
    by_city = {c: counts([r for r in pool if r["city_id"] == c])
               for c in sorted({r["city_id"] for r in pool})}

    n_noedit = sum(1 for r in pool if r["source"] == "no_edit")
    n_corr = sum(1 for r in pool if r["source"] == "correction")
    null_pid = {
        "train": sum(1 for r in pool
                     if r["split"] == "train" and r["person_id"] is None),
        "val": sum(1 for r in pool
                   if r["split"] == "val" and r["person_id"] is None),
    }
    return {
        "by_split": by_split,
        "by_stratum": by_stratum,
        "by_tier": by_tier,
        "by_city": by_city,
        "noedit_correction_ratio": round(n_noedit / n_corr, 3) if n_corr else None,
        "n_no_edit": n_noedit,
        "n_correction": n_corr,
        "null_person_id": null_pid,
    }


def write_outputs(pool: list[dict]) -> None:
    OUT_FIELDS = [
        "utterance_id", "city_id", "meeting_id", "person_id", "split",
        "stratum", "tier", "source", "start", "end", "dur", "audio_url",
        "text", "error_categories",
    ]

    def sort_key(r):
        return (r["split"], r["city_id"], r["meeting_id"],
                r["person_id"] or "", r["start"], r["utterance_id"])

    for split, path in (("train", TRAIN_JSONL), ("val", VAL_JSONL)):
        rows = sorted((r for r in pool if r["split"] == split), key=sort_key)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(
                    {k: r[k] for k in OUT_FIELDS}, ensure_ascii=False) + "\n")

    # canonical_split.csv: one row per (city, meeting, person_id)
    groups: dict[tuple, dict] = {}
    for r in pool:
        key = (r["city_id"], r["meeting_id"], r["person_id"] or "")
        g = groups.setdefault(key, {
            "city_id": r["city_id"], "meeting_id": r["meeting_id"],
            "person_id": r["person_id"] or "", "split": r["split"],
            "stratum": r["stratum"], "n_utts": 0, "minutes": 0.0,
        })
        assert g["split"] == r["split"], (
            f"group {key} spans splits {g['split']} vs {r['split']}")
        g["n_utts"] += 1
        g["minutes"] += r["dur"] / 60.0

    with SPLIT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city_id", "meeting_id", "person_id", "split", "stratum",
                    "n_utts", "minutes"])
        for key in sorted(groups):
            g = groups[key]
            w.writerow([g["city_id"], g["meeting_id"], g["person_id"],
                        g["split"], g["stratum"], g["n_utts"],
                        round(g["minutes"], 3)])


def main() -> None:
    excluded = load_excluded_keys()
    log(f"denylist: {len(excluded)} meetings")

    meta = json.loads(META.read_text())
    hr_meetings = sorted(
        (v["city_id"], v["meeting_id"]) for v in meta.values()
        if v.get("humanReview") and (v["city_id"], v["meeting_id"]) not in excluded
    )
    log(f"humanReview meetings (non-denylisted): {len(hr_meetings)}")

    corrections = load_corrections(excluded)
    correction_uids = {r["utterance_id"] for r in corrections}
    noedit, fetched = load_noedit(hr_meetings, correction_uids)

    pool = corrections + noedit
    person_map = load_person_map()
    no_match = 0
    for r in pool:
        pid = person_map.get(r["utterance_id"])
        if pid is None:
            no_match += 1
        r["person_id"] = pid
    log(f"pool: {len(pool)} utts; {no_match} without person_id (-> train, "
        f"unverifiable for val)")

    total_min = sum(r["dur"] for r in pool) / 60.0
    holdout, diag = choose_holdout_speakers(pool, total_min)
    diag["uncached_meetings_fetched"] = fetched
    log(f"hold-out: {len(holdout)} mixed-city speakers "
        f"(eligible={diag['eligible_speakers']}, "
        f"target={diag['target_minutes']}min)")

    assign_split(pool, holdout)
    val_stats = validate(pool, diag)

    write_outputs(pool)

    stats = {
        "seed": SEED,
        "val_cities": sorted(VAL_CITIES),
        "duration_gate_s": [MIN_DUR, MAX_DUR],
        "val_target_fraction": VAL_TARGET_FRAC,
        **val_stats,
        "holdout": diag,
        **summarize(pool),
    }
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    log(f"DONE -> {SPLIT_CSV.name}, {TRAIN_JSONL.name}, {VAL_JSONL.name}, "
        f"{STATS_JSON.name}")
    log(f"  val fraction={val_stats['achieved_val_fraction']} "
        f"total={val_stats['total_minutes']}min "
        f"val={val_stats['val_minutes']}min "
        f"noedit:corr ratio={stats['noedit_correction_ratio']}")


if __name__ == "__main__":
    main()
