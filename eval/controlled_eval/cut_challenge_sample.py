"""Cut the CHALLENGE SAMPLE: 40 `deletion_hard` clips from the untouched meetings.

Record: exp-2026-08-26-training-round-2. Design frozen 2026-09-06 in wayfinder #59,
built under #64. Terms: CONTEXT.md ("Challenge sample").

A TRAINING-SIDE DIAGNOSTIC, NOT AN EVALUATION

Two things are true of this sample at once, and together they decide what it is.

It is selected BECAUSE the incumbent already failed there. And its meetings are the ones
the 601 assisted clips will TRAIN on -- the same meetings, not merely similar ones. So it
cannot answer "is the new arm better": it is scored on material the arm was fitted to, at
positions chosen by the incumbent's failures. Both defects point the same way, towards
looking better than the arm is.

What it can answer is worth 15 minutes of listening anyway: **did the recipe improve the
failures in meetings it trained on?** A negative there kills the recipe outright -- if the
adapter did not fix the very spans it was retrained on, nothing downstream will save it.
A positive there means almost nothing on its own.

It is therefore reported in its own table, under its own heading, and it may NEVER be
added to the ordinary sample's number. `blind_preference.decide` refuses it outright;
`challenge_table` is the only way to score it and returns no verdict field at all.

THE ORDINARY SAMPLE MOVED AWAY FROM THIS FRAME AND THIS ONE CANNOT FOLLOW

On 2026-09-06 the ordinary sample's frame was replaced, because 376 of the 601 assisted
training clips sit inside 64 of its 67 meetings. This sample cannot make the same move:
`deletion_hard` spans exist only for queue meetings, so there is no version of it on the
new frame. That is the whole reason the two samples now sit on different frames, and the
reason this one is a diagnostic rather than a measurement.

This is a SEPARATE MODULE from `cut_ordinary_sample.py`, and the separation is the
point. The ordinary cutter's independence claim is that it never reads anything derived
from where a model failed, and that claim is enforced by an input gate that a test greps
its source for. Selecting on failure -- which is exactly what happens here -- would have
to punch a hole in that gate. So the failure-derived reads live over here instead, and
the ordinary module keeps a whitelist that mentions no model output at all.

The two samples are never summed. Three things stop it:

  * separate output directory, separate manifest, separate sha256;
  * every clip id starts `chl__` where the ordinary sample's start `ord__`, so a merged
    list of judgements is detectable by regex rather than by trust;
  * the manifest carries `sample: "challenge"`, and the scoring code keys on it.

WHAT IT REUSES

The 449 clips of `artifact-listen-queue-2026-08` that sit in the 67 untouched meetings
are already cut, already duration-checked and already hashed. This module selects 40 of
them and copies the wavs; it does not re-cut, because a re-cut clip would be a different
wav from the one whose sha256 the ledger vouches for. Their durations are the queue's
5-15 s, not the ordinary sample's 20 s -- another reason the two never pool.

SELECTION RULE, and where it goes beyond what was frozen  (unchanged; not re-cut)

The frozen design says only "40 clips from the existing `deletion_hard` rows of those
same untouched meetings". Two rules were carried over from the ordinary sample's v1
design rather than invented: **at most 2 per meeting** and **city proportions matched to
the 449-clip pool**. Both exist for the bootstrap -- the win rate is clustered by
meeting, and 40 clips drawn from a handful of meetings would have almost no clusters.
This is recorded as a carry-over rather than presented as part of the frozen design.

The selection has NOT moved since it was published. The ordinary sample's v1 machinery
that this borrowed -- `city_quotas`, a 2-per-meeting cap -- now lives in this file,
because the ordinary cutter's v2 frame made its own allocation fully determined and
deleted both. Sharing them would have let a change over there silently re-plan a
published sample here. Measured after the move: the same 40 clip ids, the same spans, the
same origin hashes.

    .venv-eval/bin/python -m eval.controlled_eval.cut_challenge_sample plan
    .venv-eval/bin/python -m eval.controlled_eval.cut_challenge_sample cut
    .venv-eval/bin/python -m eval.controlled_eval.cut_challenge_sample verify
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.controlled_eval import cut_ordinary_sample as ORD  # noqa: E402
from eval.controlled_eval.exclusion_manifest import Manifest, load_manifest  # noqa: E402

SC = ORD.SC

# The frozen listening queue, pinned by content. It belongs to THIS module alone now:
# the ordinary sample moved to a frame with no connection to it on 2026-09-06, and its
# cutter can no longer reach the queue even by accident.
QUEUE_PATH = SC / "listen-2026-08" / "generations" / "438aaea941866e08" / "queue.json"
QUEUE_SHA256 = "438aaea941866e087c0f25e02cd3a2b9b6badeea1c54ba3bda6b2400f4a0550a"
QUEUE_CLIPS_DIR = QUEUE_PATH.parent / "clips"

OUT_DIR = SC / "challenge-sample-2026-09"
GENERATIONS = OUT_DIR / "generations"
CURRENT = OUT_DIR / "current"
COUNTS_PATH = ROOT / "eval" / "results_challenge_sample.json"

# --------------------------------------------------------------------- frozen
SAMPLE_KIND = "challenge"
SOURCE = "challenge-sample-2026-09"
SEED = "challenge-sample-2026-09/v1"      # a DIFFERENT seed from the ordinary sample
POOL = "deletion_hard"

N_CLIPS = 40
# Frozen at 2 for THIS sample, and deliberately not borrowed from the ordinary cutter:
# that one moved to 3 with its v2 frame, and a shared constant would have silently
# re-planned a published sample.
MAX_PER_MEETING = 2

# The sentence that must travel with every number this sample produces.
ESTIMAND = (
    "TRAINING-SIDE DIAGNOSTIC, not an evaluation: preference on spans where the "
    "incumbent adapter already failed, inside meetings the assisted training clips are "
    "drawn from. Both the positions and the meetings favour the arm being judged. It "
    "answers only 'did the recipe fix the failures in meetings it trained on' -- a "
    "negative kills the recipe, a positive means little -- and it may never be added to "
    "the ordinary sample's number.")

CLIP_ID_PREFIX = "chl"
_CLIP_ID_RE = re.compile(r"^chl__[a-z0-9-]+__[a-z0-9_]+__\d+$")
_QUEUE_CLIP_ID_RE = re.compile(r"^[a-z0-9-]+__[a-z0-9_]+__\d+$")

# The queue's wavs are 5-15 s, so this sample checks the duration recorded in the queue
# rather than the ordinary sample's exact 320,000-frame count. Both wavs came out of the
# same `-ac 1 -ar 16000` path, so the shape check is still exact on channels, width and
# rate; only the frame count is per clip.
DURATION_TOLERANCE_MS = 500
EXPECT_WAV_BASE = {"nchannels": 1, "sampwidth": 2, "framerate": 16000}

ExclusionViolation = ORD.ExclusionViolation
GeometryError = ORD.GeometryError


class QuotaError(RuntimeError):
    """A city's clip quota cannot be met under the per-meeting cap."""


def city_quotas(pool_clips_by_city: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder apportionment of `n` clips over the pool's city proportions.

    Kept here rather than imported: the ordinary cutter had this function until its v2
    frame made the allocation fully determined, and it is gone from there. This sample is
    published and its selection must stay reproducible, so the arithmetic that produced
    it lives with it. Ties on the remainder break by city name, so the answer is one
    deterministic allocation rather than one of several equally defensible ones.
    """
    total = sum(pool_clips_by_city.values())
    if total <= 0:
        raise QuotaError("the pool is empty; there is nothing to apportion")
    if n < 0:
        raise QuotaError(f"cannot apportion {n} clips")
    exact = {c: n * v / total for c, v in pool_clips_by_city.items()}
    base = {c: int(v) for c, v in exact.items()}
    left = n - sum(base.values())
    order = sorted(exact, key=lambda c: (-(exact[c] - base[c]), c))
    for c in order[:left]:
        base[c] += 1
    return base


def queue_clips() -> list[dict]:
    """The frozen queue's rows, in full. Hash-pinned: this file defines the sample."""
    raw = QUEUE_PATH.read_bytes()
    got = ORD._sha256_bytes(raw)
    if got != QUEUE_SHA256:
        raise ValueError(
            f"{QUEUE_PATH} hashes to {got}, not the frozen {QUEUE_SHA256}.")
    rows = json.loads(raw.decode("utf-8"))["clips"]
    for r in rows:
        if r.get("pool") != POOL:
            raise ValueError(
                f"{r.get('clip_id')} is from pool {r.get('pool')!r}, not {POOL!r}; the "
                "challenge sample is defined by the deletion-hard pool")
    return rows


def eligible(rows, manifest: Manifest, touched: dict) -> list[dict]:
    """Queue rows in meetings no training source touched and no exclusion blocks."""
    out = []
    for r in rows:
        key = (r["city"], r["meeting_id"])
        if manifest.is_excluded(*key) or key in touched:
            continue
        out.append(r)
    return out


def challenge_clip_id(row: dict) -> str:
    """`chl__` + the queue's own clip id, so the origin stays readable on disk."""
    src = row["clip_id"]
    if not _QUEUE_CLIP_ID_RE.match(src or ""):
        raise GeometryError(f"queue clip id is not path-safe: {src!r}")
    cid = f"{CLIP_ID_PREFIX}__{src}"
    if not _CLIP_ID_RE.match(cid):
        raise GeometryError(f"clip id is not path-safe: {cid!r}")
    return cid


def select(rows, quotas: dict[str, int],
           max_per_meeting: int = MAX_PER_MEETING) -> list[dict]:
    """Deal `quotas[city]` clips over that city's meetings, at most two each.

    Both the meeting order and the choice of clips inside a meeting come from
    `_unit(SEED, ...)`, so the selection is reproducible from the seed and the queue and
    is not a function of how rich any witness happened to be. Which clips exist at all
    is already a function of where the adapter failed -- that is the definition of this
    sample -- but which of them are picked is not a second helping of the same bias.
    """
    by_city: dict[str, dict[tuple[str, str], list[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in rows:
        by_city[r["city"]][(r["city"], r["meeting_id"])].append(r)

    chosen: list[dict] = []
    for city, want in sorted(quotas.items()):
        meetings = by_city.get(city, {})
        # The real capacity, not `2 x meetings`: a meeting holding one eligible clip
        # supplies one. Checked before anything is dealt, so an infeasible quota fails
        # loudly instead of being silently short (Codex review 0201f130).
        cap = sum(min(max_per_meeting, len(v)) for v in meetings.values())
        if want > cap:
            raise QuotaError(
                f"{city} needs {want} clips but its {len(meetings)} untouched meetings "
                f"can supply at most {cap} at {max_per_meeting} each. Refusing to "
                "reassign the remainder to another city: that would silently change the "
                "city proportions the design froze.")
        order = sorted(meetings, key=lambda k: (ORD._unit(SEED, "order", k[0], k[1]), k))
        ranked = {
            k: sorted(meetings[k],
                      key=lambda r: (ORD._unit(SEED, "pick", r["clip_id"]),
                                     r["clip_id"]))
            for k in order}
        taken: collections.Counter = collections.Counter()
        given = 0
        for _round in range(max_per_meeting):
            for k in order:
                if given >= want:
                    break
                if taken[k] >= len(ranked[k]):
                    continue
                chosen.append(ranked[k][taken[k]])
                taken[k] += 1
                given += 1
            if given >= want:
                break
        if given < want:
            raise QuotaError(
                f"{city} could only supply {given} of {want} clips under the "
                f"{max_per_meeting}-per-meeting cap")
    return chosen


def sample_row(row: dict) -> dict:
    """The row a preference page loads. Provenance and geometry, never text.

    `origin_clip_id` and `origin_row_id` are kept on purpose: this sample IS defined by
    where the adapter failed, so its provenance has to name the failure it came from.
    """
    return {
        "clip_id": challenge_clip_id(row),
        "sample": SAMPLE_KIND,
        "city": row["city"],
        "meeting_id": row["meeting_id"],
        "meeting_date": row.get("meeting_date"),
        "audio_url": row["audio_url"],
        "clip_start": row["clip_start"],
        "clip_end": row["clip_end"],
        "clip_dur": row["clip_dur"],
        "source": SOURCE,
        "selection": "incumbent-failed-here",
        "pool": POOL,
        "origin_clip_id": row["clip_id"],
        "origin_row_id": row["origin_row_id"],
        "origin_queue_sha256": QUEUE_SHA256[:16],
        "origin_audio_sha256": row.get("audio_sha256"),
        "seed": SEED,
    }


def check_sample(rows, manifest: Manifest, touched: dict) -> None:
    """The last guard before publication. Raises, never asserts: `python -O`."""
    per_meeting = collections.Counter((r["city"], r["meeting_id"]) for r in rows)
    for key, n in per_meeting.items():
        if manifest.is_excluded(*key):
            raise ExclusionViolation(f"excluded meeting {key} reached the sample")
        if key in touched:
            raise ExclusionViolation(
                f"{key} is in {touched[key]} and is not an untouched meeting")
        if n > MAX_PER_MEETING:
            raise ExclusionViolation(f"{key} supplied {n} clips, cap is {MAX_PER_MEETING}")
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dup = [i for i, n in collections.Counter(ids).items() if n > 1]
        raise GeometryError(f"clip id collision: {dup[:3]}")
    for r in rows:
        if not _CLIP_ID_RE.match(r["clip_id"]):
            raise GeometryError(f"clip id is not path-safe: {r['clip_id']!r}")
        if r["sample"] != SAMPLE_KIND:
            raise ExclusionViolation(
                f"{r['clip_id']} is labelled {r['sample']!r}; the two samples are never "
                "summed and a mislabelled row is how they would be")


def plan() -> dict:
    manifest = load_manifest()
    touched = ORD.touched_meetings()
    rows = eligible(queue_clips(), manifest, touched)

    pool_by_city: collections.Counter = collections.Counter(r["city"] for r in rows)
    quotas = city_quotas(dict(pool_by_city), N_CLIPS)
    picked = [sample_row(r) for r in select(rows, quotas)]
    picked.sort(key=lambda r: (r["city"], r["meeting_id"], r["clip_start"]))
    check_sample(picked, manifest, touched)

    per_meeting = collections.Counter((r["city"], r["meeting_id"]) for r in picked)
    return {
        "summary": {
            "sample": SAMPLE_KIND,
            "source": SOURCE,
            "seed": SEED,
            "pool": POOL,
            "max_per_meeting": MAX_PER_MEETING,
            "queue_sha256": QUEUE_SHA256[:16],
            "exclusion_manifest_sha256": manifest.loaded_sha256,
            "n_pool_clips": len(rows),
            "n_pool_meetings": len({(r["city"], r["meeting_id"]) for r in rows}),
            "pool_clips_by_city": dict(sorted(pool_by_city.items())),
            "city_quotas": dict(sorted(quotas.items())),
            "n_clips": len(picked),
            "n_meetings_used": len(per_meeting),
            "clips_by_city": dict(sorted(collections.Counter(
                r["city"] for r in picked).items())),
            "meetings_with_two_clips": sum(1 for v in per_meeting.values() if v == 2),
            "clip_seconds_min": round(min(r["clip_dur"] for r in picked), 3),
            "clip_seconds_max": round(max(r["clip_dur"] for r in picked), 3),
            "clip_hours": round(sum(r["clip_dur"] for r in picked) / 3600, 4),
            "decides": False,
            "role": "training-side diagnostic",
            "estimand": ESTIMAND,
            "may_support": ("did the recipe fix the failures in meetings it trained on; "
                            "never a claim about overall quality, and never summed with "
                            "the ordinary sample"),
        },
        "clips": picked,
    }


def cut() -> dict:
    """Cut the whole sample or publish nothing.

    There is deliberately no `--limit`. A short cut is a different sample -- a different
    city mix and a different number of bootstrap clusters -- and the guard below exists
    precisely to stop one being published.
    """
    out = plan()
    rows = out["clips"]
    summary = out["summary"]

    stage = OUT_DIR / "_staging"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "clips").mkdir(parents=True)
    failed = []
    for row in rows:
        src = QUEUE_CLIPS_DIR / f"{row['origin_clip_id']}.wav"
        dst = stage / "clips" / f"{row['clip_id']}.wav"
        if not src.is_file():
            failed.append({"clip_id": row["clip_id"], "reason": "source-missing"})
            continue
        got = ORD.sha256_file(src)
        if row.get("origin_audio_sha256") and got != row["origin_audio_sha256"]:
            failed.append({"clip_id": row["clip_id"], "reason": "origin-hash-mismatch"})
            continue
        shape = ORD.wav_shape(src)
        if shape is None or any(shape.get(k) != v for k, v in EXPECT_WAV_BASE.items()):
            failed.append({"clip_id": row["clip_id"], "reason": "format-mismatch"})
            continue
        d = shape["nframes"] / shape["framerate"]
        if abs(d * 1000 - row["clip_dur"] * 1000) > DURATION_TOLERANCE_MS:
            failed.append({"clip_id": row["clip_id"], "reason": "duration-mismatch"})
            continue
        shutil.copyfile(src, dst)
        row["cut_method"] = "queue-clip-reuse"
        row["measured_dur"] = round(d, 3)
        row["audio_sha256"] = got
    cut_rows = [r for r in rows if r.get("audio_sha256")]
    summary["n_clips_cut"] = len(cut_rows)
    summary["n_failed"] = len(failed)
    summary["failure_reasons"] = dict(collections.Counter(
        f["reason"] for f in failed))
    for f in failed:
        print(f"  FAILED {f['clip_id']}: {f['reason']}", flush=True)
    if failed:
        # No partial publication: a sample short of its planned clips has a city mix
        # nobody chose (Codex review 0201f130).
        shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError(
            f"{len(failed)} of {len(rows)} clips did not copy "
            f"({summary['failure_reasons']}). Nothing was published.")
    rows = cut_rows

    (stage / "sample.json").write_text(
        json.dumps({"summary": summary, "clips": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    gen_sha = ORD.sha256_file(stage / "sample.json")
    GENERATIONS.mkdir(parents=True, exist_ok=True)
    gen = GENERATIONS / gen_sha[:16]
    if gen.exists():
        shutil.rmtree(gen)
    os.replace(stage, gen)
    if CURRENT.exists() and not CURRENT.is_symlink():
        raise RuntimeError(
            f"{CURRENT} is a real directory, not the generation symlink. Move it aside "
            "and re-run, or the new generation stays unpublished.")
    link_tmp = OUT_DIR / ".current.new"
    if link_tmp.exists() or link_tmp.is_symlink():
        link_tmp.unlink()
    link_tmp.symlink_to(gen, target_is_directory=True)
    os.replace(link_tmp, CURRENT)

    summary["generation"] = gen_sha[:16]
    summary["sample_sha256"] = gen_sha
    COUNTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return {"summary": summary, "clips": rows}


def verify(current: Path = CURRENT) -> dict:
    manifest = load_manifest()
    touched = ORD.touched_meetings()
    d = json.loads((Path(current) / "sample.json").read_text(encoding="utf-8"))
    rows = d["clips"]
    bad = {"excluded": [], "trained_on": [], "missing": [], "hash": [], "duration": [],
           "format": [], "clip_id": [], "wrong_sample": [], "over_cap": []}
    per_meeting = collections.Counter((r["city"], r["meeting_id"]) for r in rows)
    for key, n in per_meeting.items():
        if manifest.is_excluded(*key):
            bad["excluded"].append(f"{key[0]}/{key[1]}")
        if key in touched:
            bad["trained_on"].append(f"{key[0]}/{key[1]}")
        if n > MAX_PER_MEETING:
            bad["over_cap"].append(f"{key[0]}/{key[1]}")
    for r in rows:
        if not _CLIP_ID_RE.match(r["clip_id"]):
            bad["clip_id"].append(r["clip_id"])
        if r.get("sample") != SAMPLE_KIND:
            bad["wrong_sample"].append(r["clip_id"])
        p = Path(current) / "clips" / f"{r['clip_id']}.wav"
        if not p.is_file():
            bad["missing"].append(r["clip_id"])
            continue
        if r.get("audio_sha256") and ORD.sha256_file(p) != r["audio_sha256"]:
            bad["hash"].append(r["clip_id"])
        shape = ORD.wav_shape(p)
        if shape is None or any(shape.get(k) != v for k, v in EXPECT_WAV_BASE.items()):
            bad["format"].append(r["clip_id"])
            continue
        if abs(shape["nframes"] / shape["framerate"] * 1000
               - r["clip_dur"] * 1000) > DURATION_TOLERANCE_MS:
            bad["duration"].append(r["clip_id"])
    return {"n": len(rows),
            "n_meetings": len(per_meeting),
            "bad": {k: len(v) for k, v in bad.items()},
            "examples": {k: v[:5] for k, v in bad.items() if v}}


def main() -> None:  # pragma: no cover - operational entry point
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "cut", "verify"])
    a = ap.parse_args()
    if a.mode == "verify":
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return
    if a.mode == "plan":
        print(json.dumps(plan()["summary"], ensure_ascii=False, indent=2))
        return
    print(json.dumps(cut()["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
