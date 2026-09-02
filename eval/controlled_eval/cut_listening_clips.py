"""Cut the blind-listening clip queue for the `listen-2026-08` training round.

Spec: docs/specs/2026-08-26-training-round-2.md sections 4.1-4.2.
Record: exp-2026-08-26-training-round-2.
Plan reviewed by Codex `4cf77e98` (high effort) before the queue was published; four of
its findings were measured to be correct and are implemented here, and the two places
where this module still disagrees with it are named below.

Replaces `chooser/cutclips.py`, which cannot be reused for two independent reasons
(`cutclips.py:3-8`, `:45`): it positions the clip by column fraction rather than by a
timestamp, and every one of its meetings is excluded by section 4.2.

Source pool 1, the only pool with inputs on disk: the `deletion_hard` bucket of the
stage-2 training manifest. Each row is a span the adapter was trained to transcribe and
2,661 of the 3,921 carry a `VERIFIED` Soniox `stt-rt-v4` witness with words the training
target does not have. Those rows are also exactly the ones the earlier human review
never saw, so the witness filter is simultaneously the "unreviewed remainder first" rule
of section 4.1 -- one filter doing two jobs, which is why it must not be relaxed to fill
the pool. Pool 2 of the spec -- adapter-versus-Soniox disagreement on training-side
meetings -- is NOT implemented, because its inputs do not exist: it needs adapter decodes
of training-side meetings and Soniox calls on those spans.

FOUR THINGS THIS MODULE IS RESPONSIBLE FOR NOT GETTING WRONG

1. Exclusions. A clip taken from an evaluation meeting is training on the test set and is
   not reversible. The manifest is pinned by hash, cross-checked against its live
   sources, applied as a filter, and then checked again over the selected output -- with
   a raised exception, not an `assert`, because `python -O` deletes asserts.

2. The witness has to describe THIS span. `gap-verify/api/<row>.json` records the span
   Soniox was actually run on. Measured 2026-08-27: 71 of the surviving rows have since
   been edited and now differ from it, by up to 13.25 s. For those rows `soniox_text`
   describes different audio and the cached wav is a different clip, so they are dropped
   rather than re-cut: re-cutting would give a clip whose witness is somebody else's.

3. The overlap rule is applied LITERALLY. An earlier draft of this module argued that
   "clips overlapping any existing training clip are dropped" could not be meant
   literally, because a meeting's training rows are contiguous and a 3 s pad would always
   hit a neighbour. That argument was wrong and the measurement says so: 724 of the 899
   selected clips overlap no other training row at all, and only 175 overlap one or more.
   The training rows are a sample of a meeting, not a covering of it. So the literal rule
   is affordable and is used, which also removes the label-supersession problem -- no new
   label can contradict an existing row's target, because no clip touches one.

4. Blindness. The witness text and the current training target are written to files the
   listening server never serves, and `blind_listen.py` releases them only after a label
   is committed. Blindness is a property of the pipeline, not of the renderer.

WHERE THIS DISAGREES WITH THE REVIEW

  * `clip_id` stays readable (`<city>__<meeting>__<start_ms>`) instead of becoming a
    digest. A digest makes a stray wav on disk untraceable, which is the failure this
    project actually keeps having. Collisions are made impossible by an explicit
    uniqueness check rather than by hashing, and the id is whitelisted by regex before
    it ever reaches a path.
  * The five hidden repeat presentations and the uncertainty states of the frozen
    listening protocol live in `blind_listen.py`, not here. They are properties of the
    session, not of the queue, and a queue that already contained its own repeats could
    not be re-used for a second listener.

Nothing this module writes goes in git except counts.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.controlled_eval.exclusion_manifest import (  # noqa: E402
    Manifest, load_manifest)

SC = Path.home() / ".cache" / "oc-public"
TRAIN_MANIFEST = SC / "train-screens-2026-08" / "run1" / "manifest.jsonl"
GAP_VERIFY = SC / "gap-verify" / "manifest.jsonl"
GAP_API = SC / "gap-verify" / "api"
OUT_DIR = SC / "listen-2026-08"
# One immutable directory per cut, and a symlink that names the live one. A cut that
# dies half way leaves a partial generation nobody is pointing at, instead of a
# published queue whose clips and manifest disagree (CodeRabbit, 2026-08-27).
GENERATIONS = OUT_DIR / "generations"
CURRENT = OUT_DIR / "current"
LABELS_PATH = OUT_DIR / "labels.json"
COUNTS_PATH = ROOT / "eval" / "results_listening_queue.json"

# --------------------------------------------------------------------- frozen
POOL = "deletion_hard"
SOURCE = "listen-2026-08"
WITNESS = "soniox-stt-rt-v4"      # the FREE real-time model, never stt-async-v5
ACCEPT_TIERS = ("VERIFIED",)      # MIDDLE (12 rows) and REJECT (4) are not a witness

# PAD is 3.0 because that is the pad the gap-verify wavs on disk were cut with
# (`eval/gap2_verify_audio.py:43`, `:253`); matching it is what makes reuse exact.
PAD_MS = 3000
MIN_CLIP_MS = 5000
MAX_CLIP_MS = 15000
PAD = PAD_MS / 1000
MIN_CLIP = MIN_CLIP_MS / 1000
MAX_CLIP = MAX_CLIP_MS / 1000

# How far the row's span may sit from the span Soniox was actually run on. One
# millisecond of serialization slack, not a tolerance: a shifted clip with the same
# duration is a different clip and its witness belongs to somebody else.
WITNESS_SPAN_TOLERANCE_MS = 10

# A wav's measured duration may differ from its nominal span by mp3 frame alignment.
# Measured worst case on a matched clip was 0.20 s.
DURATION_TOLERANCE_MS = 500

_CLIP_ID_RE = re.compile(r"^[a-z0-9]+__[a-z0-9_]+__\d+$")


class GeometryError(ValueError):
    """A clip span violates a frozen invariant. Never an assert: `python -O`."""


class ExclusionViolation(RuntimeError):
    """An excluded meeting reached the queue. Nothing may continue past this."""


@dataclass(frozen=True)
class Geometry:
    """A clip span, quantised to whole milliseconds before anything else touches it."""

    start_ms: int
    end_ms: int
    pad_left_ms: int
    pad_right_ms: int
    reuse_gap_verify: bool

    @property
    def start(self) -> float:
        return self.start_ms / 1000

    @property
    def end(self) -> float:
        return self.end_ms / 1000

    @property
    def dur(self) -> float:
        return (self.end_ms - self.start_ms) / 1000

    @property
    def pad(self) -> float:
        """Kept for readability; only meaningful when the two sides are equal."""
        return self.pad_left_ms / 1000


def _ms(x) -> int:
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise GeometryError(f"non-finite timestamp: {x!r}")
    return int(round(v * 1000))


def clip_geometry(start: float, dur: float) -> Geometry | None:
    """The clip span for a row, or None when the row cannot be a 5-15 s clip.

    A row longer than the ceiling is dropped rather than truncated: a truncated clip
    asks the listener to label speech that runs off the edge of what they can hear.

    Padding is a budget, not a fixed margin on each side. Near the start of a recording
    the left pad is clipped by zero, and the remainder moves to the right -- otherwise a
    0.14 s row at t=0 would produce a 3.14 s clip and silently break the 5 s floor.
    """
    start_ms, dur_ms = _ms(start), _ms(dur)
    if dur_ms < 0:
        raise GeometryError(f"negative duration: {dur}")
    if start_ms < 0:
        raise GeometryError(f"negative start: {start}")
    if dur_ms > MAX_CLIP_MS:
        return None
    budget = min(2 * PAD_MS, MAX_CLIP_MS - dur_ms)
    left = min(budget // 2, start_ms)
    right = budget - left
    s, e = start_ms - left, start_ms + dur_ms + right
    if not (MIN_CLIP_MS <= e - s <= MAX_CLIP_MS):
        raise GeometryError(
            f"clip of {(e - s) / 1000:.3f}s is outside [{MIN_CLIP}, {MAX_CLIP}]")
    # Reuse is only exact when both pads are the frozen 3 s, which is precisely the
    # geometry `gap2_verify_audio.py` cut -- including its own `max(0, start - PAD)`.
    return Geometry(start_ms=s, end_ms=e, pad_left_ms=left, pad_right_ms=right,
                    reuse_gap_verify=(left == PAD_MS and right == PAD_MS))


def clip_id(city: str, meeting: str, geom: Geometry) -> str:
    """Stable, readable, and names its meeting so a stray wav on disk is traceable."""
    cid = f"{city}__{meeting}__{geom.start_ms}"
    if not _CLIP_ID_RE.match(cid):
        raise GeometryError(f"clip id is not path-safe: {cid!r}")
    return cid


def candidates(train_rows, gap_verify: dict) -> list[dict]:
    """Witnessed `deletion_hard` rows, as provenance dicts carrying no text.

    Rows whose geometry is None are kept with `geometry=None` so `select` can count
    them. `end - start` is canonical; a `dur` field that disagrees is an error rather
    than a rounding difference, because everything downstream keys off the span.
    """
    out = []
    for r in train_rows:
        if r.get("bucket") != POOL:
            continue
        gv = gap_verify.get(r["id"])
        if not gv or not (gv.get("soniox_text") or "").strip():
            continue
        if gv.get("tier") not in ACCEPT_TIERS:   # unknown tiers fail closed
            continue
        start_ms, end_ms = _ms(r["start"]), _ms(r["end"])
        if end_ms <= start_ms:
            raise GeometryError(f"row {r['id']} has end <= start")
        if abs((end_ms - start_ms) - _ms(r["dur"])) > 1:
            raise GeometryError(
                f"row {r['id']}: dur {r['dur']} disagrees with end - start")
        dur = (end_ms - start_ms) / 1000
        out.append({
            "origin_row_id": r["id"],
            "city": r["city_id"],
            "meeting_id": r["meeting_id"],
            "meeting_date": r.get("meeting_date"),
            "audio_url": r["audio_url"],
            "row_start": start_ms / 1000,
            "row_dur": dur,
            "n_added": int(gv.get("n_added") or 0),
            "found_frac": float(gv.get("found_frac") or 0.0),
            "gap_verify_clip": gv.get("clip_padded"),
            "geometry": clip_geometry(start_ms / 1000, dur),
        })
    return out


def witness_matches(cand: dict, prov: dict | None) -> bool:
    """Was Soniox run on THIS span, of THIS recording?

    `gap-verify/api/<row>.json` is the record of what was verified. A row edited since
    then still carries its old `soniox_text`, which now describes different audio; the
    cached wav is that other clip too. Same duration at a different offset must fail,
    so both endpoints are compared, not the length.
    """
    if not prov:
        return False
    if prov.get("audio_url") != cand["audio_url"]:
        return False
    try:
        ds = abs(_ms(prov["start"]) - _ms(cand["row_start"]))
        de = abs(_ms(prov["end"]) - _ms(cand["row_start"] + cand["row_dur"]))
    except (KeyError, GeometryError, TypeError):
        return False
    return max(ds, de) <= WITNESS_SPAN_TOLERANCE_MS


def queue_row(cand: dict) -> dict:
    """The row a listening page loads. It carries provenance and no transcript."""
    g = cand["geometry"]
    if g is None:
        raise GeometryError("a row with no geometry never reaches the queue")
    return {
        "clip_id": clip_id(cand["city"], cand["meeting_id"], g),
        "city": cand["city"],
        "meeting_id": cand["meeting_id"],
        "meeting_date": cand["meeting_date"],
        "audio_url": cand["audio_url"],
        "clip_start": g.start,
        "clip_end": g.end,
        "clip_dur": round(g.dur, 3),
        "pool": POOL,
        "origin_row_id": cand["origin_row_id"],
        "source": SOURCE,
        "witness": WITNESS,
        "reused_gap_verify_clip": g.reuse_gap_verify,
    }


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _schedule(rows: list[dict]) -> tuple[list[dict], int]:
    """Pick the largest non-overlapping set of clips in one meeting.

    Weighted interval scheduling rather than a pairwise greedy pass: with a chain
    `A-B, B-C, A/C disjoint` a greedy rule drops a clip it did not have to. The
    objective is (how many clips, then how many witness-only words, then the
    lexicographically smaller set of row ids) so the result is a single deterministic
    answer rather than one of several equally good ones.
    """
    items = sorted(rows, key=lambda c: (c["geometry"].end_ms,
                                        c["geometry"].start_ms,
                                        c["origin_row_id"]))
    n = len(items)
    best: list[tuple] = [(0, 0, (), ())] * (n + 1)   # (count, weight, ids, chosen idx)
    for i in range(1, n + 1):
        cur = items[i - 1]
        skip = best[i - 1]
        j = i - 1
        while j > 0 and items[j - 1]["geometry"].end_ms > cur["geometry"].start_ms:
            j -= 1
        prev = best[j]
        take = (prev[0] + 1, prev[1] + cur["n_added"],
                prev[2] + (cur["origin_row_id"],), prev[3] + (i - 1,))
        # bigger count, then bigger weight, then the lexicographically smaller id tuple
        best[i] = take if (take[0], take[1], tuple(-ord(ch) for ch in "".join(take[2]))) \
            > (skip[0], skip[1], tuple(-ord(ch) for ch in "".join(skip[2]))) else skip
    chosen = [items[k] for k in best[n][3]]
    return chosen, n - len(chosen)


def select(cands, manifest: Manifest, training_spans: dict, provenance: dict,
           *, _skip_exclusion_filter_for_test: bool = False):
    """Apply every filter, in order, and return (queue rows, drop counts, ledger).

    Order matters only for the counts; the filters are independent. The ledger records
    one line per candidate with its drop reason and stays out of git.
    """
    drops = collections.Counter({
        "excluded": 0, "too_long": 0, "witness_span_mismatch": 0,
        "training_overlap": 0, "queue_overlap": 0})
    ledger = []
    kept = []
    for c in cands:
        rid = c["origin_row_id"]
        if not _skip_exclusion_filter_for_test and manifest.is_excluded(
                c["city"], c["meeting_id"]):
            drops["excluded"] += 1
            ledger.append({"row": rid, "drop": "excluded"})
            continue
        if c["geometry"] is None:
            drops["too_long"] += 1
            ledger.append({"row": rid, "drop": "too_long"})
            continue
        if not witness_matches(c, provenance.get(rid)):
            drops["witness_span_mismatch"] += 1
            ledger.append({"row": rid, "drop": "witness_span_mismatch"})
            continue
        g = c["geometry"]
        others = [i for s, e, i in training_spans.get((c["city"], c["meeting_id"]), ())
                  if i != rid and _overlaps((g.start_ms, g.end_ms), (s, e))]
        if others:
            drops["training_overlap"] += 1
            ledger.append({"row": rid, "drop": "training_overlap", "with": others})
            continue
        kept.append(c)

    by_meeting: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for c in kept:
        by_meeting[(c["city"], c["meeting_id"])].append(c)

    chosen: list[dict] = []
    for key in sorted(by_meeting):
        take, lost = _schedule(by_meeting[key])
        drops["queue_overlap"] += lost
        taken_ids = {c["origin_row_id"] for c in take}
        for c in by_meeting[key]:
            if c["origin_row_id"] not in taken_ids:
                ledger.append({"row": c["origin_row_id"], "drop": "queue_overlap"})
        chosen.extend(take)

    chosen.sort(key=lambda c: (c["city"], c["meeting_id"], c["geometry"].start_ms))
    out = [queue_row(c) for c in chosen]

    ids = [r["clip_id"] for r in out]
    if len(set(ids)) != len(ids):
        dup = [i for i, n in collections.Counter(ids).items() if n > 1]
        raise GeometryError(f"clip id collision: {dup[:3]}")
    for row in out:
        if manifest.is_excluded(row["city"], row["meeting_id"]):
            raise ExclusionViolation(
                f"excluded meeting {row['city']}/{row['meeting_id']} reached the queue")
    for c in chosen:
        ledger.append({"row": c["origin_row_id"], "drop": None,
                       "clip_id": clip_id(c["city"], c["meeting_id"], c["geometry"])})
    return out, drops, ledger


# ------------------------------------------------------------------ operational

def _jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_pool():
    train = list(_jsonl(TRAIN_MANIFEST))
    gv = {r["id"]: r for r in _jsonl(GAP_VERIFY)}
    return train, gv


def load_provenance(row_ids) -> dict:
    """The span Soniox was actually run on, per row, from the gap-verify API snapshot."""
    out = {}
    for rid in row_ids:
        p = GAP_API / f"{rid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        out[rid] = {"start": d.get("start"), "end": d.get("end"),
                    "audio_url": d.get("audio_url")}
    return out


def training_span_index(train_rows) -> dict:
    idx: dict[tuple[str, str], list[tuple[int, int, str]]] = collections.defaultdict(list)
    for r in train_rows:
        idx[(r["city_id"], r["meeting_id"])].append(
            (_ms(r["start"]), _ms(r["end"]), r["id"]))
    for k in idx:
        idx[k].sort()
    return idx


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_duration(path: Path) -> float | None:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


# What every clip in the queue must be. The wavs come from two sources -- ffmpeg with
# `-ac 1 -ar 16000`, and gap-verify files cut the same way -- so a clip that is not
# mono 16 kHz PCM came from somewhere unexpected and must not be trained on silently.
EXPECT_STREAM = {"codec_name": "pcm_s16le", "sample_rate": "16000", "channels": 1}


def probe_stream(path: Path) -> dict | None:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True)
    try:
        streams = json.loads(p.stdout)["streams"]
    except (ValueError, KeyError):
        return None
    return streams[0] if streams else None


# ffmpeg writes the input URL into its own error messages, and the failure summary of a
# cut is written to a file that lives IN GIT. A raw stderr tail would therefore put a
# council recording URL in the repository, which is the rule this project purged history
# for once. The classification is deliberately coarse; the raw text stays in the
# in-memory `failed` list, where an operator can see it and git cannot.
_FAILURE_CLASSES = (
    ("http-fetch-failed", ("http error", "403", "404", "server returned", "connection")),
    ("source-missing", ("no such file", "does not exist")),
    ("decode-failed", ("invalid data", "moov atom", "error while decoding")),
    ("seek-failed", ("could not seek", "seek to")),
)


def classify_failure(stderr: str) -> str:
    low = (stderr or "").lower()
    for name, needles in _FAILURE_CLASSES:
        if any(n in low for n in needles):
            return name
    return "ffmpeg-error"


def cut_one(cand: dict, dst: Path) -> dict:
    """Reuse the gap-verify wav when its geometry matches, otherwise cut from source.

    Reuse is only reached for rows whose witness provenance already matched, so the
    remaining question is whether the cached file is the clip it claims to be; its
    measured duration answers that.
    """
    g = cand["geometry"]
    src = Path(cand["gap_verify_clip"]) if cand.get("gap_verify_clip") else None
    if g.reuse_gap_verify and src and src.exists():
        d = probe_duration(src)
        if d is not None and abs(_ms(d) - (g.end_ms - g.start_ms)) <= DURATION_TOLERANCE_MS:
            shutil.copyfile(src, dst)
            return {"method": "gap-verify-reuse", "measured_dur": round(d, 3)}
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{g.start:.3f}",
         "-i", cand["audio_url"], "-t", f"{g.dur:.3f}", "-ac", "1", "-ar", "16000",
         str(dst)], capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not dst.exists():
        return {"method": None, "error": p.stderr.strip()[-200:],
                "reason": classify_failure(p.stderr)}
    d = probe_duration(dst)
    if d is None or abs(_ms(d) - (g.end_ms - g.start_ms)) > DURATION_TOLERANCE_MS:
        dst.unlink(missing_ok=True)
        return {"method": None, "reason": "duration-mismatch",
                "error": f"cut is {d}s, expected {g.dur:.3f}s (end of recording?)"}
    return {"method": "ffmpeg", "measured_dur": round(d, 3)}


def build(limit: int | None = None, cut: bool = False) -> dict:
    manifest = load_manifest()          # hash-pinned and cross-checked against sources
    train, gv = load_pool()
    cands = candidates(train, gv)
    prov = load_provenance(c["origin_row_id"] for c in cands)
    rows, drops, ledger = select(cands, manifest, training_span_index(train), prov)

    # Keyed on the origin row id, which the training manifest guarantees unique, and NOT
    # on the clip id: two rows of one meeting starting at the same millisecond would
    # share a clip id while carrying different spans, and the cut would then be made for
    # one row while the witness was written for the other.
    by_row = {c["origin_row_id"]: c for c in cands}
    if len(by_row) != len(cands):
        raise GeometryError("origin row ids are not unique")

    if limit:
        rows = rows[:limit]

    summary = {
        "source": SOURCE,
        "pool": POOL,
        "witness": WITNESS,
        "exclusion_manifest_sha256": sha256_file(
            SC / "exclusion-manifest-2026-08" / "manifest.json"),
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST)[:16],
        "gap_verify_sha256": sha256_file(GAP_VERIFY)[:16],
        "n_train_rows": len(train),
        "n_pool_rows": sum(1 for r in train if r.get("bucket") == POOL),
        "n_candidates": len(cands),
        "n_clips": len(rows),
        "drops": dict(drops),
        "clip_hours": round(sum(r["clip_dur"] for r in rows) / 3600, 4),
        "labelled_span_hours": round(
            sum(by_row[r["origin_row_id"]]["row_dur"] for r in rows) / 3600, 4),
        "cities": dict(collections.Counter(r["city"] for r in rows)),
        "meetings": len({(r["city"], r["meeting_id"]) for r in rows}),
        "reused_gap_verify": sum(1 for r in rows if r["reused_gap_verify_clip"]),
    }

    if not cut:
        return {"summary": summary, "queue": rows}

    labels = LABELS_PATH
    if labels.exists() and json.loads(labels.read_text(encoding="utf-8")):
        raise RuntimeError(
            f"{labels} already holds labels. Re-cutting would republish a queue the "
            "listener has already worked through; move the labels aside first.")

    # Stage into a temporary generation and publish by moving ONE symlink, so a crash
    # halfway cannot leave a shorter queue looking like a complete one.
    stage = OUT_DIR / "_staging"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "clips").mkdir(parents=True)
    published = CURRENT / "clips"
    failed = []
    for i, row in enumerate(rows, 1):
        cand = by_row[row["origin_row_id"]]
        dst = stage / "clips" / f"{row['clip_id']}.wav"
        old = published / f"{row['clip_id']}.wav"
        if old.exists() and abs(_ms(probe_duration(old) or 0)
                                - _ms(row["clip_dur"])) <= DURATION_TOLERANCE_MS:
            shutil.copyfile(old, dst)     # a previous generation cut this correctly
            row["cut_method"] = "carried-over"
            row["measured_dur"] = round(probe_duration(dst) or 0.0, 3)
        else:
            res = cut_one(cand, dst)
            if res.get("method") is None:
                failed.append({"clip_id": row["clip_id"], "error": res.get("error"),
                               "reason": res.get("reason", "ffmpeg-error")})
                continue
            row["cut_method"] = res["method"]
            row["measured_dur"] = res.get("measured_dur")
        row["audio_sha256"] = sha256_file(dst)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    rows = [r for r in rows if r.get("audio_sha256")]
    summary["n_clips_cut"] = len(rows)
    summary["n_failed"] = len(failed)
    # Classes only -- see `classify_failure`. This dict is written to the repository.
    summary["failure_reasons"] = dict(collections.Counter(
        f.get("reason") or "ffmpeg-error" for f in failed))
    for f in failed:
        print(f"  FAILED {f['clip_id']}: {f['reason']}: {f['error']}", flush=True)

    (stage / "queue.json").write_text(
        json.dumps({"summary": summary, "clips": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    # Blindness by pipeline: `blind_listen.py` never serves these two files, and
    # releases their contents only after a label is committed for that clip.
    (stage / "witness.json").write_text(json.dumps(
        {r["clip_id"]: gv[r["origin_row_id"]]["soniox_text"] for r in rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    train_by_id = {r["id"]: r for r in train}
    (stage / "current_target.json").write_text(json.dumps(
        {r["clip_id"]: train_by_id[r["origin_row_id"]]["text"] for r in rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (stage / "selection_ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")

    gen_sha = sha256_file(stage / "queue.json")
    GENERATIONS.mkdir(parents=True, exist_ok=True)
    gen = GENERATIONS / gen_sha[:16]
    if gen.exists():
        shutil.rmtree(gen)
    os.replace(stage, gen)
    # os.symlink cannot replace an existing name, so the swap goes through a temporary
    # link and one rename -- which is atomic, unlike unlink-then-create.
    if CURRENT.exists() and not CURRENT.is_symlink():
        raise RuntimeError(
            f"{CURRENT} is a real directory, not the generation symlink. os.replace "
            "would fail here and leave the new generation unpublished; move it aside "
            "and re-run.")
    link_tmp = OUT_DIR / ".current.new"
    if link_tmp.exists() or link_tmp.is_symlink():
        link_tmp.unlink()
    link_tmp.symlink_to(gen, target_is_directory=True)
    os.replace(link_tmp, CURRENT)

    summary["generation"] = gen_sha[:16]
    summary["queue_sha256"] = gen_sha
    COUNTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return {"summary": summary, "queue": rows}


def excluded_recordings(train_rows, manifest: Manifest) -> set:
    """Source URLs that belong to at least one excluded meeting.

    `is_excluded` compares ids. A recording that also appears under a second, unexcluded
    meeting id -- a rename, a duplicate upload -- would pass that check while being the
    same audio. Comparing the recording itself closes it, and costs one pass over a
    manifest that is already in memory (Codex review 4cf77e98).
    """
    return {r["audio_url"] for r in train_rows
            if manifest.is_excluded(r["city_id"], r["meeting_id"])}


def verify() -> dict:
    """Re-check a published queue: exclusions, hashes, geometry, witness bijection."""
    manifest = load_manifest()
    d = json.loads((CURRENT / "queue.json").read_text(encoding="utf-8"))
    witness = json.loads((CURRENT / "witness.json").read_text(encoding="utf-8"))
    train, _ = load_pool()
    banned_urls = excluded_recordings(train, manifest)
    bad = {"excluded": [], "alias_recording": [], "missing": [], "hash": [],
           "bounds": [], "no_witness": [], "duration": [], "format": []}
    for r in d["clips"]:
        if manifest.is_excluded(r["city"], r["meeting_id"]):
            bad["excluded"].append(r["clip_id"])
        if r["audio_url"] in banned_urls:
            bad["alias_recording"].append(r["clip_id"])
        if r["clip_id"] not in witness:
            bad["no_witness"].append(r["clip_id"])
        p = CURRENT / "clips" / f"{r['clip_id']}.wav"
        if not p.exists():
            bad["missing"].append(r["clip_id"])
            continue
        if r.get("audio_sha256") and sha256_file(p) != r["audio_sha256"]:
            bad["hash"].append(r["clip_id"])
        m = probe_duration(p)
        if m is None or abs(_ms(m) - _ms(r["clip_dur"])) > DURATION_TOLERANCE_MS:
            bad["duration"].append(r["clip_id"])
        st = probe_stream(p)
        if st is None or any(str(st.get(k)) != str(v)
                             for k, v in EXPECT_STREAM.items()):
            bad["format"].append(r["clip_id"])
        if not (MIN_CLIP - 1e-6 <= r["clip_dur"] <= MAX_CLIP + 1e-6):
            bad["bounds"].append(r["clip_id"])
    extra = set(witness) - {r["clip_id"] for r in d["clips"]}
    return {"n": len(d["clips"]), "witness_without_clip": len(extra),
            "bad": {k: len(v) for k, v in bad.items()},
            "examples": {k: v[:5] for k, v in bad.items() if v}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "cut", "verify"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "verify":
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return
    out = build(limit=a.limit, cut=(a.mode == "cut"))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - operational entry point
    main()
