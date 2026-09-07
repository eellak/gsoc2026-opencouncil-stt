"""Cut Pool 2 of the `listen-2026-08` training round: adapter-vs-Soniox disagreement.

Spec: docs/specs/2026-08-26-training-round-2.md section 4.1, pool 2.
Record: exp-2026-08-26-training-round-2.
Companion to `cut_listening_clips.py`, which builds pool 1 (`deletion_hard`). The two
modules share the exclusion manifest, the clip geometry and the overlap rule, and they
select on different evidence.

WHAT POOL 2 IS, AND WHY IT COSTS NO BENCHMARK WINDOWS

Pool 1 selects a span because a human edited it and Soniox confirmed words the training
target lacks. Pool 2 selects a span because OUR ADAPTER and the Soniox witness disagree
about what was said there -- no human edit required. Both pools are cut from meetings
that are already council training material, so neither spends an evaluation window.

`chooser/cutclips.py` cannot be reused, and both of the spec's stated reasons were
re-checked against the code before this module was written:

  * its rows come from `chooser/deletions.py:20`, which keeps only windows whose city is
    in `autoprompt.A.CONFIRM_CITIES`. Those five cities are a WHOLE-CITY source of the
    exclusion manifest (`exclusion_manifest.CONFIRM_CITIES`), so every row it can
    produce is excluded -- the yield is exactly zero, not merely reduced.
  * its clip position is a column-fraction proxy, not a timestamp: `deletions.py:44`
    stores `pos = island_column / n_columns` and `cutclips.py:45` turns that into
    seconds as `startSec + pos * durationSec`. Its own docstring says so. A pool whose
    clip boundaries are guesses cannot carry the overlap rule, which is stated in
    seconds.

THE THREE MEETING-LEVEL SETS, AND WHY THEY MUST NOT MIX

  excluded      the hash-pinned exclusion manifest: 283 meetings and 5 whole cities of
                held-out, calibration, validation, gold and confirmation material. A
                clip from one of these is training on the test set. NO FURTHER CITY
                RELEASES ARE PERMITTED (user decision 2026-09-04).
  measurement   the listening queue's meetings that do NOT appear in any council
                training pack. The measurement design of 2026-09-06 draws its ordinary
                sample and its challenge sample from exactly these. A Pool 2 clip taken
                here would make one meeting both training material and evaluation
                material, which is the failure the whole redesign exists to remove.
  training side the meetings that already appear in `clean-pack-arm-2026-08`,
                `spliced-pack-arm-2026-08` or `spliced-pack-arm-jitter-2026-08`. Pool 2
                is confined to these. That is what "training-side meetings" means in
                section 4.1, and it is what makes the `measurement` set disjoint by
                construction rather than by a promise.

`(city, meeting_id)` is the key everywhere. `meeting_id` alone is not unique across
cities and collapses 117 real meetings into 66 (`exp_composition_postjune.py:212-216`).

WHAT THIS MODULE DOES NOT INVENT

The Soniox witness is not re-run. The only Soniox output this project holds on
training-side audio is the `gap-verify` cache -- every other Soniox cache
(`composition-rt-2026-08`, `gold-set-2026-08`) covers evaluation windows. So the
candidate substrate IS `gap-verify`: 12,946 spans on which Soniox actually ran, each
with the span it ran on (`gap-verify/api/<id>.json`) and a cached padded wav.

The served Soniox model is **`stt-rt-v5`**. Every call this project made sent the string
`stt-rt-v4`, but Soniox removed v4 on 2026-06-30 and routes the name, and every run of
ours is dated August 2026. `stt-rt-v5` output is never pooled with `stt-async-v5`.

THE INPUT THAT DID NOT EXIST YET

Our adapter's hypothesis on those spans. Nothing in `~/.cache/oc-public` held one when
this module was written, apart from 300 clean packs in `train-wer-v2-2026-08-n300`, and
none of the training-side adapter caches carries a timestamp. So `plan` reports the
funnel down to the STRUCTURAL CEILING -- everything that survives the filters which need
no model output -- and stops there with `adapter_hypotheses_missing` and a measured
decode cost. That is a measurement, not a failure.

`decode` produces the cache, and it decodes the ceiling rather than the whole substrate:
spending CPU on rows that can never become clips would also price the pool wrong.

`ADAPTER_HYP_DIR` is the contract for the cache: one `<row_id>.json` per span, holding
`{"text": ..., "artifact": ..., "model_sha256": ..., "config": {...}}`. Rows whose
`artifact` disagrees with `EXPECT_ARTIFACT` are refused rather than mixed -- two
adapters' hypotheses in one pool is the same error as pooling two Soniox models.

WHAT THIS MODULE DOES NOT DECIDE

How big the pool should be. The ELIGIBILITY rule is frozen (`del >= 1`, from section
4.1's own wording, written down before any hypothesis was decoded); the SEVERITY grid is
reporting; and `cut` refuses to run without `--i-have-chosen-the-size`. A threshold
picked off a yield curve after the curve was measured is a tuned threshold, and this
project's expensive mistakes have all been measurement mistakes.

FOUR THINGS THIS MODULE IS RESPONSIBLE FOR NOT GETTING WRONG

1. The measurement set is FROZEN, not recomputed. "The queue minus the training side"
   looks stable and is not: the training side grows every time an arm is built, so a
   meeting added to a future pack would quietly leave the evaluation design that was
   frozen on 2026-09-06. `load_measurement_set` reads a hashed file and refuses when the
   live derivation disagrees.

2. The drift check compares against a THIRD record. Pool 1 checks the training manifest
   against `gap-verify/api`; Pool 2 takes its span from that api file, so the same
   comparison would be vacuous. `TranscriptSpans` reads the live OpenCouncil dump
   instead, which is the record that can actually have moved.

3. The overlap rule either ran or the meeting is dropped. Six diarization jobs on this
   machine are zero-filled, so three eligible meetings have no reconstructable pack
   spans. Those meetings are refused rather than admitted under a weaker check, because
   a degraded check that reports success is worse than no clip.

4. Hypothesis, witness and clip are the same seconds. Soniox transcribed the PADDED wav
   (`gap2_verify_audio.py:263`), `decode` reads that same wav, and a row whose clip
   geometry would be narrower than it is dropped.

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

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exclusion_manifest import (  # noqa: E402
    Manifest, load_manifest)
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402

SC = Path.home() / ".cache" / "oc-public"

GAP_VERIFY = SC / "gap-verify" / "manifest.jsonl"
GAP_API = SC / "gap-verify" / "api"
GAP_CLIPS = SC / "gap-verify" / "clips"
TRAIN_MANIFEST = SC / "train-screens-2026-08" / "run1" / "manifest.jsonl"
# The pool-1 listening queue, pinned to the GENERATION file and its digest rather than
# to the mutable `current` symlink -- the same file and the same hash the ordinary and
# challenge samples pin, so all three modules define the 67 measurement meetings from
# identical bytes (Codex review `c0325025`).
LISTEN_QUEUE = (SC / "listen-2026-08" / "generations" / "438aaea941866e08"
                / "queue.json")
LISTEN_QUEUE_SHA256 = (
    "438aaea941866e087c0f25e02cd3a2b9b6badeea1c54ba3bda6b2400f4a0550a")
PACK_MANIFESTS = (
    SC / "clean-pack-arm-2026-08" / "manifest.jsonl",
    SC / "spliced-pack-arm-2026-08" / "manifest.jsonl",
    SC / "spliced-pack-arm-jitter-2026-08" / "manifest.jsonl",
)
PACK_SPANS = SC / "pool2-2026-09" / "pack-spans.json"
MEASUREMENT_SET = SC / "pool2-2026-09" / "measurement-meetings.json"
TRANSCRIPTS = SC / "training-transcripts-2026-08"
ADAPTER_HYP_DIR = SC / "pool2-2026-09" / "adapter-hyps"

OUT_DIR = SC / "pool2-2026-09"
GENERATIONS = OUT_DIR / "generations"
CURRENT = OUT_DIR / "current"
COUNTS_PATH = ROOT / "eval" / "results_pool2_queue.json"

# --------------------------------------------------------------------- frozen
POOL = "adapter_soniox_disagreement"
SOURCE = "pool2-2026-09"
WITNESS = "soniox-stt-rt-v5"        # the SERVED model; the string sent was stt-rt-v4
EXPECT_ARTIFACT = "artifact-ct2-cleanpack-cont-s47"

# Geometry, identical to pool 1 so the two queues can be overlap-checked against each
# other in one coordinate system (`cut_listening_clips.py`, PAD_MS / MIN_CLIP_MS /
# MAX_CLIP_MS). PAD is 3.0 s because that is the pad the gap-verify wavs on disk were
# cut with (`eval/gap2_verify_audio.py:43`, `:253`); matching it is what makes reuse
# exact rather than approximate.
PAD_MS = 3000
MIN_CLIP_MS = 5000
MAX_CLIP_MS = 15000
PAD = PAD_MS / 1000
MIN_CLIP = MIN_CLIP_MS / 1000
MAX_CLIP = MAX_CLIP_MS / 1000

WITNESS_SPAN_TOLERANCE_MS = 10
DURATION_TOLERANCE_MS = 500

# THE ELIGIBILITY RULE, frozen before a single hypothesis was decoded, and taken from
# the spec rather than invented here: section 4.1 opens with "short clips (5-15 s) where
# our adapter is wrong AND AN INDEPENDENT WITNESS HAS WORDS WE DO NOT". That is
# directional, so eligibility is `del >= 1` -- at least one witness word the adapter did
# not emit -- and NOT a total edit rate. A span where the adapter only added words is a
# different finding and does not belong in a pool whose labels are meant to recover
# speech the adapter lost.
MIN_WITNESS_ONLY_WORDS = 1

# The severity grid is REPORTING, not selection. It is written down here, before any
# number existed, so a sensitivity curve can be shown without the threshold on it
# becoming the rule; `--min-disagreement` is an OPTIONAL extra filter an operator may
# add on top of the frozen rule, never a substitute for it.
DISAGREEMENT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

# `build_clean_pack_arm.py:33` and `build_spliced_pack_arm.py:38` move each cut edge to
# the quietest frame within this window. The snap needs the audio and is not reproduced
# here, so every reconstructed pack interval is widened by it on both sides.
PACK_SNAP_SEARCH_SEC = 0.25

_CLIP_ID_RE = re.compile(r"^[a-z0-9]+__[a-z0-9_]+__\d+$")


class GeometryError(ValueError):
    """A clip span violates a frozen invariant. Never an assert: `python -O`."""


class ExclusionViolation(RuntimeError):
    """An excluded or measurement meeting reached the queue. Nothing continues."""


class ProvenanceError(RuntimeError):
    """A cache does not identify the model that produced it, or names the wrong one."""


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


def _ms(x) -> int:
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise GeometryError(f"non-finite timestamp: {x!r}")
    return int(round(v * 1000))


def clip_geometry(start: float, dur: float) -> Geometry | None:
    """The clip span for a row, or None when the row cannot be a 5-15 s clip.

    Same rule as pool 1, restated rather than imported: `cut_listening_clips.py` is
    another agent's file and is frozen while its queue is being listened to. The two
    copies are held identical by `test_pool2_geometry_matches_pool1`.
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
    return Geometry(start_ms=s, end_ms=e, pad_left_ms=left, pad_right_ms=right,
                    reuse_gap_verify=(left == PAD_MS and right == PAD_MS))


def clip_id(city: str, meeting: str, geom: Geometry) -> str:
    cid = f"{city}__{meeting}__{geom.start_ms}"
    if not _CLIP_ID_RE.match(cid):
        raise GeometryError(f"clip id is not path-safe: {cid!r}")
    return cid


# ------------------------------------------------------------------ disagreement

def disagreement(adapter_text: str, witness_text: str) -> dict:
    """How far our adapter is from the Soniox witness on this span.

    The witness is the denominator and the module says so everywhere it reports this
    number. That is a choice of unit, NOT a claim that Soniox is right: both systems
    are wrong here often and are wrong together more often than chance (odds ratio 17.8
    on the gold set, `docs/reports/2026-08-16-adapter-confidence.md:175-183`). What the
    rate measures is where a human should listen, and nothing else.

    `sub` / `del` / `ins` are kept apart rather than summed into one score, because the
    interesting direction is asymmetric: `del` is words the witness has and the adapter
    does not, which is the pool-1 signal arrived at from the other side, while `ins` is
    words only the adapter emitted. A selection rule that merges them cannot tell the
    two apart afterwards.

    Both texts go through the frozen `ftoks` (scoring tokenizer, minus hesitation
    fillers). `soniox_text` in `gap-verify` is already joined into words, so the
    sub-word-piece trap that gave a previous scorer recall 0.32 does not arise here --
    `group_words` ran upstream, in the client that wrote the cache.
    """
    ref = ftoks(witness_text or "")
    hyp = ftoks(adapter_text or "")
    s, d, i = sdi(ref, hyp)
    n = len(ref)
    return {
        "n_witness_tokens": n,
        "n_adapter_tokens": len(hyp),
        "sub": s, "del": d, "ins": i,
        "rate": (s + d + i) / n if n else None,
        "del_rate": d / n if n else None,
    }


# ------------------------------------------------------------------ meeting sets

def _jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def training_side_meetings(paths=PACK_MANIFESTS) -> frozenset:
    """(city, meeting) of every meeting that already trained an adapter.

    The union of the three council packs. A meeting here is training material already,
    so a Pool 2 clip from it adds no new contamination surface -- and it is exactly the
    set the measurement design refuses to draw from.
    """
    keys: set = set()
    for p in paths:
        for r in _jsonl(p):
            keys.add((r["city_id"], r["meeting_id"]))
    if not keys:
        raise ValueError("the council training packs resolved to no meetings")
    return frozenset(keys)


def derive_measurement_meetings(queue_path: Path = LISTEN_QUEUE,
                                training_side: frozenset | None = None) -> frozenset:
    """The listening-queue meetings the 2026-09-06 measurement design draws from.

    The queue's meetings MINUS the training side, which is the subtraction wayfinder
    #60 made to get 67 meetings / 449 clips. This is the DERIVATION; it is not what the
    filter reads at run time -- see `load_measurement_set`.
    """
    if training_side is None:
        training_side = training_side_meetings()
    raw = Path(queue_path).read_bytes()
    if queue_path == LISTEN_QUEUE:
        got = hashlib.sha256(raw).hexdigest()
        if got != LISTEN_QUEUE_SHA256:
            raise ValueError(
                f"{queue_path} hashes to {got}, not the frozen "
                f"{LISTEN_QUEUE_SHA256}. The measurement meetings are defined by those "
                "exact bytes; work out what moved before cutting anything.")
    d = json.loads(raw.decode("utf-8"))
    queue_keys = {(r["city"], r["meeting_id"]) for r in d["clips"]}
    if not queue_keys:
        raise ValueError(f"{queue_path} carries no clips")
    return frozenset(queue_keys - training_side)


def _measurement_payload(keys: frozenset) -> str:
    return json.dumps({"meeting_keys": sorted([c, m] for c, m in keys)},
                      ensure_ascii=False, sort_keys=True, indent=2)


def freeze_measurement_set(path: Path = MEASUREMENT_SET) -> dict:
    """Write the measurement meetings to a hashed file, once.

    Recomputing "the queue minus the training side" on every run looks safe and is not:
    the training side GROWS every time an arm is built. A meeting added to a future
    training pack would silently leave
    the measurement set, shrinking an evaluation design that was frozen on 2026-09-06 --
    after the fact, invisibly, in the direction that makes a result easier to get.
    So the set is frozen to a file, the file is hashed, and the hash is pinned in the
    code (Codex review `c0325025`).
    """
    keys = derive_measurement_meetings()
    body = _measurement_payload(keys)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(body, encoding="utf-8")
    return {"n_meetings": len(keys),
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "path": str(path)}


# The frozen measurement set of 2026-09-06: the 67 listening-queue meetings that no
# council training pack contains. `freeze_measurement_set` writes the file this hash
# describes.
FROZEN_MEASUREMENT_SHA256 = (
    "32a918cc840e0df8b584baa29fb79dbca8c430336a060f9242f66f2109253fe3")


def load_measurement_set(path: Path = MEASUREMENT_SET,
                         expect_sha: str = None,
                         cross_check: bool = True) -> frozenset:
    """Read the frozen measurement set, verify its hash, cross-check the derivation.

    The cross-check is one-directional on purpose. A key that has LEFT the derivation
    (its meeting has since entered a training pack) is an error: the frozen design still
    draws its ordinary sample from that meeting, and Pool 2 must keep refusing it. A key
    that has been ADDED is also an error, because the design is frozen. Either way the
    right response is a human looking at it, not a filter quietly changing size.
    """
    expect = FROZEN_MEASUREMENT_SHA256 if expect_sha is None else expect_sha
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if expect and got != expect:
        raise ValueError(
            f"the measurement set {path} hashes to {got}, not the frozen {expect}. "
            "Do not cut Pool 2 against it: work out what moved and re-record the hash.")
    keys = frozenset((c, m) for c, m in json.loads(raw.decode("utf-8"))["meeting_keys"])
    if not keys:
        raise ValueError(f"{path} carries no meetings")
    if cross_check:
        live = derive_measurement_meetings()
        if live != keys:
            raise ValueError(
                f"{path} holds {len(keys)} measurement meetings but the live "
                f"derivation gives {len(live)} "
                f"(+{len(live - keys)} / -{len(keys - live)}). The frozen evaluation "
                "design and the current training packs disagree; that has to be "
                "resolved by a person before Pool 2 is cut.")
    return keys


# ------------------------------------------------------------------ candidates

def load_witness_rows(manifest_path: Path = GAP_VERIFY, api_dir: Path = GAP_API):
    """Every span Soniox was run on, with the span it was actually run on.

    Two records per row and they are not the same thing. `manifest.jsonl` holds the
    witness text and the tier; `api/<id>.json` holds the span, the recording and the
    meeting AS THEY ARE NOW. Pool 1 measured 108 rows where those two have drifted
    apart, one by 13.25 s; `witness_matches` is where that is caught.
    """
    rows = []
    for r in _jsonl(manifest_path):
        p = Path(api_dir) / f"{r['id']}.json"
        if not p.exists():
            rows.append({"id": r["id"], "witness": r, "api": None})
            continue
        rows.append({"id": r["id"], "witness": r,
                     "api": json.loads(p.read_text(encoding="utf-8"))})
    return rows


def load_adapter_hyps(directory: Path = ADAPTER_HYP_DIR,
                      expect_artifact: str = EXPECT_ARTIFACT) -> dict:
    """`row_id -> text`, from a cache that has to name the model that wrote it.

    A hypothesis file with no `artifact` is refused rather than defaulted: an output
    whose producing model is unknown is not evidence, and two adapters' hypotheses in
    one pool is the same mistake as pooling two Soniox models.
    """
    out: dict[str, str] = {}
    d = Path(directory)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        art = payload.get("artifact")
        if not art:
            raise ProvenanceError(
                f"{p} carries no `artifact`; a hypothesis whose producing model is "
                "unknown is not evidence and must not enter a pool")
        if art != expect_artifact:
            raise ProvenanceError(
                f"{p} was produced by {art!r}, not {expect_artifact!r}. Two adapters' "
                "hypotheses must not be pooled; cut a separate pool per adapter")
        out[p.stem] = payload.get("text") or ""
    return out


def candidates(witness_rows, manifest: Manifest, training_side: frozenset,
               measurement: frozenset, adapter_hyps: dict):
    """Everything that could become a Pool 2 clip, with a drop reason for the rest.

    The meeting-level filters run before the row-level ones, so the funnel reads as
    "how much of the corpus is even allowed" before "how much of it is interesting".
    """
    drops = collections.Counter()
    kept = []
    ledger = []
    for row in witness_rows:
        rid = row["id"]
        api = row["api"]
        w = row["witness"]

        def drop(reason, **extra):
            drops[reason] += 1
            ledger.append({"row": rid, "drop": reason, **extra})

        if api is None:
            drop("no_api_provenance")
            continue
        city, meeting = api.get("city_id"), api.get("meeting_id")
        start, end = api.get("start"), api.get("end")
        if not city or not meeting or start is None or end is None:
            drop("no_span")
            continue
        text = (w.get("soniox_text") or "").strip()
        if not text or w.get("error"):
            drop("no_witness_text")
            continue
        if manifest.is_excluded(city, meeting):
            drop("excluded_meeting")
            continue
        if (city, meeting) in measurement:
            drop("measurement_meeting")
            continue
        if (city, meeting) not in training_side:
            drop("not_training_side")
            continue

        start_ms, end_ms = _ms(start), _ms(end)
        if end_ms <= start_ms:
            drop("bad_span")
            continue
        cand = {
            "origin_row_id": rid,
            "city": city,
            "meeting_id": meeting,
            "meeting_date": api.get("meeting_date"),
            "audio_url": api.get("audio_url"),
            "row_start": start_ms / 1000,
            "row_dur": (end_ms - start_ms) / 1000,
            "tier": w.get("tier"),
            "witness_text": text,
            "gap_verify_clip": w.get("clip_padded"),
            "geometry": clip_geometry(start_ms / 1000, (end_ms - start_ms) / 1000),
        }
        cand["adapter_text"] = adapter_hyps.get(rid)
        kept.append(cand)
    return kept, drops, ledger


class TranscriptSpans:
    """The CURRENT span of every utterance, read from the OpenCouncil transcript dumps.

    This is the half of the drift check that `gap-verify` cannot supply. Pool 1 compares
    the training manifest's span against `gap-verify/api/<id>.json`, which are two
    independent records of the same utterance; Pool 2 takes its span FROM that api file,
    so comparing it against itself would pass every row and detect nothing -- the check
    would look present and be vacuous (Codex review `c0325025`).

    `training-transcripts-2026-08/<city>__<meeting>.json` is the third record: the
    utterance as OpenCouncil held it when the dumps were taken. If it has moved since the
    api snapshot, the cached wav and `soniox_text` describe different audio, which is the
    108-row defect pool 1 measured. Loaded one meeting at a time: the dumps are 492 MB in
    total and Pool 2 needs about a tenth of them.

    THE LIMIT OF THIS CHECK, stated because a filter that fires zero times invites the
    wrong conclusion. Both records are SNAPSHOTS: the api files were written 2026-08-13
    to 08-15 and the transcript dumps on 2026-08-19. They agree to within 0.5 ms on all
    1,758 eligible spans, so nothing drifted in those four days -- but an edit made after
    2026-08-19 is invisible here. Re-pulling the dumps before a cut would close that, and
    the summary carries `transcript_dump_date` so the gap is visible rather than assumed
    away.
    """

    def __init__(self, root: Path = TRANSCRIPTS):
        self.root = Path(root)
        self._cache: dict[tuple[str, str], dict | None] = {}

    def meeting(self, city: str, meeting: str) -> dict | None:
        key = (city, meeting)
        if key not in self._cache:
            path = self.root / f"{city}__{meeting}.json"
            if not path.is_file():
                self._cache[key] = None
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._cache[key] = {
                    u["id"]: (float(u["startTimestamp"]), float(u["endTimestamp"]))
                    for seg in payload["transcript"]
                    for u in seg.get("utterances", [])}
        return self._cache[key]

    def span(self, city: str, meeting: str, utterance_id: str):
        m = self.meeting(city, meeting)
        if m is None:
            return None
        return m.get(utterance_id)


def witness_matches(cand: dict, current: tuple | None) -> bool:
    """Does the utterance STILL occupy the span Soniox was run on?

    `current` is `(start, end)` from the live transcript dump. Both endpoints are
    compared, not the length: a span of the same duration at a different offset is a
    different clip, and its witness belongs to somebody else. A missing meeting dump or
    a vanished utterance fails closed -- an utterance we cannot locate is not one we can
    prove has not moved.
    """
    if not current:
        return False
    try:
        ds = abs(_ms(current[0]) - _ms(cand["row_start"]))
        de = abs(_ms(current[1]) - _ms(cand["row_start"] + cand["row_dur"]))
    except (IndexError, GeometryError, TypeError):
        return False
    return max(ds, de) <= WITNESS_SPAN_TOLERANCE_MS


# ------------------------------------------------------------------ overlap rule

def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def occupied_spans(pack_spans_path: Path = PACK_SPANS,
                   train_manifest: Path = TRAIN_MANIFEST,
                   queue_path: Path = LISTEN_QUEUE) -> tuple[dict, dict]:
    """Every span in a meeting that a training or evaluation clip already occupies.

    Three sources, and they are not interchangeable:

      `train-screens-2026-08/run1/manifest.jsonl`  the stage-2 training rows, with
          spans. This is the source pool 1's overlap rule used.
      the council pack spans                        what the CURRENT adapters actually
          heard. The published pack manifests carry `span_sec` but NOT `start`/`end`,
          so these have to be reconstructed by `rebuild_pack_spans` and cached. A
          meeting whose spans could not be reconstructed is reported in the second
          return value rather than silently treated as empty -- an unreconstructed
          meeting looks identical to an unused one, and the difference is whether the
          overlap rule ran at all.
      the published listening queue                 pool 1's 721 clips.
    """
    index: dict[tuple[str, str], list[tuple[int, int, str]]] = collections.defaultdict(list)
    for r in _jsonl(train_manifest):
        index[(r["city_id"], r["meeting_id"])].append(
            (_ms(r["start"]), _ms(r["end"]), f"train:{r['id']}"))

    covered_by_packs: set = set()
    n_unpublished = 0
    if Path(pack_spans_path).exists():
        payload = json.loads(Path(pack_spans_path).read_text(encoding="utf-8"))
        for p in payload["packs"]:
            # Only packs that were actually CUT occupy the timeline. A pack this
            # reconstruction can derive but that the builder threw away was never heard
            # by any adapter, and indexing it would block Pool 2 out of free time
            # (Codex review `c0325025`).
            if not p.get("published"):
                n_unpublished += 1
                continue
            key = (p["city_id"], p["meeting_id"])
            covered_by_packs.add(key)
            index[key].append((_ms(p["start"]), _ms(p["end"]), f"pack:{p['pack_id']}"))

    if Path(queue_path).exists():
        d = json.loads(Path(queue_path).read_text(encoding="utf-8"))
        for r in d["clips"]:
            index[(r["city"], r["meeting_id"])].append(
                (_ms(r["clip_start"]), _ms(r["clip_end"]), f"listen:{r['clip_id']}"))

    for k in index:
        index[k].sort()
    return index, {"pack_spans_available_for": covered_by_packs,
                   "unpublished_packs_ignored": n_unpublished}


def _schedule(rows: list[dict]) -> tuple[list[dict], int]:
    """Pick the largest non-overlapping set of clips in one meeting.

    Weighted interval scheduling, not a pairwise greedy pass: with a chain
    `A-B, B-C, A/C disjoint` a greedy rule drops a clip it did not have to. The
    objective is (how many clips, then how much disagreement, then the
    lexicographically smaller set of row ids), so the answer is a single deterministic
    set rather than one of several equally good ones.
    """
    items = sorted(rows, key=lambda c: (c["geometry"].end_ms,
                                        c["geometry"].start_ms,
                                        c["origin_row_id"]))
    n = len(items)
    best: list[tuple] = [(0, 0.0, (), ())] * (n + 1)
    for i in range(1, n + 1):
        cur = items[i - 1]
        skip = best[i - 1]
        j = i - 1
        while j > 0 and items[j - 1]["geometry"].end_ms > cur["geometry"].start_ms:
            j -= 1
        prev = best[j]
        weight = (cur.get("disagreement") or {}).get("rate") or 0.0
        take = (prev[0] + 1, prev[1] + weight,
                prev[2] + (cur["origin_row_id"],), prev[3] + (i - 1,))
        key = lambda t: (t[0], t[1], tuple(-ord(ch) for ch in "".join(t[2])))  # noqa: E731
        best[i] = take if key(take) > key(skip) else skip
    chosen = [items[k] for k in best[n][3]]
    return chosen, n - len(chosen)


# ------------------------------------------------------------------ selection

def queue_row(cand: dict) -> dict:
    """The row a listening page loads. Provenance, no transcript."""
    g = cand["geometry"]
    if g is None:
        raise GeometryError("a row with no geometry never reaches the queue")
    dis = cand["disagreement"]
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
        "adapter": EXPECT_ARTIFACT,
        "witness_tier": cand["tier"],
        "disagreement_rate": round(dis["rate"], 4) if dis["rate"] is not None else None,
        "disagreement_del_rate": (
            round(dis["del_rate"], 4) if dis["del_rate"] is not None else None),
        "reused_gap_verify_clip": g.reuse_gap_verify,
    }


def select(cands, manifest: Manifest, measurement: frozenset, occupied: dict,
           transcripts, *, min_disagreement: float | None = None,
           overlap_checkable: frozenset | None = None,
           ignore_disagreement: bool = False):
    """Apply the row-level filters and return (queue rows, drop counts, ledger).

    `ignore_disagreement` runs everything EXCEPT the three filters that need an adapter
    hypothesis. It is how `plan` reports the structural ceiling of the pool -- how many
    clips the geometry, the witness provenance and the overlap rule leave standing -- on
    a day when no hypothesis exists. That ceiling is not a queue and is never cut: it is
    the upper bound any threshold will be applied to.
    """
    drops = collections.Counter()
    ledger: list[dict] = []
    kept = []
    for c in cands:
        rid = c["origin_row_id"]
        if not ignore_disagreement:
            if c.get("adapter_text") is None:
                drops["no_adapter_hypothesis"] += 1
                ledger.append({"row": rid, "drop": "no_adapter_hypothesis"})
                continue
            dis = disagreement(c["adapter_text"], c["witness_text"])
            c["disagreement"] = dis
            # The frozen rule: the witness has words we do not. Not a rate, and not
            # chosen after the yield was seen.
            if dis["del"] < MIN_WITNESS_ONLY_WORDS:
                drops["no_witness_only_words"] += 1
                ledger.append({"row": rid, "drop": "no_witness_only_words"})
                continue
            if min_disagreement is not None and (
                    dis["rate"] is None or dis["rate"] < min_disagreement):
                drops["below_severity_threshold"] += 1
                ledger.append({"row": rid, "drop": "below_severity_threshold"})
                continue
        else:
            c.setdefault("disagreement", {"rate": None, "del_rate": None})
        if c["geometry"] is None:
            drops["too_long"] += 1
            ledger.append({"row": rid, "drop": "too_long"})
            continue
        # The witness text and the adapter hypothesis both describe the PADDED span
        # `[start - 3 s, end + 3 s]` -- that is the clip `gap2_verify_audio.py:263,267`
        # sent to Soniox and the clip `decode_spans` reads. A row longer than 9 s gets
        # a shrunken pad budget, so its clip would be NARROWER than the audio the two
        # hypotheses describe, and the disagreement would be attributed to seconds the
        # listener never hears. Those rows are dropped rather than re-scored: one
        # coordinate system for the clip, the witness and the hypothesis, or none.
        if not c["geometry"].reuse_gap_verify:
            drops["clip_narrower_than_witness_span"] += 1
            ledger.append({"row": rid, "drop": "clip_narrower_than_witness_span"})
            continue
        if not witness_matches(c, transcripts.span(c["city"], c["meeting_id"], rid)):
            drops["witness_span_mismatch"] += 1
            ledger.append({"row": rid, "drop": "witness_span_mismatch"})
            continue
        # A meeting whose pack spans could not be reconstructed cannot be overlap
        # checked, and a degraded check that reports success is worse than no clip:
        # the whole meeting is dropped rather than admitted under a weaker rule
        # (Codex review `c0325025`).
        if (overlap_checkable is not None
                and (c["city"], c["meeting_id"]) not in overlap_checkable):
            drops["overlap_not_checkable"] += 1
            ledger.append({"row": rid, "drop": "overlap_not_checkable"})
            continue
        g = c["geometry"]
        hits = [i for s, e, i in occupied.get((c["city"], c["meeting_id"]), ())
                if _overlaps((g.start_ms, g.end_ms), (s, e))]
        if hits:
            drops["existing_clip_overlap"] += 1
            ledger.append({"row": rid, "drop": "existing_clip_overlap",
                           "with": hits[:3]})
            continue
        kept.append(c)

    by_meeting: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for c in kept:
        by_meeting[(c["city"], c["meeting_id"])].append(c)

    chosen: list[dict] = []
    for key in sorted(by_meeting):
        take, lost = _schedule(by_meeting[key])
        drops["queue_overlap"] += lost
        taken = {c["origin_row_id"] for c in take}
        for c in by_meeting[key]:
            if c["origin_row_id"] not in taken:
                ledger.append({"row": c["origin_row_id"], "drop": "queue_overlap"})
        chosen.extend(take)

    chosen.sort(key=lambda c: (c["city"], c["meeting_id"], c["geometry"].start_ms))
    out = [queue_row(c) for c in chosen]

    ids = [r["clip_id"] for r in out]
    if len(set(ids)) != len(ids):
        dup = [i for i, n in collections.Counter(ids).items() if n > 1]
        raise GeometryError(f"clip id collision: {dup[:3]}")
    # Checked again over the OUTPUT, with a raised exception rather than an assert,
    # because `python -O` deletes asserts and this is the check that stands between a
    # training clip and a contaminated benchmark.
    for row in out:
        if manifest.is_excluded(row["city"], row["meeting_id"]):
            raise ExclusionViolation(
                f"excluded meeting {row['city']}/{row['meeting_id']} reached the queue")
        if (row["city"], row["meeting_id"]) in measurement:
            raise ExclusionViolation(
                f"measurement meeting {row['city']}/{row['meeting_id']} reached the "
                "queue; a meeting cannot be training material and evaluation material")
    for c in chosen:
        ledger.append({"row": c["origin_row_id"], "drop": None,
                       "clip_id": clip_id(c["city"], c["meeting_id"], c["geometry"])})
    return out, drops, ledger


# ------------------------------------------------------------------ pack spans

def rebuild_pack_spans(out_path: Path = PACK_SPANS) -> dict:
    """Recover the (start, end) of every published council pack, and cache it.

    The pack manifests carry `span_sec` and not the span, so the only way to place a
    pack on a timeline is to re-derive it from the diarization jobs and the transcripts
    the builders read (`build_clean_pack_arm.py:64-105`,
    `build_spliced_pack_arm.py:84-155`). A reconstructed pack is matched back to the
    published manifests by `pack_id` -- the sha256 the builders compute from the
    recording URL and the first utterance id -- so a pack that this function can derive
    but that was never actually cut does not enter the overlap rule.

    BOTH arms are reconstructed and they occupy the timeline differently. A CONTIGUOUS
    pack is one span, so it contributes one interval. A SPLICED pack is a set of
    utterances glued together with synthetic silence, so it contributes ONE INTERVAL
    PER UTTERANCE -- the silence between them is not audio the adapter heard. Treating
    a spliced pack as the interval from its first to its last utterance would block
    Pool 2 out of gaps that are genuinely free, and treating it as one point would
    let a clip be cut on top of speech that is already a training row.

    Corrupt diarization jobs are counted, not skipped silently: their meetings have no
    reconstructable pack spans, and a caller that does not know that would apply a
    weaker overlap rule while believing it applied the full one.
    """
    from eval.controlled_eval import overlap_clip_census as OC

    diar = SC / "training-diarization-2026-08"
    transcripts = SC / "training-transcripts-2026-08"
    jobs: dict[str, dict] = {}
    corrupt: list[str] = []
    for path in sorted((diar / "jobs").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            corrupt.append(path.name)
            continue
        turns = [(t["speaker"], float(t["start"]), float(t["end"]))
                 for t in payload["output"]["diarization"]]
        jobs[payload["audio_url"]] = {
            "turns": turns,
            "audio_end": max((e for _, _, e in turns), default=0.0),
            "city_id": payload["city_id"],
            "meeting_id": payload["meeting_id"],
            "audio_sha256": payload["audio_sha256"],
        }
    rows, _ = OC.census(OC.transcript_rows(transcripts, jobs), jobs)

    # Two different identities, because the two builders wrote two different files.
    # The spliced manifest's `pack_id` IS `sha256(url|first_utterance|spliced)[:16]`
    # and its `audio` is that same digest. The CLEAN manifest's `pack_id` is something
    # else entirely -- its `audio` is `sha256(url|first_utterance)[:16]`, and the two
    # fields disagree on every row -- so a clean pack has to be matched on its wav name
    # (`build_clean_pack_arm.py:101-103`), not on `pack_id`. Matching clean packs on
    # `pack_id` silently matches nothing, which looks exactly like "this meeting has no
    # packs" and would switch the overlap rule off for the whole contiguous arm.
    published_ids: set = set()
    published_audio: set = set()
    for p in PACK_MANIFESTS:
        for r in _jsonl(p):
            if r.get("pack_id"):
                published_ids.add(r["pack_id"])
            if r.get("audio"):
                published_audio.add(Path(r["audio"]).stem)

    out = []
    contiguous = OC.contiguous_packs(rows, {u: j["turns"] for u, j in jobs.items()},
                                     {u: j["audio_end"] for u, j in jobs.items()})
    for p in contiguous:
        pid = hashlib.sha256(
            f"{p['audio_url']}|{p['utterance_ids'][0]}".encode()).hexdigest()[:16]
        # The published clip's edges were moved acoustically by up to SNAP_SEARCH
        # (`build_clean_pack_arm.py:33,49-57`), and the snap is not reproducible without
        # decoding the audio. The interval is widened by that amount on both sides so
        # the overlap rule errs toward dropping a Pool 2 clip rather than toward cutting
        # one on top of training audio.
        out.append({"pack_id": pid, "arm": "contiguous", "city_id": p["city_id"],
                    "meeting_id": p["meeting_id"],
                    "start": round(max(0.0, p["start"] - PACK_SNAP_SEARCH_SEC), 3),
                    "end": round(p["end"] + PACK_SNAP_SEARCH_SEC, 3),
                    "published": pid in published_audio})

    # Same arguments the spliced builder used (`build_spliced_pack_arm.py:97-101`),
    # including its 80%-of-target filter. A pack this function derives that the builder
    # would have thrown away is not on the timeline and must not block a Pool 2 clip.
    n_spliced = 0
    spliced = [p for p in OC.build_packs(rows, target_speech_sec=22.0,
                                         separator_sec=0.4)
               if OC.pack_speech_sec(p) >= 22.0 * 0.8]
    for pack in spliced:
        head = pack[0]
        pid = hashlib.sha256(
            f"{head['audio_url']}|{head['utterance_id']}|spliced".encode()).hexdigest()[:16]
        n_spliced += 1
        for r in pack:
            out.append({"pack_id": pid, "arm": "spliced", "city_id": r["city_id"],
                        "meeting_id": r["meeting_id"],
                        "start": round(max(0.0, r["start"] - PACK_SNAP_SEARCH_SEC), 3),
                        "end": round(r["end"] + PACK_SNAP_SEARCH_SEC, 3),
                        "published": pid in published_ids})

    reconstructed_meetings = {(p["city_id"], p["meeting_id"]) for p in out}
    pack_meetings = training_side_meetings()
    payload = {
        "packs": out,
        "corrupt_diarization_jobs": corrupt,
        "n_intervals": len(out),
        "n_contiguous_packs": len(contiguous),
        "n_spliced_packs": n_spliced,
        "n_matched_published": len({p["pack_id"] for p in out if p["published"]}),
        "n_published_pack_ids": len(published_ids),
        "n_published_pack_wavs": len(published_audio),
        "meetings_without_reconstructable_spans": sorted(
            [c, m] for c, m in (pack_meetings - reconstructed_meetings)),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {k: v for k, v in payload.items() if k != "packs"}


# ------------------------------------------------------------------ operational

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


_FAILURE_CLASSES = (
    ("http-fetch-failed", ("http error", "403", "404", "server returned", "connection")),
    ("source-missing", ("no such file", "does not exist")),
    ("decode-failed", ("invalid data", "moov atom", "error while decoding")),
    ("seek-failed", ("could not seek", "seek to")),
)


def classify_failure(stderr: str) -> str:
    """Coarse on purpose: ffmpeg writes the recording URL into its own errors, and the
    failure summary is written to a file that lives IN GIT."""
    low = (stderr or "").lower()
    for name, needles in _FAILURE_CLASSES:
        if any(n in low for n in needles):
            return name
    return "ffmpeg-error"


def cut_one(cand: dict, dst: Path) -> dict:
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


# ------------------------------------------------------------------ the missing input

# The local CPU decode that produces the hypothesis cache. Same model directory, same
# CTranslate2 compute type and the same decode options as the v2 training-WER run
# (`train_wer_v2.py`), so a Pool 2 hypothesis and that run's hypotheses are the same
# measurement of the same adapter -- and neither is comparable to a GPU float16 decode.
DECODE_MODEL_DIR = Path("/home/harold/oc-asr-serve/ct2-v2")
DECODE_CONFIG = {
    "language": "el",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "word_timestamps": False,
    "vad_filter": False,
    "task": "transcribe",
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
}
DECODE_SEED = 20260906


def decode_order(cands) -> list[dict]:
    """A frozen, model-blind order over the eligible spans.

    `limit` takes a prefix of this, so a pilot is a random sample of the pool rather
    than whichever meetings happen to sort first -- the cities are very unevenly
    represented and an alphabetical prefix would be athens only.
    """
    def rank(c):
        return hashlib.sha256(
            f"{DECODE_SEED}|{c['origin_row_id']}".encode()).hexdigest()
    return sorted(cands, key=rank)


def wav_seconds(path: Path) -> float | None:
    """Duration of a 16 kHz mono 16-bit wav from its size, with no ffprobe call.

    Used to price the decode over thousands of files. `probe_duration` is the check
    that a clip IS what it claims; this is the arithmetic that says how long it is.
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        return None
    return max(0.0, (size - 44) / (16000 * 2))


def decode_cost(cands) -> dict:
    """What the missing input costs, measured on the wavs that would be decoded."""
    secs, missing = 0.0, 0
    for c in cands:
        w = c.get("gap_verify_clip")
        d = wav_seconds(w) if w else None
        if d is None:
            missing += 1
        else:
            secs += d
    return {"spans": len(cands), "audio_hours": round(secs / 3600, 3),
            "cached_wav_missing": missing}


def decode_spans(cands, limit: int | None = None, threads: int = 4) -> dict:
    """Write one hypothesis file per eligible span, resuming an interrupted run.

    Decodes the CACHED PADDED WAV, not the exact span: the padded clip is what
    `gap2_verify_audio.py:263` sent to Soniox, so the hypothesis, the witness and the
    audio a listener will hear are the same seconds. A row whose cached wav is missing
    is skipped and counted rather than re-cut from the recording -- a re-cut clip is a
    different clip and its witness would belong to somebody else.

    The caller passes the spans that have ALREADY survived every outcome-independent
    filter. Decoding the whole substrate and filtering afterwards would spend CPU on
    rows that can never become clips and would price the pool wrong (Codex review
    `c0325025`).
    """
    from faster_whisper import WhisperModel

    ADAPTER_HYP_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    skipped_no_wav = 0
    for c in decode_order(cands):
        dst = ADAPTER_HYP_DIR / f"{c['origin_row_id']}.json"
        if dst.exists():
            continue
        wav = Path(c["gap_verify_clip"]) if c.get("gap_verify_clip") else None
        if not wav or not wav.exists():
            skipped_no_wav += 1
            continue
        todo.append((c, wav, dst))
        if limit and len(todo) >= limit:
            break

    model = WhisperModel(str(DECODE_MODEL_DIR), device="cpu", compute_type="int8",
                         cpu_threads=threads)
    model_sha = sha256_file(DECODE_MODEL_DIR / "model.bin")
    done = 0
    for c, wav, dst in todo:
        opts = dict(DECODE_CONFIG)
        opts["temperature"] = tuple(opts["temperature"])
        segments, _ = model.transcribe(str(wav), **opts)
        text = "".join(s.text for s in segments).strip()
        # Written through a temporary name and one rename. A decode of this size gets
        # interrupted, and a half-written hypothesis file is worse than a missing one:
        # the resume skips anything that exists, so a truncated file would be kept
        # forever and would fail to parse at the least convenient moment.
        tmp = dst.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "text": text,
            "artifact": EXPECT_ARTIFACT,
            "model_dir": str(DECODE_MODEL_DIR),
            "model_sha256": model_sha,
            "device": "cpu",
            "compute_type": "int8",
            "audio": str(wav),
            "audio_sha256": sha256_file(wav),
            "config": DECODE_CONFIG,
        }, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dst)
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(todo)}", flush=True)
    return {"decoded": done, "queued": len(todo), "skipped_no_cached_wav": skipped_no_wav,
            "model_sha256": model_sha, "out": str(ADAPTER_HYP_DIR)}


def yield_curve(cands) -> list[dict]:
    """Clips, hours and meetings at every threshold of the frozen grid.

    Every row here has ALREADY passed the frozen eligibility rule (`del >= 1`) and every
    outcome-independent filter. The grid is a severity axis on top of that rule, offered
    so somebody can see how the pool shrinks as the disagreement gets worse -- it is
    sensitivity reporting, and the row at threshold 0.0 is the pool the frozen rule
    selects. How much of it to listen to is the operator's decision.
    """
    scored = [c for c in cands
              if c.get("adapter_text") is not None
              and (c.get("disagreement") or {}).get("rate") is not None
              and c["disagreement"]["del"] >= MIN_WITNESS_ONLY_WORDS]
    out = []
    for t in DISAGREEMENT_GRID:
        hit = [c for c in scored if c["disagreement"]["rate"] >= t]
        out.append({
            "threshold": t,
            "spans": len(hit),
            "meetings": len({(c["city"], c["meeting_id"]) for c in hit}),
            "cities": len({c["city"] for c in hit}),
            "span_hours": round(sum(c["row_dur"] for c in hit) / 3600, 4),
        })
    return out


def survivors(cands, manifest, measurement, occupied, transcripts,
              overlap_checkable):
    """Everything that passes every filter that does NOT need a model output.

    Geometry, witness-span drift, overlap-checkability and the overlap rule itself are
    all properties of the audio and the calendar. Running them first means the decode
    is spent only on rows that could become clips, and it means the reported ceiling is
    a real upper bound rather than an aspiration.
    """
    rows, drops, ledger = select(cands, manifest, measurement, occupied, transcripts,
                                 min_disagreement=None,
                                 overlap_checkable=overlap_checkable,
                                 ignore_disagreement=True)
    kept = {r["origin_row_id"] for r in rows}
    for c in cands:
        c.pop("disagreement", None)
    return [c for c in cands if c["origin_row_id"] in kept], drops, ledger


def build(min_disagreement: float | None = None, limit: int | None = None,
          cut: bool = False) -> dict:
    manifest = load_manifest()          # hash-pinned and cross-checked against sources
    training_side = training_side_meetings()
    measurement = load_measurement_set()
    if training_side & measurement:
        raise ExclusionViolation(
            "the training side and the measurement set share a meeting; one of the two "
            "definitions moved and Pool 2 must not be cut until it is worked out")

    witness_rows = load_witness_rows()
    adapter_hyps = load_adapter_hyps()
    cands, meeting_drops, meeting_ledger = candidates(
        witness_rows, manifest, training_side, measurement, adapter_hyps)

    occupied, occupancy_notes = occupied_spans()
    checkable = frozenset(occupancy_notes["pack_spans_available_for"])
    pack_spans_missing = sorted(f"{c}__{m}" for c, m in (training_side - checkable))
    transcripts = TranscriptSpans()

    summary = {
        "source": SOURCE,
        "pool": POOL,
        "witness": WITNESS,
        "adapter": EXPECT_ARTIFACT,
        "eligibility_rule": (
            f"del >= {MIN_WITNESS_ONLY_WORDS} witness-only word(s), frozen before any "
            "hypothesis was decoded (spec section 4.1)"),
        "exclusion_manifest_sha256": sha256_file(
            SC / "exclusion-manifest-2026-08" / "manifest.json"),
        "measurement_set_sha256": sha256_file(MEASUREMENT_SET),
        "pack_spans_sha256": sha256_file(PACK_SPANS)[:16] if PACK_SPANS.exists() else None,
        "gap_verify_sha256": sha256_file(GAP_VERIFY)[:16],
        # The drift check compares two snapshots, not the live database. An edit made
        # after this date is invisible to it.
        "transcript_dump_date": __import__("datetime").date.fromtimestamp(
            max(p.stat().st_mtime for p in TRANSCRIPTS.glob("*.json"))).isoformat(),
        "pack_manifest_sha256": {p.parent.name: sha256_file(p)[:16]
                                 for p in PACK_MANIFESTS},
        "n_witness_rows": len(witness_rows),
        "n_training_side_meetings": len(training_side),
        "n_measurement_meetings": len(measurement),
        "meeting_level_drops": dict(meeting_drops),
        "n_eligible_spans": len(cands),
        "eligible_meetings": len({(c["city"], c["meeting_id"]) for c in cands}),
        "eligible_cities": dict(collections.Counter(c["city"] for c in cands)),
        "eligible_tiers": dict(collections.Counter(c["tier"] for c in cands)),
        "eligible_span_hours": round(sum(c["row_dur"] for c in cands) / 3600, 4),
        "pack_span_meetings_unreconstructable": pack_spans_missing,
        "unpublished_packs_ignored": occupancy_notes["unpublished_packs_ignored"],
    }

    # Everything that survives the filters that need no model output. This is the
    # structural ceiling of the pool and the exact set the decode has to cover.
    ceiling, ceiling_drops, _ = survivors(
        cands, manifest, measurement, occupied, transcripts, checkable)
    summary["structural_ceiling"] = {
        "drops": dict(ceiling_drops),
        "spans": len(ceiling),
        "clip_hours": round(
            sum(c["geometry"].dur for c in ceiling) / 3600, 4),
        "meetings": len({(c["city"], c["meeting_id"]) for c in ceiling}),
        "cities": dict(collections.Counter(c["city"] for c in ceiling)),
        "tiers": dict(collections.Counter(c["tier"] for c in ceiling)),
        "meetings_per_city": {
            city: len({m for c, m in {(x["city"], x["meeting_id"]) for x in ceiling}
                       if c == city})
            for city in sorted({c["city"] for c in ceiling})},
        "largest_meeting_share": (
            round(max(collections.Counter(
                (c["city"], c["meeting_id"]) for c in ceiling).values()) / len(ceiling),
                4) if ceiling else None),
        "largest_city_share": (
            round(max(collections.Counter(c["city"] for c in ceiling).values())
                  / len(ceiling), 4) if ceiling else None),
        "note": ("this is not a queue and is never cut. It is the upper bound the "
                 "eligibility rule is applied to. The pool is four municipalities, so "
                 "no result from it may be read as generalising across cities."),
    }

    n_scored = sum(1 for c in ceiling if c.get("adapter_text") is not None)
    summary["n_adapter_hypotheses"] = len(adapter_hyps)
    summary["n_ceiling_spans_with_adapter_hypothesis"] = n_scored
    summary["adapter_hypotheses_missing"] = len(ceiling) - n_scored
    for c in ceiling:
        if c.get("adapter_text") is not None:
            c["disagreement"] = disagreement(c["adapter_text"], c["witness_text"])

    if n_scored == 0:
        # Not a failure. Pool 2's remaining input is a decode, and this is what it
        # costs; saying so in the summary is the finding.
        summary["status"] = "blocked_on_adapter_hypotheses"
        summary["missing_input"] = {
            "what": f"a hypothesis from {EXPECT_ARTIFACT} for each surviving span",
            "where_it_would_go": str(ADAPTER_HYP_DIR),
            "cost": decode_cost(ceiling),
            "no_new_soniox_calls_required": True,
        }
        return {"summary": summary, "queue": []}

    summary["yield_curve"] = yield_curve(ceiling)
    # A curve measured on part of the pool is a PILOT and is labelled one. Reading a
    # pilot curve as the pool's is how a sample of 80 turns into a plan for 1,758.
    summary["yield_curve_basis"] = {
        "spans_scored": n_scored,
        "spans_in_ceiling": len(ceiling),
        "covers_the_whole_ceiling": n_scored == len(ceiling),
        "sampling": ("frozen sha256 order over origin_row_id, seed "
                     f"{DECODE_SEED} -- model-blind and city-blind"),
        "eligible_under_the_frozen_rule": sum(
            1 for c in ceiling
            if (c.get("disagreement") or {}).get("del", 0) >= MIN_WITNESS_ONLY_WORDS),
    }
    if n_scored != len(ceiling):
        summary["status"] = "pilot_only"
        summary["note"] = (
            f"only {n_scored} of {len(ceiling)} surviving spans have an adapter "
            "hypothesis, so nothing is selected. The numbers above are a pilot: they "
            "estimate the eligible fraction, they are not the pool.")
        summary["remaining_cost"] = decode_cost(
            [c for c in ceiling if c.get("adapter_text") is None])
        return {"summary": summary, "queue": []}

    rows, drops, ledger = select(ceiling, manifest, measurement, occupied, transcripts,
                                 min_disagreement=min_disagreement,
                                 overlap_checkable=checkable)
    if limit:
        rows = rows[:limit]
    summary["status"] = "selected"
    summary["min_disagreement"] = min_disagreement
    summary["row_level_drops"] = dict(drops)
    summary["n_clips"] = len(rows)
    summary["clip_hours"] = round(sum(r["clip_dur"] for r in rows) / 3600, 4)
    summary["cities"] = dict(collections.Counter(r["city"] for r in rows))
    summary["meetings"] = len({(r["city"], r["meeting_id"]) for r in rows})
    summary["tiers"] = dict(collections.Counter(r["witness_tier"] for r in rows))
    summary["reused_gap_verify"] = sum(1 for r in rows if r["reused_gap_verify_clip"])

    if not cut:
        return {"summary": summary, "queue": rows}

    by_row = {c["origin_row_id"]: c for c in ceiling}
    stage = OUT_DIR / "_staging"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "clips").mkdir(parents=True)
    failed = []
    for i, row in enumerate(rows, 1):
        cand = by_row[row["origin_row_id"]]
        dst = stage / "clips" / f"{row['clip_id']}.wav"
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
    summary["failure_reasons"] = dict(collections.Counter(
        f.get("reason") or "ffmpeg-error" for f in failed))

    (stage / "queue.json").write_text(
        json.dumps({"summary": summary, "clips": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    # Blindness by pipeline: neither text is in `queue.json`, and a listening page
    # releases these two only after a label is committed for that clip.
    (stage / "witness.json").write_text(json.dumps(
        {r["clip_id"]: by_row[r["origin_row_id"]]["witness_text"] for r in rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (stage / "adapter_hypothesis.json").write_text(json.dumps(
        {r["clip_id"]: by_row[r["origin_row_id"]]["adapter_text"] for r in rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (stage / "selection_ledger.json").write_text(
        json.dumps(meeting_ledger + ledger, ensure_ascii=False), encoding="utf-8")

    gen_sha = sha256_file(stage / "queue.json")
    GENERATIONS.mkdir(parents=True, exist_ok=True)
    gen = GENERATIONS / gen_sha[:16]
    if gen.exists():
        shutil.rmtree(gen)
    os.replace(stage, gen)
    if CURRENT.exists() and not CURRENT.is_symlink():
        raise RuntimeError(
            f"{CURRENT} is a real directory, not the generation symlink; move it aside")
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


def verify() -> dict:
    """Re-check a published queue: exclusions, measurement set, hashes, geometry."""
    manifest = load_manifest()
    training_side = training_side_meetings()
    measurement = load_measurement_set()
    d = json.loads((CURRENT / "queue.json").read_text(encoding="utf-8"))
    witness = json.loads((CURRENT / "witness.json").read_text(encoding="utf-8"))
    bad = {"excluded": [], "measurement": [], "not_training_side": [], "missing": [],
           "hash": [], "bounds": [], "no_witness": [], "duration": [], "format": []}
    for r in d["clips"]:
        key = (r["city"], r["meeting_id"])
        if manifest.is_excluded(*key):
            bad["excluded"].append(r["clip_id"])
        if key in measurement:
            bad["measurement"].append(r["clip_id"])
        if key not in training_side:
            bad["not_training_side"].append(r["clip_id"])
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
        if st is None or any(str(st.get(k)) != str(v) for k, v in EXPECT_STREAM.items()):
            bad["format"].append(r["clip_id"])
        if not (MIN_CLIP - 1e-6 <= r["clip_dur"] <= MAX_CLIP + 1e-6):
            bad["bounds"].append(r["clip_id"])
    return {"n": len(d["clips"]),
            "bad": {k: len(v) for k, v in bad.items()},
            "examples": {k: v[:5] for k, v in bad.items() if v}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "cut", "verify", "packspans", "decode",
                                     "freeze-measurement-set"])
    ap.add_argument("--min-disagreement", type=float, default=None,
                    help="select spans at or above this adapter-vs-witness rate. No "
                         "default: `plan` without it reports the yield curve only.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--i-have-chosen-the-size", action="store_true",
                    help="required by `cut`. The frozen rule decides which spans are "
                         "eligible; how many of them to listen to is a decision a "
                         "person makes after reading `plan`.")
    a = ap.parse_args()
    if a.mode == "verify":
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return
    if a.mode == "packspans":
        print(json.dumps(rebuild_pack_spans(), ensure_ascii=False, indent=2))
        return
    if a.mode == "freeze-measurement-set":
        print(json.dumps(freeze_measurement_set(), ensure_ascii=False, indent=2))
        return
    if a.mode == "decode":
        manifest = load_manifest()
        training_side = training_side_meetings()
        measurement = load_measurement_set()
        cands, _, _ = candidates(load_witness_rows(), manifest, training_side,
                                 measurement, {})
        occupied, notes = occupied_spans()
        checkable = frozenset(notes["pack_spans_available_for"])
        keep, _, _ = survivors(cands, manifest, measurement, occupied,
                               TranscriptSpans(), checkable)
        print(json.dumps(decode_spans(keep, limit=a.limit, threads=a.threads),
                         ensure_ascii=False, indent=2))
        return
    if a.mode == "cut" and not a.i_have_chosen_the_size:
        raise SystemExit(
            "cut refuses to run without --i-have-chosen-the-size. Run `plan`, put the "
            "yield in front of a person, and let them decide how much of it to listen "
            "to. This module does not choose the size of the pool.")
    out = build(min_disagreement=a.min_disagreement, limit=a.limit,
                cut=(a.mode == "cut"))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - operational entry point
    main()
