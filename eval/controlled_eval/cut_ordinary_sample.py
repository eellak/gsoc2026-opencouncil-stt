"""Cut the ORDINARY SAMPLE v2: 45 clips of 20 s, 3 in each of 15 meetings.

Record: exp-2026-08-26-training-round-2. Design frozen 2026-09-06 in wayfinder #59,
built under #64, **frame replaced 2026-09-06** after the v1 frame was found to be about
to become training-contaminated. Terms: CONTEXT.md ("Ordinary sample", "Blind preference
judgement").

WHAT THIS IS FOR

An ordinary sample is the only listening evidence that may support "the model is better
overall". Its defining property is not its size and not its city mix: it is that
**nobody looked at where a model failed before choosing the positions, and nobody
trained on the meetings**.

WHY THE FRAME MOVED, AND WHAT IT COST

v1 drew from the 67 untouched meetings of the `listen-2026-08` queue. Two things were
wrong with that frame, one known at the time and one found afterwards:

  * it was FAILURE-SELECTED. The queue's 95 meetings were reached through `deletion_hard`
    rows, and v1's city weights were counts of failure-selected clips. Positions were
    independent of model output; the frame was not.
  * it was about to be TRAINING-CONTAMINATED. 376 of the 601 assisted training clips sit
    inside 64 of those 67 meetings, so the round being judged would have trained on all
    but 3 of the meetings judging it.

v2 draws from `~/.cache/oc-public/meetings/` instead: 151 released council meetings,
disjoint from the listening queue by `(city, meeting_id)` AND by recording URL (measured:
overlap exactly 0 on both). Nothing in this frame has any connection to where any model
failed, so the v1 caveat is gone rather than restated.

The price is stated up front rather than discovered later, and it is the real cost of the
move: **15 meeting clusters against v1's 67.** At that cluster count the decision rule's
interval is not merely wider -- it is a different order of precision, and only a large
effect will be visible at all. `PRECISION_NOTE` carries that sentence into the plan
output and into the manifest.

THE CRITICAL CONSTRAINT, AND HOW IT IS ENFORCED STRUCTURALLY

This module must never read model output, correction rows, Soniox agreement, published
transcript text, or anything derived from where any model failed. Four mechanisms, none
of them a comment:

1. `ALLOWED_INPUTS` is a frozen list and `_open_input` refuses any other path. Every read
   in this module goes through it, so a future edit that reaches for a hypothesis dump or
   a gap-verify witness raises `InputNotAllowed` instead of quietly working.
   `test_ordinary_sample.py` parses this module's own source to keep it that way.

2. **A meeting file carries its published transcript, and that transcript is partly ASR
   output.** Each file is projected immediately into `Meeting`, a frozen four-field
   dataclass with `slots`, so `transcript`, `subjects`, `speakerTags` and `taskStatus`
   have nowhere to go. Reading any of them to place a clip would reopen exactly the leak
   this design exists to close. `_project` raises if the whitelist stops matching, and
   `test_the_published_transcript_never_survives_the_projection` is the check.

3. Positions come from `_unit(SEED, "pos", city, meeting, ...)`. A clip's offset is a
   function of the frozen seed and the meeting's identity and of nothing else, so it
   cannot depend on any measurement, in this run or a later one.

4. There is no clean-speech filter and no filter of any kind on content. Clean speech
   decides what the model LEARNS, never what MEASURES it, and a sample filtered to easy
   passages measures performance on the easy passages.

THE ALLOCATION IS FULLY DETERMINED

Every meeting supplies exactly `CLIPS_PER_MEETING = 3`, and `N_CLIPS = 45 = 3 x 15`.
There is no city-quota step: v1 had one because it apportioned 80 clips over uneven city
pools, and there is nothing left for it to decide. Weighting by available audio duration
was considered and REJECTED by measurement, not by taste -- one 4.26 h thira meeting
holds 32.8% of the eligible audio and can supply 3 clips, so duration weights and a
per-meeting cap have no feasible point together above N = 10. The estimand is therefore
"a uniformly random position inside a meeting, with meetings drawn evenly", and NOT "a
uniformly random second of council audio", which is the stronger sentence and is not
what this does.

THE MEETING UNIVERSE

Keyed `(city, meeting_id)`, never the meeting slug alone -- slugs repeat across cities
and a meeting-only key has already produced one wrong analysis in this project. The
universe is the 151 released meetings, less every meeting any council training source has
touched (`clean-pack-arm-2026-08`, `spliced-pack-arm-2026-08`,
`spliced-pack-arm-jitter-2026-08`, keyed `city_id`/`meeting_id`), less everything the
hash-pinned exclusion manifest blocks **at both the meeting level and the city level**.
The city level matters: skipping it leaves the five confirmation cities of
`exp-2026-08-26-category-chooser` (`argos, samothraki, sparta, xylokastro, argithea`) in
the frame and turns 15 meetings into 33. Measured 2026-09-06: 0 touched by training, 75
blocked by meeting, 61 blocked by city, **15 meetings in 7 cities surviving**.

WHAT LANDS WHERE

Clips and the sample manifest live under `~/.cache/oc-public/ordinary-sample-2026-09/`,
one immutable generation per cut with a `current` symlink. The v1 generation stays on
disk untouched: it is not wrong, it answers a narrower question. Only counts reach git.
No transcript text is read or written by this module at all -- it has no text field
anywhere.

    .venv-eval/bin/python -m eval.controlled_eval.cut_ordinary_sample plan
    .venv-eval/bin/python -m eval.controlled_eval.cut_ordinary_sample cut
    .venv-eval/bin/python -m eval.controlled_eval.cut_ordinary_sample verify
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.controlled_eval.exclusion_manifest import (  # noqa: E402
    MANIFEST_PATH as EXCLUSION_MANIFEST_PATH, Manifest, load_manifest)

SC = Path.home() / ".cache" / "oc-public"

# The released-meeting cache. One JSON per meeting, named `<city>__<meeting>.json`.
# It is read for meeting identity and the recording URL, and for nothing else.
MEETINGS_DIR = SC / "meetings"

# Every council training source. A meeting appearing in any of them is not untouched.
TRAINING_MANIFESTS = (
    SC / "clean-pack-arm-2026-08" / "manifest.jsonl",
    SC / "spliced-pack-arm-2026-08" / "manifest.jsonl",
    SC / "spliced-pack-arm-jitter-2026-08" / "manifest.jsonl",
)

# The files this module may open: the three training manifests, the exclusion manifest,
# and the `.json` children of MEETINGS_DIR. Nothing else, and the bug it is written to
# catch is "just peek at the hypotheses to pick better clips".
#
# One honest note about the exclusion manifest. `exclusion_manifest.load_manifest`
# cross-checks the pinned file against five live sources, two of which are benchmark
# RESULT files. It reads `(cityId, meetingId)` out of them and nothing else, and the
# result can only REMOVE a meeting -- it never chooses a position and never orders
# anything. That is meeting identity, not a failure location, so it does not spend this
# sample's independence; recorded here rather than left for someone to discover.
ALLOWED_FILES = TRAINING_MANIFESTS + (EXCLUSION_MANIFEST_PATH,)
ALLOWED_DIRS = (MEETINGS_DIR,)
ALLOWED_SUFFIX = ".json"

OUT_DIR = SC / "ordinary-sample-2026-09"
GENERATIONS = OUT_DIR / "generations"
CURRENT = OUT_DIR / "current"
DURATIONS_CACHE = OUT_DIR / "durations.json"
COUNTS_PATH = ROOT / "eval" / "results_ordinary_sample.json"

# --------------------------------------------------------------------- frozen
SAMPLE_KIND = "ordinary"
DESIGN_VERSION = "v2"
SOURCE = "ordinary-sample-2026-09-v2"
# The seed is part of the design, not a tuning knob. Changing this string produces a
# different sample and therefore a different measurement.
SEED = "ordinary-sample-2026-09/v2"

# The v1 generation, kept on disk. It is not wrong; it answers a narrower question on a
# frame that was failure-selected and about to be trained on.
SUPERSEDED_GENERATION = "9dc8a18fa43cd6bf"

CLIPS_PER_MEETING = 3
N_MEETINGS = 15
N_CLIPS = CLIPS_PER_MEETING * N_MEETINGS      # 45, and the code checks it
CLIP_MS = 20_000

# The sentence that must travel with every number this sample produces.
ESTIMAND = (
    "preference at a uniformly random position inside a meeting, with meetings drawn "
    "evenly (3 clips each) from the released council meetings that no council training "
    "source touched, that the exclusion manifest permits at both the meeting and the "
    "city level, and that share no meeting and no recording with the listening queue "
    "any model was scored or trained on; the frame is independent of every model output, "
    "including where any model failed. NOT a uniformly random second of council audio: "
    "meetings are drawn evenly, not in proportion to their length.")

# The cost of the frame move, stated where a reader of the result cannot miss it.
PRECISION_NOTE = (
    "15 meeting clusters, against v1's 67. At this cluster count the decision rule's "
    "confidence interval is not merely wider -- it is a different order of precision, "
    "and only a large effect will be visible at all.")

# A clip must fit inside the recording with room to spare. `END_MARGIN_MS` exists
# because an mp3's reported duration can be a bitrate estimate: a position chosen inside
# the estimate but past the real end comes back as a short clip. 30 s of margin costs
# 21% of the shortest meeting in this frame (141.5 s) and removes the whole failure
# class; that meeting still has 91.5 s to draw 3 non-overlapping clips from.
END_MARGIN_MS = 30_000
# Clips of one meeting may not overlap. They may still be adjacent.
MIN_SEPARATION_MS = CLIP_MS
MAX_DRAW_ATTEMPTS = 512

# A cut clip is checked for the EXACT sample count, not a duration tolerance: 20 s of
# mono 16 kHz 16-bit PCM is 320,000 frames and nothing else. A tolerance would accept a
# clip that came back a frame short because the seek landed past the end of the
# recording. `wave` reads the header, so no ffprobe parse sits between the file and the
# check (Codex review 0201f130).
EXPECT_FRAMES = CLIP_MS * 16_000 // 1000
EXPECT_WAV = {"nchannels": 1, "sampwidth": 2, "framerate": 16000,
              "nframes": EXPECT_FRAMES}

# A remote read may fail transiently. The SAME planned clip is retried; a position is
# never redrawn and a meeting is never substituted, because either would make the
# published sample a function of which downloads happened to fail.
CUT_ATTEMPTS = 3
RANGE_PREFLIGHT_BYTES = 1024

# `ord__<city>__<meeting>__<start_ms>`. The prefix is not decoration: it is what makes a
# stray wav traceable AND what makes an accidentally merged score set detectable, since
# the challenge sample's ids all start `chl__`.
CLIP_ID_PREFIX = "ord"
_CLIP_ID_RE = re.compile(r"^ord__[a-z0-9-]+__[a-z0-9_]+__\d+$")

# What a meeting file is allowed to contribute. Everything else it carries -- above all
# `transcript`, which is published OpenCouncil text and therefore partly ASR output --
# is dropped at the projection and can never reach a position.
MEETING_FIELDS = ("city", "meeting_id", "meeting_date", "audio_url")


class InputNotAllowed(RuntimeError):
    """A path outside the allow-list was opened. This is the independence guard."""


class IndependenceViolation(RuntimeError):
    """A forbidden field survived the projection. Nothing may continue past this."""


class ExclusionViolation(RuntimeError):
    """An excluded or training-touched meeting reached the sample."""


class FrameError(RuntimeError):
    """The meeting frame is not the frame this design was frozen against."""


class GeometryError(ValueError):
    """A clip span violates a frozen invariant. Never an assert: `python -O`."""


@dataclass(frozen=True, slots=True)
class Meeting:
    """A meeting, and nothing about what was said in it or how any model did.

    `slots` is load bearing. Without it a later edit could attach `transcript` to an
    instance and the independence claim would quietly stop being true.
    """

    city: str
    meeting_id: str
    meeting_date: str | None
    audio_url: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.city, self.meeting_id)


# ------------------------------------------------------------------- input gate

def _open_input(path: Path) -> bytes:
    """Read one of the permitted inputs, or refuse.

    Membership is by resolved path, so a symlink pointing somewhere else is refused too.
    The meetings cache is admitted as a directory because it holds one file per meeting
    and the set grows; only its `.json` children are readable, and only at that one
    level.
    """
    p = Path(path)
    try:
        got = p.resolve()
    except OSError as e:                      # pragma: no cover - filesystem error
        raise InputNotAllowed(f"{p} cannot be resolved: {e}") from e
    if got in {Path(a).resolve() for a in ALLOWED_FILES}:
        return got.read_bytes()
    for d in ALLOWED_DIRS:
        try:
            root = Path(d).resolve()
        except OSError:                       # pragma: no cover
            continue
        if got.parent == root and got.suffix == ALLOWED_SUFFIX:
            return got.read_bytes()
    raise InputNotAllowed(
        f"{p} is not an input this cutter may read. The ordinary sample's whole value is "
        "that its frame and its positions were chosen without consulting any model "
        "output; widening this list spends that.")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------- the meeting universe

def _project(payload: dict) -> Meeting:
    """One meeting file -> a `Meeting`, dropping everything else on the floor.

    `payload` is the whole decoded file, including `transcript`. Only `payload["meeting"]`
    is consulted, and only four of its keys. The projection is checked rather than
    trusted: if `MEETING_FIELDS` ever stops covering `Meeting`'s constructor the mismatch
    is an error, not a silent default.
    """
    m = payload.get("meeting")
    if not isinstance(m, dict):
        raise IndependenceViolation("meeting file carries no `meeting` object")
    kept = {"city": m.get("cityId"), "meeting_id": m.get("id"),
            "meeting_date": m.get("dateTime"), "audio_url": m.get("audioUrl")}
    missing = [k for k in ("city", "meeting_id", "audio_url") if not kept[k]]
    if missing:
        raise IndependenceViolation(
            f"meeting file is missing {missing}; refusing to guess a meeting's identity")
    if set(kept) != set(Meeting.__slots__) or set(kept) != set(MEETING_FIELDS):
        raise IndependenceViolation(
            f"the whitelist {sorted(kept)} does not match Meeting{Meeting.__slots__}")
    return Meeting(**kept)


def meetings_from_payloads(payloads) -> tuple[Meeting, ...]:
    """The pure core: decoded meeting files -> meetings, sorted, deduplicated.

    Everything except the four whitelisted fields is dropped here, so a test can put a
    whole published transcript in every payload and check that the plan does not move.
    Input ORDER is irrelevant too -- meetings come out sorted by key.
    """
    seen: dict[tuple[str, str], Meeting] = {}
    by_url: dict[str, tuple[str, str]] = {}
    for payload in payloads:
        m = _project(payload)
        prev = seen.setdefault(m.key, m)
        if prev != m:
            raise FrameError(
                f"meeting {m.key} appears with two different identities in the cache")
        other = by_url.setdefault(m.audio_url, m.key)
        if other != m.key:
            raise FrameError(
                f"meetings {other} and {m.key} name the same recording. One recording "
                "under two meeting ids would put the same audio in the sample twice and "
                "break the meeting clustering the decision rule depends on.")
    return tuple(seen[k] for k in sorted(seen))


def meeting_universe(meetings_dir: Path = MEETINGS_DIR) -> tuple[Meeting, ...]:
    """Every released meeting in the cache, projected."""
    files = sorted(Path(meetings_dir).glob(f"*{ALLOWED_SUFFIX}"))
    if not files:
        raise FrameError(f"{meetings_dir} holds no meeting files")
    return meetings_from_payloads(
        json.loads(_open_input(f).decode("utf-8")) for f in files)


def touched_meetings() -> dict[tuple[str, str], list[str]]:
    """`(city, meeting_id)` -> the training sources that used it.

    Keyed on the pair for the same reason everything else is: `meeting_id` alone collides
    across cities, and a meeting-only key here would over-exclude real material.
    """
    out: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for path in TRAINING_MANIFESTS:
        name = Path(path).parent.name
        for line in _open_input(path).decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["city_id"], row["meeting_id"])
            if name not in out[key]:
                out[key].append(name)
    return dict(out)


def eligible_meetings(manifest: Manifest,
                      meetings_dir: Path = MEETINGS_DIR) -> tuple[tuple[Meeting, ...], dict]:
    """The frame: released meetings no training source touched and no exclusion blocks.

    `manifest.is_excluded` applies BOTH the meeting keys and the city keys. The city
    level is not optional: skipping it leaves the five confirmation cities of
    `exp-2026-08-26-category-chooser` in the frame, which turns 15 meetings into 33 and
    spends held-out confirmation material nobody released.
    """
    universe = meeting_universe(meetings_dir)
    touched = touched_meetings()
    drops = collections.Counter({"trained_on": 0, "excluded_meeting": 0,
                                 "excluded_city": 0})
    kept = []
    for m in universe:
        if m.key in touched:
            drops["trained_on"] += 1
            continue
        if m.city in manifest.city_keys:
            drops["excluded_city"] += 1
            continue
        if m.key in manifest.meeting_keys:
            drops["excluded_meeting"] += 1
            continue
        kept.append(m)
    return tuple(kept), dict(drops)


def frame_document(meetings, manifest: Manifest) -> str:
    """The frame as one canonical, hashable document.

    The frame is the part of a sample whose provenance has to be quotable, so it is
    written into every generation and hashed into the summary rather than recomputed
    silently on every run (Codex review 0201f130).
    """
    return json.dumps({
        "design_version": DESIGN_VERSION,
        "estimand": ESTIMAND,
        "precision_note": PRECISION_NOTE,
        "exclusion_manifest_sha256": manifest.loaded_sha256,
        "training_manifest_sha256": {
            Path(x).parent.name: sha256_file(x) for x in TRAINING_MANIFESTS},
        "meetings": [
            {"city": m.city, "meeting_id": m.meeting_id,
             "meeting_date": m.meeting_date, "audio_url": m.audio_url}
            for m in sorted(meetings, key=lambda m: m.key)],
    }, ensure_ascii=False, sort_keys=True, indent=2)


# The frame this design was frozen against, so a cache that has since grown a meeting is
# caught here rather than in a listening session six weeks later. Set after the first
# `plan`; an empty string disables the check for a first run on a new machine.
FROZEN_FRAME_SHA256 = "977f0ee015fe6e01044f3e4f3d396b8163aa26f227555679899023f8589f6c10"


def check_frame(frame: str, expect: str = FROZEN_FRAME_SHA256) -> str:
    got = _sha256_bytes(frame.encode("utf-8"))
    if expect and got != expect:
        raise FrameError(
            f"the meeting frame hashes to {got}, not the frozen {expect}. The cache has "
            "gained or lost an eligible meeting, or an exclusion moved. Work out what "
            "changed and record it before cutting anything: a frame that drifts is a "
            "different sample wearing the same name.")
    return got


# ----------------------------------------------------------------- the draw

def _unit(*parts) -> float:
    """A deterministic number in [0, 1) from the frozen seed and the given labels.

    Keyed per meeting rather than drawn from one stream, so adding or removing a meeting
    changes that meeting's clips and leaves every other meeting's positions alone.
    """
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") / 2 ** 64


def allocation(meetings, clips_per_meeting: int = CLIPS_PER_MEETING,
               n_clips: int = N_CLIPS) -> dict[tuple[str, str], int]:
    """Exactly `clips_per_meeting` from every meeting. There is nothing to decide.

    v1 had a largest-remainder city-quota step because it apportioned 80 clips over
    uneven city pools. v2 does not: the allocation is fully determined by the frame, so
    the quota function is gone rather than kept as something that always returns the same
    answer. Weighting by audio duration was rejected by measurement -- one 4.26 h thira
    meeting holds 32.8% of the eligible audio and can supply 3 clips, so duration weights
    and a per-meeting cap have no feasible point together above N = 10.

    A frame that does not multiply out to `n_clips` is an error, not a rounding problem.
    """
    want = clips_per_meeting * len(meetings)
    if want != n_clips:
        raise FrameError(
            f"{len(meetings)} meetings x {clips_per_meeting} clips is {want}, not the "
            f"frozen {n_clips}. The frame moved; nothing is reallocated to cover it, "
            "because a sample of a different size is a different measurement.")
    return {m.key: clips_per_meeting for m in meetings}


def positions(meeting: Meeting, k: int, duration_ms: int) -> tuple[int, ...]:
    """`k` uniformly random, non-overlapping 20 s start offsets inside one meeting.

    Rejection sampling with a bounded, deterministic attempt counter: the alternative --
    drawing into shrinking sub-intervals -- is not uniform over the recording, and this
    sample's only claim is that its positions were uniform and blind. Redrawing all `k`
    on a collision keeps the draw uniform over the non-overlapping region, which drawing
    one replacement point would not.

    The attempt budget is generous because the shortest meeting in the frame is 141.5 s:
    3 non-overlapping clips in its 91.5 s of usable span land about 18% of the time, so
    512 attempts fail with probability around 1e-40.
    """
    if k <= 0:
        raise GeometryError(f"{meeting.key} was allocated {k} clips")
    usable = duration_ms - CLIP_MS - END_MARGIN_MS
    if usable <= 0:
        raise GeometryError(
            f"{meeting.key} is {duration_ms / 1000:.1f}s, too short for a "
            f"{CLIP_MS / 1000:.0f}s clip with a {END_MARGIN_MS / 1000:.0f}s tail margin")
    for attempt in range(MAX_DRAW_ATTEMPTS):
        xs = sorted(int(_unit(SEED, "pos", meeting.city, meeting.meeting_id, i,
                              attempt) * (usable + 1))
                    for i in range(k))
        if all(xs[j + 1] - xs[j] >= MIN_SEPARATION_MS for j in range(k - 1)):
            return tuple(xs)
    raise GeometryError(
        f"{meeting.key}: {MAX_DRAW_ATTEMPTS} draws all produced overlapping clips in "
        f"{duration_ms / 1000:.1f}s of audio")


def clip_id(meeting: Meeting, start_ms: int) -> str:
    cid = f"{CLIP_ID_PREFIX}__{meeting.city}__{meeting.meeting_id}__{start_ms}"
    if not _CLIP_ID_RE.match(cid):
        raise GeometryError(f"clip id is not path-safe: {cid!r}")
    return cid


def sample_row(meeting: Meeting, start_ms: int) -> dict:
    """The row a preference page loads. Provenance and geometry, never text."""
    return {
        "clip_id": clip_id(meeting, start_ms),
        "sample": SAMPLE_KIND,
        "design_version": DESIGN_VERSION,
        "city": meeting.city,
        "meeting_id": meeting.meeting_id,
        "meeting_date": meeting.meeting_date,
        "audio_url": meeting.audio_url,
        "clip_start": start_ms / 1000,
        "clip_end": (start_ms + CLIP_MS) / 1000,
        "clip_dur": CLIP_MS / 1000,
        "source": SOURCE,
        "selection": "uniform-random-position",
        "seed": SEED,
    }


def build_rows(meetings, alloc: dict[tuple[str, str], int],
               durations_ms: dict[tuple[str, str], int]) -> list[dict]:
    by_key = {m.key: m for m in meetings}
    rows: list[dict] = []
    for key in sorted(alloc):
        m = by_key[key]
        for start in positions(m, alloc[key], durations_ms[key]):
            rows.append(sample_row(m, start))
    rows.sort(key=lambda r: (r["city"], r["meeting_id"], r["clip_start"]))
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dup = [i for i, n in collections.Counter(ids).items() if n > 1]
        raise GeometryError(f"clip id collision: {dup[:3]}")
    return rows


def check_sample(rows, manifest: Manifest, touched: dict) -> None:
    """The last guard before publication. Raises, never asserts: `python -O`."""
    per_meeting = collections.Counter((r["city"], r["meeting_id"]) for r in rows)
    for key, n in per_meeting.items():
        if manifest.is_excluded(*key):
            raise ExclusionViolation(f"excluded meeting {key} reached the sample")
        if key in touched:
            raise ExclusionViolation(
                f"{key} is in {touched[key]} and is not an untouched meeting")
        if n != CLIPS_PER_MEETING:
            raise ExclusionViolation(
                f"{key} supplied {n} clips; every meeting supplies exactly "
                f"{CLIPS_PER_MEETING} and an uneven one is a different design")
    # Per-row checks before the count: a mislabelled or malformed row is a more
    # specific fault than "the sample is the wrong size", and reporting the size first
    # would bury it.
    for r in rows:
        if not _CLIP_ID_RE.match(r["clip_id"]):
            raise GeometryError(f"clip id is not path-safe: {r['clip_id']!r}")
        if r["sample"] != SAMPLE_KIND:
            raise ExclusionViolation(
                f"{r['clip_id']} is labelled {r['sample']!r}; the two samples are never "
                "summed and a mislabelled row is how they would be")
        if abs((r["clip_end"] - r["clip_start"]) - CLIP_MS / 1000) > 1e-6:
            raise GeometryError(f"{r['clip_id']} is not {CLIP_MS / 1000:.0f}s long")
    if len(rows) != N_CLIPS:
        raise FrameError(f"{len(rows)} clips, not the frozen {N_CLIPS}")


# ------------------------------------------------------------------ operational

def probe_duration(target: str) -> float | None:
    """Seconds, from the container header. Over HTTP this is a range read, not a
    download: `data.opencouncil.gr` answers `accept-ranges: bytes`."""
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(target)], capture_output=True, text=True, timeout=120)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def load_durations(path: Path = DURATIONS_CACHE) -> dict[tuple[str, str], int]:
    """The cache is OUR OWN output, not an input, so it does not go through the gate.

    It carries meeting ids and whole-recording lengths; nothing about any model.
    """
    if not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {tuple(k.split("__", 1)): int(v) for k, v in d.items()}


def save_durations(durations: dict[tuple[str, str], int],
                   path: Path = DURATIONS_CACHE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {f"{c}__{m}": v for (c, m), v in sorted(durations.items())}, indent=1),
        encoding="utf-8")


def probe_missing(meetings, known: dict, workers: int = 8) -> tuple[dict, list]:
    todo = [m for m in meetings if m.key not in known]
    out = dict(known)
    failed = []
    if not todo:
        return out, failed
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for m, d in zip(todo, ex.map(lambda x: probe_duration(x.audio_url), todo)):
            if d is None or d <= 0:
                failed.append(m.key)
            else:
                out[m.key] = int(round(d * 1000))
    return out, failed


# ffmpeg writes the input URL into its own error text, and the failure summary of a cut
# is written to a file that lives IN GIT. The classification is deliberately coarse; the
# raw text stays in memory where an operator can see it and git cannot.
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


def wav_shape(path: Path) -> dict | None:
    """The wav header, read by stdlib. `None` when the file is not a readable wav."""
    try:
        with wave.open(str(path), "rb") as fh:
            return {"nchannels": fh.getnchannels(), "sampwidth": fh.getsampwidth(),
                    "framerate": fh.getframerate(), "nframes": fh.getnframes()}
    except (wave.Error, OSError, EOFError):
        return None


def range_supported(url: str, nbytes: int = RANGE_PREFLIGHT_BYTES) -> bool:
    """Does the server actually answer a byte range, or only claim to?

    `Accept-Ranges: bytes` is a claim. A 206 with a `Content-Range` is the answer, and
    the difference decides whether cutting 45 clips costs 45 range reads or 45 whole
    recordings (Codex review 0201f130).
    """
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 206 and bool(r.headers.get("Content-Range"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def cut_one(row: dict, dst: Path, attempts: int = CUT_ATTEMPTS) -> dict:
    """One 20 s mono 16 kHz wav, cut from the remote recording by range request.

    `-ss` before `-i` so ffmpeg seeks rather than decoding from zero -- over a multi-hour
    recording that is the difference between a range read and a whole download. The
    streams are named explicitly (`-map 0:a:0 -vn -sn -dn`) so a recording that grows a
    cover-art video stream cannot silently change what gets encoded, and the result is
    checked for EXACTLY 320,000 frames rather than a duration within a tolerance.

    Retries re-cut the SAME planned clip. A position is never redrawn and a meeting is
    never substituted: either would make the published sample a function of which
    downloads happened to fail, which is a selection effect with no record of itself.

    RESIDUAL RISK, not closed here: a correctly sized clip cut from the wrong timestamp
    would pass every check above. Proving the seek landed where it was asked would mean
    decoding a whole multi-hour recording locally for comparison, which is not affordable
    per clip. What is recorded instead is the request and the resulting hash, so a later
    challenge has something to reproduce.
    """
    last = {"method": None, "reason": "not-attempted", "error": ""}
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y",
             "-reconnect", "1", "-reconnect_streamed", "1",
             "-reconnect_delay_max", "5",
             "-ss", f"{row['clip_start']:.3f}", "-i", row["audio_url"],
             "-t", f"{row['clip_dur']:.3f}",
             "-map", "0:a:0", "-vn", "-sn", "-dn",
             "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000", str(dst)],
            capture_output=True, text=True, timeout=900)
        if proc.returncode != 0 or not dst.exists():
            dst.unlink(missing_ok=True)
            last = {"method": None, "error": (proc.stderr or "").strip()[-200:],
                    "reason": classify_failure(proc.stderr), "attempts": attempt}
            continue
        shape = wav_shape(dst)
        if shape != EXPECT_WAV:
            dst.unlink(missing_ok=True)
            last = {"method": None, "reason": "wav-shape-mismatch", "attempts": attempt,
                    "error": f"wav is {shape}, expected {EXPECT_WAV}"}
            continue
        return {"method": "ffmpeg", "measured_dur": shape["nframes"] / shape["framerate"],
                "attempts": attempt}
    return last


def plan(durations: dict | None = None,
         meetings_dir: Path = MEETINGS_DIR) -> dict:
    """Everything `cut` would do, computed and returned, with nothing written."""
    manifest = load_manifest()
    meetings, drops = eligible_meetings(manifest, meetings_dir)
    touched = touched_meetings()
    frame = frame_document(meetings, manifest)
    frame_sha = check_frame(frame)

    known = load_durations() if durations is None else dict(durations)
    durations_ms, failed = probe_missing(meetings, known)
    if failed:
        raise FrameError(
            f"{len(failed)} meetings would not report a duration ({failed[:3]}). Every "
            "meeting in the frame supplies clips, so a frame with an unreadable "
            "recording is not this sample.")
    rows = build_rows(meetings, allocation(meetings), durations_ms)
    check_sample(rows, manifest, touched)

    usable_h = sum(max(0, durations_ms[m.key] - CLIP_MS - END_MARGIN_MS)
                   for m in meetings) / 1000 / 3600
    return {
        "summary": {
            "sample": SAMPLE_KIND,
            "design_version": DESIGN_VERSION,
            "source": SOURCE,
            "seed": SEED,
            "supersedes_generation": SUPERSEDED_GENERATION,
            "estimand": ESTIMAND,
            "precision_note": PRECISION_NOTE,
            "n_meeting_clusters": len(meetings),
            "v1_meeting_clusters": 67,
            "clip_seconds": CLIP_MS / 1000,
            "clips_per_meeting": CLIPS_PER_MEETING,
            "allocation": "fully determined: every meeting supplies the same number",
            "city_weighting": ("none; meetings are drawn evenly. Duration weighting was "
                               "rejected by measurement: one 4.26 h thira meeting holds "
                               "32.8% of the eligible audio and can supply 3 clips, so "
                               "duration weights and a per-meeting cap have no feasible "
                               "point together above N = 10."),
            "frame_sha256": frame_sha[:16],
            "exclusion_manifest_sha256": manifest.loaded_sha256,
            "training_manifest_sha256": {
                Path(p).parent.name: sha256_file(p)[:16] for p in TRAINING_MANIFESTS},
            "n_released_meetings": len(meetings) + sum(drops.values()),
            "n_eligible_meetings": len(meetings),
            "meeting_drops": drops,
            "meetings_by_city": dict(sorted(
                collections.Counter(m.city for m in meetings).items())),
            "n_clips": len(rows),
            "clips_by_city": dict(sorted(
                collections.Counter(r["city"] for r in rows).items())),
            "clip_hours": round(len(rows) * CLIP_MS / 1000 / 3600, 4),
            "eligible_audio_hours": round(usable_h, 2),
        },
        "clips": rows,
        "durations_ms": durations_ms,
        "meeting_frame": frame,
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

    urls = sorted({r["audio_url"] for r in rows})
    no_range = [u for u in urls if not range_supported(u)]
    if no_range:
        raise RuntimeError(
            f"{len(no_range)} of {len(urls)} recordings did not answer a byte range with "
            f"a 206. Cutting {len(rows)} clips would download whole meetings instead; "
            "check the host before continuing.")

    stage = OUT_DIR / "_staging"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "clips").mkdir(parents=True)
    failed = []
    for i, row in enumerate(rows, 1):
        dst = stage / "clips" / f"{row['clip_id']}.wav"
        res = cut_one(row, dst)
        if res.get("method") is None:
            failed.append({"clip_id": row["clip_id"], "error": res.get("error"),
                           "reason": res.get("reason", "ffmpeg-error")})
            continue
        row["cut_method"] = res["method"]
        row["measured_dur"] = res.get("measured_dur")
        row["audio_sha256"] = sha256_file(dst)
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    cut_rows = [r for r in rows if r.get("audio_sha256")]
    summary["n_clips_cut"] = len(cut_rows)
    summary["n_failed"] = len(failed)
    summary["failure_reasons"] = dict(collections.Counter(
        f.get("reason") or "ffmpeg-error" for f in failed))
    for f in failed:
        print(f"  FAILED {f['clip_id']}: {f['reason']}: {f['error']}", flush=True)
    if failed:
        # No partial publication. A sample short of the clips the plan apportioned is a
        # DIFFERENT sample -- its city mix and its cluster count are whatever the network
        # allowed -- and publishing it would put a selection effect in the measurement
        # with no record of itself (Codex review 0201f130).
        shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError(
            f"{len(failed)} of {len(rows)} clips did not cut "
            f"({summary['failure_reasons']}). Nothing was published; fix the cause and "
            f"re-run. The plan is deterministic, so the same {len(rows)} clips come back.")
    rows = cut_rows

    save_durations(out["durations_ms"])
    (stage / "meeting_frame.json").write_text(out["meeting_frame"], encoding="utf-8")
    (stage / "durations_ms.json").write_text(json.dumps(
        {f"{c}__{m}": v for (c, m), v in sorted(out["durations_ms"].items())}, indent=1),
        encoding="utf-8")
    (stage / "sample.json").write_text(
        json.dumps({"summary": summary, "clips": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    gen_sha = sha256_file(stage / "sample.json")
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
    """Re-check a published sample: frame, exclusions, hashes, geometry."""
    manifest = load_manifest()
    touched = touched_meetings()
    d = json.loads((Path(current) / "sample.json").read_text(encoding="utf-8"))
    rows = d["clips"]
    bad = {"excluded": [], "trained_on": [], "missing": [], "hash": [], "duration": [],
           "format": [], "clip_id": [], "wrong_sample": [], "wrong_count": []}
    per_meeting = collections.Counter((r["city"], r["meeting_id"]) for r in rows)
    for key, n in per_meeting.items():
        if manifest.is_excluded(*key):
            bad["excluded"].append(f"{key[0]}/{key[1]}")
        if key in touched:
            bad["trained_on"].append(f"{key[0]}/{key[1]}")
        if n != CLIPS_PER_MEETING:
            bad["wrong_count"].append(f"{key[0]}/{key[1]}")
    for r in rows:
        if not _CLIP_ID_RE.match(r["clip_id"]):
            bad["clip_id"].append(r["clip_id"])
        if r.get("sample") != SAMPLE_KIND:
            bad["wrong_sample"].append(r["clip_id"])
        p = Path(current) / "clips" / f"{r['clip_id']}.wav"
        if not p.is_file():
            bad["missing"].append(r["clip_id"])
            continue
        if r.get("audio_sha256") and sha256_file(p) != r["audio_sha256"]:
            bad["hash"].append(r["clip_id"])
        shape = wav_shape(p)
        if shape is None or shape["nframes"] != EXPECT_FRAMES:
            bad["duration"].append(r["clip_id"])
        if shape is not None and any(shape.get(k) != v for k, v in EXPECT_WAV.items()):
            bad["format"].append(r["clip_id"])
    return {"n": len(rows),
            "n_meeting_clusters": len(per_meeting),
            "design_version": d["summary"].get("design_version"),
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
