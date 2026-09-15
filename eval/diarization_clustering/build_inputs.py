#!/usr/bin/env python3
"""Build the frozen inputs for the diarization-clustering experiment.

Two artifacts per meeting, both derived from the human-reviewed transcript that
opencouncil.gr serves and that this repo already caches under
``~/.cache/oc-public/meetings``:

``transcript.json``
    The Gladia-shaped ``Transcript`` the eval harness consumes, built from the
    human utterance boundaries. Using human boundaries is deliberate: it removes
    ASR segmentation noise so the measurement is about *who* a segment is given
    to, not about where segments start. It also means no number here transfers
    to production, which runs on Scribe boundaries.

``slices.json``
    Topic slices. Contiguous runs of utterances sharing ``discussionSubjectId``
    are merged forward until a slice reaches ``--min-slice`` seconds, because raw
    topic runs have a median of 19-46 s and diarizing those is meaningless.
    A meeting whose agenda yields a single slice is recorded as such: for that
    meeting the sliced arm is identical to the whole-meeting arm and must be
    reported as not testing anything.

Nothing here calls a paid API. Transcript text stays out of git by construction:
outputs are written under ``~/.cache/oc-public``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CACHE = Path(os.environ.get("OC_CACHE", Path.home() / ".cache/oc-public"))
MEETINGS = CACHE / "meetings"
DEFAULT_OUT = CACHE / "diarization-clustering-2026-09"

# Frozen before any number was looked at. See README.md, "Frozen parameters".
MIN_SLICE_S = 300.0


def load_meeting(city: str, meeting: str) -> dict:
    p = MEETINGS / f"{city}__{meeting}.json"
    if not p.exists():
        raise SystemExit(f"no cached meeting at {p} — fetch it first")
    return json.loads(p.read_text())


def human_utterances(meeting: dict) -> list[dict]:
    """Every reviewed utterance, in time order, carrying its speaker tag."""
    out = []
    for seg in meeting.get("transcript") or []:
        tag = seg.get("speakerTagId")
        for u in seg.get("utterances") or []:
            out.append(
                {
                    "start": float(u["startTimestamp"]),
                    "end": float(u["endTimestamp"]),
                    "text": u.get("text") or "",
                    "tag": tag,
                    "subject": u.get("discussionSubjectId"),
                }
            )
    out.sort(key=lambda u: (u["start"], u["end"]))
    return out


def build_transcript(utts: list[dict]) -> dict:
    """Gladia-shaped Transcript, one Word per utterance.

    ``speaker`` is set to 0 for every utterance on purpose: the harness assigns
    speakers from the diarization, and a non-zero value here would look like a
    prior it does not have.
    """
    duration = max((u["end"] for u in utts), default=0.0)
    return {
        "metadata": {
            "audio_duration": duration,
            "number_of_distinct_channels": 1,
            "billing_time": duration,
            "transcription_time": 0,
        },
        "transcription": {
            "languages": ["el"],
            "full_transcript": " ".join(u["text"] for u in utts),
            "utterances": [
                {
                    "text": u["text"],
                    "language": "el",
                    "start": u["start"],
                    "end": u["end"],
                    "confidence": 1.0,
                    "channel": 0,
                    "speaker": 0,
                    "drift": 0,
                    "words": [
                        {
                            "word": u["text"],
                            "start": u["start"],
                            "end": u["end"],
                            "confidence": 1.0,
                        }
                    ],
                }
                for u in utts
            ],
        },
    }


def topic_runs(utts: list[dict]) -> list[dict]:
    """Contiguous runs of utterances sharing one subject id (None included)."""
    runs: list[dict] = []
    cur: dict | None = None
    for u in utts:
        if cur is not None and cur["subject"] == u["subject"]:
            cur["end"] = max(cur["end"], u["end"])
            cur["utterances"] += 1
        else:
            if cur is not None:
                runs.append(cur)
            cur = {
                "subject": u["subject"],
                "start": u["start"],
                "end": u["end"],
                "utterances": 1,
            }
    if cur is not None:
        runs.append(cur)
    return runs


def merge_runs(runs: list[dict], min_slice: float = MIN_SLICE_S) -> list[dict]:
    """Merge forward until each slice reaches ``min_slice`` seconds.

    A trailing remainder shorter than ``min_slice`` is absorbed into the previous
    slice rather than shipped as a runt — a 40 s tail would diarize to noise and
    would still cost a job.
    """
    slices: list[dict] = []
    cur: dict | None = None
    for r in runs:
        if cur is None:
            cur = {"start": r["start"], "end": r["end"], "runs": 1,
                   "subjects": [r["subject"]]}
        else:
            cur["end"] = r["end"]
            cur["runs"] += 1
            cur["subjects"].append(r["subject"])
        if cur["end"] - cur["start"] >= min_slice:
            slices.append(cur)
            cur = None
    if cur is not None:
        if slices:
            slices[-1]["end"] = cur["end"]
            slices[-1]["runs"] += cur["runs"]
            slices[-1]["subjects"].extend(cur["subjects"])
        else:
            slices.append(cur)
    for i, s in enumerate(slices):
        s["index"] = i
        s["duration"] = s["end"] - s["start"]
        s["subjects"] = [x for x in s["subjects"] if x is not None]
    return slices


def named_speaker_count(meeting: dict) -> int:
    """Distinct reviewed people, used as the count hint for arms A1/A2.

    This is reference-derived, not an oracle acoustic count: an attendee who
    never speaks inflates it, an unnamed interjector deflates it. Arms built on
    it are reported as privileged-information sensitivity, never as deployable.
    """
    tags = meeting.get("speakerTags") or []
    people = {t["personId"] for t in tags if t.get("personId")}
    return len(people) if people else len(tags)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(city: str, meeting_id: str, out_dir: Path, min_slice: float) -> dict:
    meeting = load_meeting(city, meeting_id)
    utts = human_utterances(meeting)
    if not utts:
        raise SystemExit(f"{city}/{meeting_id}: no reviewed utterances")

    d = out_dir / f"{city}__{meeting_id}"
    d.mkdir(parents=True, exist_ok=True)

    transcript = build_transcript(utts)
    (d / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False))

    runs = topic_runs(utts)
    slices = merge_runs(runs, min_slice)
    (d / "slices.json").write_text(json.dumps(slices, ensure_ascii=False, indent=2))

    audio = CACHE / "mp3" / f"{city}__{meeting_id}.mp3"
    record = {
        "city": city,
        "meeting": meeting_id,
        "utterances": len(utts),
        "duration_s": transcript["metadata"]["audio_duration"],
        "named_speakers": named_speaker_count(meeting),
        "human_review": (meeting.get("taskStatus") or {}).get("humanReview"),
        "topic_runs": len(runs),
        "slices": len(slices),
        "slice_durations_s": [round(s["duration"], 1) for s in slices],
        "min_slice_s": min_slice,
        "slicing_is_null": len(slices) <= 1,
        "audio_sha256": sha256_file(audio) if audio.exists() else None,
        "audio_path": str(audio) if audio.exists() else None,
    }
    (d / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


MEETINGS_DEFAULT = [
    "chania/jul27_2026",
    "chalandri/may25_2026",
    "athens/may18_2026",
    "athens/may18_2_2026",
    "athens/may28_2026",
    "vrilissia/may27_2_2026",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=MEETINGS_DEFAULT,
                    help="city/meeting_id pairs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-slice", type=float, default=MIN_SLICE_S)
    a = ap.parse_args()

    records = []
    for m in a.meetings:
        city, mid = m.split("/", 1)
        rec = build(city, mid, a.out, a.min_slice)
        records.append(rec)
        null = "  SLICING IS NULL (1 slice)" if rec["slicing_is_null"] else ""
        print(f'{m:26} utt={rec["utterances"]:5} dur={rec["duration_s"]/60:6.1f}min '
              f'named={rec["named_speakers"]:3} runs={rec["topic_runs"]:3} '
              f'slices={rec["slices"]:3}{null}')

    (a.out / "cohort.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    total_h = sum(r["duration_s"] for r in records) / 3600
    sliced = sum(1 for r in records if not r["slicing_is_null"])
    print(f"\ncohort: {len(records)} meetings, {total_h:.2f} audio-hours")
    print(f"meetings where the sliced arm actually differs: {sliced}/{len(records)}")
    print(f"budget: A0+A1+A2+A3 = {total_h*4:.1f} audio-hours (A0-local is free)")
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
