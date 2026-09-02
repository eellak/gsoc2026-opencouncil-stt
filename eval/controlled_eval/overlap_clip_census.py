"""Judge every in-domain training clip against the frozen overlap-free criteria.

Frozen criteria: docs/specs/2026-08-19-overlap-clean-selection.md. Diarization is the
non-exclusive precision-2 output of the FULL meeting recording; labels are local to one
job, so nothing is ever compared or grouped across audio files.

    .venv-eval/bin/python -m eval.controlled_eval.overlap_clip_census --meetings 4
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Frozen in the spec. Not tuned against any observed survival rate.
GUARD_SEC = 0.3
MIN_COVERAGE = 0.90
TOL_COMPETING_INSIDE = 0.05
TOL_OVERLAP_GUARD = 0.05
TOL_COMPETING_GUARD = 0.20
# Ordered: the first flag an utterance carries is the one the waterfall charges it to.
FLAG_ORDER = ["no_primary", "low_coverage", "competing_inside", "overlap_guarded",
              "competing_guard"]

DEFAULT_DIAR = Path.home() / ".cache/oc-public/training-diarization-2026-08"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results_overlap_clip_census.json"


def _clip(lo: float, hi: float, start: float, end: float) -> float:
    return max(0.0, min(hi, end) - max(lo, start))


def active_seconds(turns, start: float, end: float) -> dict[str, float]:
    """Seconds each label is active in [start, end), same-label overlap counted once."""
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for label, turn_start, turn_end in turns:
        lo, hi = max(turn_start, start), min(turn_end, end)
        if hi > lo:
            spans[label].append((lo, hi))
    out = {}
    for label, items in spans.items():
        total, current_end = 0.0, None
        current_start = None
        for lo, hi in sorted(items):
            if current_end is None or lo > current_end:
                if current_end is not None:
                    total += current_end - current_start
                current_start, current_end = lo, hi
            else:
                current_end = max(current_end, hi)
        if current_end is not None:
            total += current_end - current_start
        out[label] = round(total, 6)
    return out


def overlap_seconds(turns, start: float, end: float) -> float:
    """Seconds where two or more DISTINCT labels are active at once in [start, end)."""
    edges = sorted({t for _, s, e in turns for t in (max(s, start), min(e, end))}
                   | {start, end})
    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        if hi <= lo or lo < start or hi > end:
            continue
        mid = (lo + hi) / 2
        labels = {label for label, s, e in turns if s <= mid < e}
        if len(labels) > 1:
            total += hi - lo
    return round(total, 6)


def judge(turns, start: float, end: float, audio_duration: float) -> dict:
    """Apply the frozen predicate to one utterance. Flags are independent, not ordered."""
    duration = end - start
    guard_lo = max(0.0, start - GUARD_SEC)
    guard_hi = min(audio_duration, end + GUARD_SEC)
    inside = active_seconds(turns, start, end)
    guard = active_seconds(turns, guard_lo, guard_hi)

    flags: list[str] = []
    primary, coverage = None, 0.0
    if inside:
        ranked = sorted(inside.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            flags.append("no_primary")
        else:
            primary, best = ranked[0]
            coverage = best / duration if duration > 0 else 0.0
    else:
        flags.append("no_primary")

    competing_inside = sum(v for k, v in inside.items() if k != primary) if primary else 0.0
    competing_guard = sum(v for k, v in guard.items() if k != primary) if primary else 0.0
    overlap_guard = overlap_seconds(turns, guard_lo, guard_hi)

    if primary is not None and coverage < MIN_COVERAGE:
        flags.append("low_coverage")
    if competing_inside > TOL_COMPETING_INSIDE:
        flags.append("competing_inside")
    if overlap_guard > TOL_OVERLAP_GUARD:
        flags.append("overlap_guarded")
    if competing_guard > TOL_COMPETING_GUARD:
        flags.append("competing_guard")

    clearance = None
    if primary is not None:
        edges = [t for label, s, e in turns if label == primary for t in (s, e)]
        if edges:
            clearance = round(min(min(abs(start - t) for t in edges),
                                  min(abs(end - t) for t in edges)), 3)
    return {
        "accepted": not flags,
        "flags": [flag for flag in FLAG_ORDER if flag in flags],
        "primary": primary,
        "coverage": round(coverage, 4),
        "competing_inside_sec": round(competing_inside, 3),
        "competing_guard_sec": round(competing_guard, 3),
        "overlap_guard_sec": round(overlap_guard, 3),
        "boundary_clearance_sec": clearance,   # measured, never an acceptance criterion
        "guard": [round(guard_lo, 3), round(guard_hi, 3)],
        "duration_sec": round(duration, 3),
    }


def waterfall(rows) -> dict[str, int]:
    """Mutually exclusive rejection reasons: each rejected row charged to its first flag."""
    counts = {flag: 0 for flag in FLAG_ORDER}
    for row in rows:
        if row["accepted"]:
            continue
        first = next((flag for flag in FLAG_ORDER if flag in row["flags"]), "unflagged")
        counts[first] = counts.get(first, 0) + 1
    return {k: v for k, v in counts.items() if v}


def load_jobs(root: Path) -> dict[str, dict]:
    """audio_url -> {turns, audio_end, city_id, meeting_id}, one entry per diarized file."""
    jobs = {}
    for path in sorted((root / "jobs").glob("*.json")):
        payload = json.loads(path.read_text())
        turns = [(t["speaker"], float(t["start"]), float(t["end"]))
                 for t in payload["output"]["diarization"]]
        jobs[payload["audio_url"]] = {
            "turns": turns,
            "audio_end": max((end for _, _, end in turns), default=0.0),
            "city_id": payload["city_id"],
            "meeting_id": payload["meeting_id"],
            "audio_sha256": payload["audio_sha256"],
        }
    return jobs


def transcript_rows(transcripts: Path, jobs: dict[str, dict]) -> list[dict]:
    """Every utterance of every fetched meeting whose audio file has a diarization job.

    The parquet holds a ~10% sample per meeting, so packs cannot be built from it; the
    full transcript is the supply. Speaker identity comes from the segment's personId,
    which is OpenCouncil metadata, not from the diarization labels.
    """
    by_meeting = {(job["city_id"], job["meeting_id"]): url for url, job in jobs.items()}
    rows = []
    for city, meeting in sorted(by_meeting):
        path = transcripts / f"{city}__{meeting}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        for segment in payload["transcript"]:
            person = (segment.get("speakerTag") or {}).get("personId")
            for utterance in segment.get("utterances", []):
                rows.append({
                    "utterance_id": utterance["id"],
                    "city_id": city,
                    "meeting_id": meeting,
                    "audio_url": by_meeting[(city, meeting)],
                    "source": "transcript",
                    "person_id": person,
                    "segment_id": segment["id"],
                    "start": float(utterance["startTimestamp"]),
                    "end": float(utterance["endTimestamp"]),
                    "text_chars": len(utterance.get("text") or ""),
                })
    return rows


def census(frame, jobs: dict[str, dict]) -> tuple[list[dict], dict]:
    """Judge every utterance whose audio file has a diarization job."""
    rows, skipped = [], {"no_job": 0, "outside_recording": 0}
    records = frame if isinstance(frame, list) else frame.to_dict("records")
    for record in records:
        job = jobs.get(record["audio_url"])
        if job is None:
            skipped["no_job"] += 1
            continue
        start, end = float(record["start"]), float(record["end"])
        if end <= start or start < 0 or start > job["audio_end"]:
            skipped["outside_recording"] += 1
            continue
        verdict = judge(job["turns"], start, end, job["audio_end"])
        rows.append({**{k: v for k, v in record.items() if k not in ("start", "end")},
                     "start": start, "end": end, **verdict})
    return rows, skipped


def summarize(rows: list[dict], label: str) -> dict:
    hours = sum(row["duration_sec"] for row in rows) / 3600
    accepted = [row for row in rows if row["accepted"]]
    accepted_hours = sum(row["duration_sec"] for row in accepted) / 3600
    return {
        "label": label,
        "rows": len(rows),
        "accepted_rows": len(accepted),
        "accepted_row_fraction": round(len(accepted) / len(rows), 4) if rows else None,
        "speech_hours": round(hours, 3),
        "accepted_speech_hours": round(accepted_hours, 3),
        "accepted_hour_fraction": round(accepted_hours / hours, 4) if hours else None,
    }


def runs(rows: list[dict], max_gap_sec: float = 2.0) -> list[int]:
    """Lengths of packable same-speaker runs, broken by the frozen rules in the spec."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["audio_url"]].append(row)
    lengths = []
    for file_rows in by_file.values():
        file_rows.sort(key=lambda r: r["start"])
        current = 0
        previous = None
        for row in file_rows:
            keep = (row["accepted"] and previous is not None
                    and row["primary"] == previous["primary"]
                    and row["start"] >= previous["end"]
                    and row["start"] - previous["end"] <= max_gap_sec)
            if not row["accepted"]:
                if current:
                    lengths.append(current)
                current, previous = 0, None
                continue
            current = current + 1 if keep else (lengths.append(current) if current else None) or 1
            previous = row
        if current:
            lengths.append(current)
    return lengths


def build_packs(rows: list[dict], target_speech_sec: float = 22.0,
                separator_sec: float = 0.4, max_waveform_sec: float = 29.0,
                max_gap_sec: float = 2.0) -> list[list[dict]]:
    """Greedy deterministic packer over accepted rows.

    A pack never crosses an audio file, a diarization job, or a speaker: the frozen
    break rules are in the spec. Rows stay chronological and are never reused.
    """
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["accepted"]:
            by_file[row["audio_url"]].append(row)
    packs: list[list[dict]] = []
    for _, file_rows in sorted(by_file.items()):
        file_rows.sort(key=lambda r: r["start"])
        current: list[dict] = []
        speech = waveform = 0.0
        previous = None
        for row in file_rows:
            breaks = (previous is None or row["primary"] != previous["primary"]
                      or row["start"] < previous["end"]
                      or row["start"] - previous["end"] > max_gap_sec)
            extra = row["duration_sec"] + (0.0 if breaks or not current else separator_sec)
            if breaks or waveform + extra > max_waveform_sec:
                if current:
                    packs.append(current)
                current, speech, waveform = [], 0.0, 0.0
                extra = row["duration_sec"]
            current.append(row)
            speech += row["duration_sec"]
            waveform += extra
            previous = row
            if speech >= target_speech_sec:
                packs.append(current)
                current, speech, waveform, previous = [], 0.0, 0.0, None
        if current:
            packs.append(current)
    return packs


def pack_speech_sec(pack: list[dict]) -> float:
    return round(sum(row["duration_sec"] for row in pack), 3)


def packable_hours(rows: list[dict], target_speech_sec: float = 22.0,
                   separator_sec: float = 0.4, max_waveform_sec: float = 29.0,
                   max_gap_sec: float = 2.0) -> dict:
    """Yield of the deterministic packer: only packs that reach 80% of target count."""
    packs = build_packs(rows, target_speech_sec, separator_sec, max_waveform_sec, max_gap_sec)
    kept = [p for p in packs if pack_speech_sec(p) >= target_speech_sec * 0.8]
    dropped = [p for p in packs if pack_speech_sec(p) < target_speech_sec * 0.8]
    packed = sum(pack_speech_sec(p) for p in kept)
    return {
        "packs": len(kept),
        "packed_speech_hours": round(packed / 3600, 3),
        "dropped_tail_hours": round(sum(pack_speech_sec(p) for p in dropped) / 3600, 3),
        "mean_pack_speech_sec": round(packed / len(kept), 2) if kept else None,
        "mean_utterances_per_pack": round(sum(len(p) for p in kept) / len(kept), 2) if kept else None,
        "target_speech_sec": target_speech_sec,
        "max_waveform_sec": max_waveform_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path,
                        default=Path.home() / ".cache/oc-public/training-transcripts-2026-08")
    parser.add_argument("--root", type=Path, default=DEFAULT_DIAR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    jobs = load_jobs(args.root)
    rows, skipped = census(transcript_rows(args.transcripts, jobs), jobs)

    independent = {flag: sum(flag in row["flags"] for row in rows) for flag in FLAG_ORDER}
    by_city = defaultdict(list)
    by_source = defaultdict(list)
    by_file = defaultdict(list)
    for row in rows:
        by_city[row["city_id"]].append(row)
        by_source[row["source"]].append(row)
        by_file[row["audio_url"]].append(row)
    clearances = sorted(row["boundary_clearance_sec"] for row in rows
                        if row["boundary_clearance_sec"] is not None)
    lengths = runs(rows)
    result = {
        "experiment": "overlap-clean-selection-2026-08-19",
        "spec": "docs/specs/2026-08-19-overlap-clean-selection.md",
        "stage": "pilot" if len(by_file) < 8 else "stratified",
        "audio_files_diarized": len(jobs),
        "overall": summarize(rows, "all diarized files"),
        "skipped": skipped,
        "independent_flags": independent,
        "rejection_waterfall": waterfall(rows),
        "by_city": [summarize(v, k) for k, v in sorted(by_city.items())],
        "by_source": [summarize(v, k) for k, v in sorted(by_source.items())],
        "per_file_accepted_fraction": sorted(
            round(summarize(v, k)["accepted_row_fraction"] or 0.0, 4) for k, v in by_file.items()),
        "boundary_clearance_sec": {
            "p10": clearances[len(clearances) // 10] if clearances else None,
            "median": clearances[len(clearances) // 2] if clearances else None,
        },
        "same_speaker_runs": {
            "n_runs": len(lengths),
            "mean_len": round(sum(lengths) / len(lengths), 2) if lengths else None,
            "max_len": max(lengths) if lengths else None,
        },
        "packer": packable_hours(rows),
        "guards": [
            "A pilot over few meetings is an engineering smoke test; its surviving fraction does not estimate the corpus.",
            "Speaker labels are local to one diarization job and are never grouped across audio files.",
            "Diarization is not ground truth; an outcome-blind human audit is required before GPU spend.",
            "boundary_clearance is measured, not an acceptance criterion: diarization cannot prove a cut misses a word.",
        ],
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"files": len(jobs), **result["overall"],
                      "waterfall": result["rejection_waterfall"],
                      "packer": result["packer"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


def pad_bounds(turns, start: float, end: float, audio_end: float,
               max_pad: float = 0.25) -> tuple[float, float]:
    """Grow the cut outwards into silence only, never into anyone else's speech.

    The first human check of spliced packs came back 10/12 unsure on boundaries: cutting
    exactly on a database timestamp clips syllables. Diarization cannot place a word edge,
    but it can say whether the next 0.25 s belongs to nobody, and taking that space back
    is strictly safer than cutting tight.
    """
    lo = max(0.0, start - max_pad)
    hi = min(audio_end, end + max_pad)
    for label, turn_start, turn_end in turns:
        if turn_end <= start and turn_end > lo:
            lo = max(lo, turn_end)                 # somebody else stops here
        if turn_start >= end and turn_start < hi:
            hi = min(hi, turn_start)
    return round(lo, 3), round(hi, 3)


def contiguous_packs(rows: list[dict], turns_by_file: dict[str, list],
                     audio_end_by_file: dict[str, float],
                     min_speech_sec: float = 20.0, max_speech_sec: float = 25.0,
                     max_span_sec: float = 29.0, max_gap_sec: float = 2.0) -> list[dict]:
    """Group consecutive same-speaker utterances and cut ONE continuous span each.

    A spliced pack has one cut per utterance; a continuous span has two, and it keeps the
    real pauses, which is what an inference window contains anyway. The whole span must
    pass the frozen predicate — the gaps between utterances are inside the audio now, so
    they have to be clean too.
    """
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["audio_url"]].append(row)

    packs = []
    for url, file_rows in sorted(by_file.items()):
        turns = turns_by_file[url]
        audio_end = audio_end_by_file[url]
        file_rows.sort(key=lambda r: r["start"])
        group: list[dict] = []
        for row in file_rows + [None]:
            extends = (row is not None and group
                       and row["person_id"] == group[0]["person_id"]
                       and row["person_id"] is not None
                       and row["start"] >= group[-1]["end"]
                       and row["start"] - group[-1]["end"] <= max_gap_sec
                       and row["end"] - group[0]["start"] <= max_span_sec)
            if extends:
                group.append(row)
                if sum(r["duration_sec"] for r in group) >= max_speech_sec:
                    extends = False
                else:
                    continue
            if group:
                speech = sum(r["duration_sec"] for r in group)
                start, end = group[0]["start"], group[-1]["end"]
                if min_speech_sec <= speech and end - start <= max_span_sec:
                    lo, hi = pad_bounds(turns, start, end, audio_end)
                    verdict = judge(turns, lo, hi, audio_end)
                    if verdict["accepted"]:
                        packs.append({
                            "audio_url": url,
                            "city_id": group[0]["city_id"],
                            "meeting_id": group[0]["meeting_id"],
                            "person_id": group[0]["person_id"],
                            "start": lo,
                            "end": hi,
                            "span_sec": round(hi - lo, 3),
                            "speech_sec": round(speech, 3),
                            "utterance_ids": [r["utterance_id"] for r in group],
                            "n_utterances": len(group),
                            "pad_start_sec": round(start - lo, 3),
                            "pad_end_sec": round(hi - end, 3),
                        })
            group = [row] if row is not None else []
    return packs
