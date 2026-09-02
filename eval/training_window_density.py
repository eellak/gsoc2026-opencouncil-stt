#!/usr/bin/env python3
"""Aggregate-only audit of training versus inference window occupancy.

This module deliberately never serializes row ids, transcript text, audio or
per-window results. Definitions are frozen in
docs/specs/2026-08-18-training-window-density-audit.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import wave
from collections import Counter
from pathlib import Path
from typing import Iterable

WINDOW_SECONDS = 30.0
EXTRA_PAD_SECONDS = 0.2
DURATION_EDGES = (0.0, 1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
POSITION_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
EXPECTED_TRAIN_ROWS = 28_967
EXPECTED_RAW_HOURS = 22.47254726944446

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = ROOT / "data/hf-dataset/public/train.parquet"
DEFAULT_DECODE = Path.home() / ".cache/oc-public/conf-substrate-2026-08/decode-rw.json"
DEFAULT_AUDIO = Path.home() / ".cache/oc-public/bench_windows"
DEFAULT_OUT = ROOT / "eval/results_training_window_density.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile(values: list[float], q: float) -> float:
    """Linear quantile, matching the common type-7 definition."""
    if not values:
        raise ValueError("quantile of empty input")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summary(values: Iterable[float]) -> dict:
    xs = [float(x) for x in values]
    if not xs:
        raise ValueError("summary of empty input")
    return {
        "mean": round(statistics.fmean(xs), 6),
        "p10": round(quantile(xs, 0.10), 6),
        "p25": round(quantile(xs, 0.25), 6),
        "median": round(quantile(xs, 0.50), 6),
        "p75": round(quantile(xs, 0.75), 6),
        "p90": round(quantile(xs, 0.90), 6),
        "p95": round(quantile(xs, 0.95), 6),
        "max": round(max(xs), 6),
    }


def histogram(values: Iterable[float], edges=DURATION_EDGES) -> list[dict]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        x = float(value)
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            if lo <= x < hi or (i == len(counts) - 1 and x == hi):
                counts[i] += 1
                break
        else:
            raise ValueError(f"value {x} outside frozen histogram range")
    total = sum(counts) or 1
    return [
        {"lo_s": lo, "hi_s": hi, "count": count,
         "share": round(count / total, 6)}
        for lo, hi, count in zip(edges, edges[1:], counts)
    ]


def interval_bin_overlaps(start: float, end: float,
                          edges=POSITION_EDGES) -> list[float]:
    start = max(float(start), edges[0])
    end = min(float(end), edges[-1])
    return [max(0.0, min(end, hi) - max(start, lo))
            for lo, hi in zip(edges, edges[1:])]


def aggregate_bin_overlaps(intervals: Iterable[tuple[float, float]]) -> list[dict]:
    intervals = list(intervals)
    totals = [0.0] * (len(POSITION_EDGES) - 1)
    nonzero = [0] * len(totals)
    for start, end in intervals:
        overlap = interval_bin_overlaps(start, end)
        for i, seconds in enumerate(overlap):
            totals[i] += seconds
            nonzero[i] += seconds > 0
    n = len(intervals) or 1
    all_seconds = sum(totals) or 1.0
    return [
        {
            "lo_s": lo,
            "hi_s": hi,
            "mean_seconds_per_window": round(totals[i] / n, 6),
            "windows_with_any_share": round(nonzero[i] / n, 6),
            "share_of_all_interval_seconds": round(totals[i] / all_seconds, 6),
        }
        for i, (lo, hi) in enumerate(zip(POSITION_EDGES, POSITION_EDGES[1:]))
    ]


def _good(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def training_aggregate(rows: list[dict]) -> dict:
    raw_seconds, aligned_seconds, clip_seconds = [], [], []
    raw_intervals, support_intervals = [], []
    sources: Counter = Counter()
    source_raw_seconds: Counter = Counter()
    for row in rows:
        raw_start, raw_end = float(row["start"]), float(row["end"])
        if _good(row.get("start_adj")) and _good(row.get("end_adj")):
            aligned_start = float(row["start_adj"])
            aligned_end = float(row["end_adj"])
        else:
            aligned_start, aligned_end = raw_start, raw_end
        raw = max(0.0, raw_end - raw_start)
        aligned = max(0.0, aligned_end - aligned_start)
        clip_start = max(0.0, aligned_start - EXTRA_PAD_SECONDS)
        intended = min(WINDOW_SECONDS,
                       max(0.0, aligned_end + EXTRA_PAD_SECONDS - clip_start))
        raw_rel_start = max(0.0, raw_start - clip_start)
        raw_rel_end = min(WINDOW_SECONDS, max(raw_rel_start, raw_end - clip_start))

        raw_seconds.append(raw)
        aligned_seconds.append(aligned)
        clip_seconds.append(intended)
        raw_intervals.append((raw_rel_start, raw_rel_end))
        support_intervals.append((0.0, intended))
        source = str(row.get("source", "unknown"))
        sources[source] += 1
        source_raw_seconds[source] += raw

    occupancy = [x / WINDOW_SECONDS for x in clip_seconds]
    raw_occupancy = [x / WINDOW_SECONDS for x in raw_seconds]
    raw_onsets = [a for a, _ in raw_intervals]
    raw_offsets = [b for _, b in raw_intervals]
    return {
        "n_rows": len(rows),
        "raw_speech_span_proxy": {
            "hours": round(sum(raw_seconds) / 3600, 6),
            "seconds": summary(raw_seconds),
            "occupancy": summary(raw_occupancy),
            "histogram_seconds": histogram(raw_seconds),
            "position_by_5s": aggregate_bin_overlaps(raw_intervals),
            "onset_seconds": summary(raw_onsets),
            "offset_seconds": summary(raw_offsets),
        },
        "aligned_span": {
            "hours": round(sum(aligned_seconds) / 3600, 6),
            "seconds": summary(aligned_seconds),
            "histogram_seconds": histogram(aligned_seconds),
        },
        "intended_clip_audio_before_source_end_clamp": {
            "hours": round(sum(clip_seconds) / 3600, 6),
            "seconds": summary(clip_seconds),
            "occupancy": summary(occupancy),
            "digital_padding_share": summary([1.0 - x for x in occupancy]),
            "histogram_seconds": histogram(clip_seconds),
            "position_by_5s": aggregate_bin_overlaps(support_intervals),
        },
        "by_source": {
            source: {"rows": sources[source],
                     "raw_span_hours": round(source_raw_seconds[source] / 3600, 6)}
            for source in sorted(sources)
        },
    }


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def wav_header_fingerprint(audio_dir: Path, window_ids: Iterable[str]) -> dict:
    """Hash only filenames and WAV geometry, never audio bytes."""
    h = hashlib.sha256()
    paths = [audio_dir / f"{window_id}.wav" for window_id in sorted(window_ids)]
    paths = [path for path in paths if path.exists()]
    for path in paths:
        with wave.open(str(path), "rb") as f:
            record = (f"{path.name}\0{f.getnframes()}\0{f.getframerate()}\0"
                      f"{f.getnchannels()}\0{f.getsampwidth()}\n")
        h.update(record.encode())
    return {"n_wav_headers": len(paths), "sha256": h.hexdigest()}


def inference_aggregate(decode: dict, audio_dir: Path) -> dict:
    # Importing reuses the already-frozen faster-whisper seek reconstruction.
    from eval.window_position import inference_windows

    support_intervals: list[tuple[float, float]] = []
    clip_seconds: list[float] = []
    latest_word_phases: list[float] = []
    word_bin_windows = [0] * (len(POSITION_EDGES) - 1)
    missing_audio = 0
    n_clips = 0
    for window_id, window in decode["windows"].items():
        audio_path = audio_dir / f"{window_id}.wav"
        if not audio_path.exists():
            missing_audio += 1
            continue
        n_clips += 1
        duration = wav_duration(audio_path)
        bounds = inference_windows(window.get("segments") or [])
        words = [w for segment in window.get("segments") or []
                 for w in (segment.get("words") or [])]
        for lo, hi in bounds:
            available = max(0.0, min(hi, duration) - min(max(lo, 0.0), duration))
            available = min(WINDOW_SECONDS, available)
            clip_seconds.append(available)
            support_intervals.append((0.0, available))
            phases = [float(w["s"]) - lo for w in words
                      if lo <= float(w["s"]) < hi]
            if phases:
                latest_word_phases.append(max(phases))
            occupied_bins = {min(int(p // 5.0), 5) for p in phases if 0 <= p <= 30}
            for i in occupied_bins:
                word_bin_windows[i] += 1

    if not clip_seconds:
        raise ValueError("no reconstructed inference windows with local audio")
    n = len(clip_seconds)
    occupancy = [x / WINDOW_SECONDS for x in clip_seconds]
    return {
        "n_decode_clips": len(decode["windows"]),
        "n_clips_with_local_audio": n_clips,
        "n_missing_audio": missing_audio,
        "n_reconstructed_windows": n,
        "available_source_audio": {
            "seconds": summary(clip_seconds),
            "occupancy": summary(occupancy),
            "histogram_seconds": histogram(clip_seconds),
            "position_by_5s": aggregate_bin_overlaps(support_intervals),
            "full_30s_windows": sum(x >= 30.0 - 1e-6 for x in clip_seconds),
            "at_least_25s_windows": sum(x >= 25.0 for x in clip_seconds),
        },
        "emitted_word_proxy": {
            "windows_with_any_word": len(latest_word_phases),
            "latest_word_phase_seconds": summary(latest_word_phases),
            "windows_with_word_by_5s": [
                {"lo_s": lo, "hi_s": hi, "share": round(word_bin_windows[i] / n, 6)}
                for i, (lo, hi) in enumerate(zip(POSITION_EDGES, POSITION_EDGES[1:]))
            ],
        },
    }


def dense_candidate_aggregate(rows: list[dict]) -> dict:
    from eval.hf_export import build_packs

    packs = build_packs.build(rows)
    existing_gate_failures = build_packs.gates(packs, rows)
    durations = [float(p["dur_sec"]) for p in packs]
    occupancy = [x / WINDOW_SECONDS for x in durations]
    share_20_30 = sum(20.0 <= x <= 30.0 for x in durations) / len(durations)
    frozen_gates = {
        "existing_accounting_and_timing_gates_pass": not existing_gate_failures,
        "at_least_80pct_packs_are_20_to_30s": share_20_30 >= 0.80,
        "median_pack_duration_at_least_20s": quantile(durations, 0.5) >= 20.0,
        "mean_encoder_audio_occupancy_at_least_75pct": statistics.fmean(occupancy) >= 0.75,
    }
    return {
        "n_packs": len(packs),
        "duration_seconds": summary(durations),
        "duration_histogram_seconds": histogram(durations),
        "encoder_audio_occupancy": summary(occupancy),
        "share_20_to_30_seconds": round(share_20_30, 6),
        "source_aligned_hours": round(sum(float(p["speech_sec"]) for p in packs) / 3600, 6),
        "inserted_silence_hours": round(sum(float(p["silence_sec"]) for p in packs) / 3600, 6),
        "existing_gate_failure_count": len(existing_gate_failures),
        "frozen_gates": frozen_gates,
        "construction_feasible": all(frozen_gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--decode", type=Path, default=DEFAULT_DECODE)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.train)
    rows = df.to_dict("records")
    raw_hours = float((df["end"].astype(float) - df["start"].astype(float)).sum() / 3600)
    if len(rows) != EXPECTED_TRAIN_ROWS or abs(raw_hours - EXPECTED_RAW_HOURS) > 1e-9:
        raise SystemExit(
            f"training identity check failed: rows={len(rows)}, raw_hours={raw_hours:.12f}"
        )
    decode = json.loads(args.decode.read_text())

    training = training_aggregate(rows)
    inference = inference_aggregate(decode, args.audio_dir)
    dense = dense_candidate_aggregate(rows)
    result = {
        "schema_version": 1,
        "audit": "training-window-density-2026-08-18",
        "definitions_frozen_in": "docs/specs/2026-08-18-training-window-density-audit.md",
        "inputs": {
            "training_parquet": {"path": str(args.train), "sha256": sha256_file(args.train)},
            "inference_decode_cache": {"path": str(args.decode), "sha256": sha256_file(args.decode)},
            "audio_headers": {
                "path": str(args.audio_dir),
                **wav_header_fingerprint(args.audio_dir, decode["windows"]),
            },
        },
        "training": training,
        "inference": inference,
        "comparison": {
            "mean_audio_occupancy_gap_inference_minus_training": round(
                inference["available_source_audio"]["occupancy"]["mean"]
                - training["intended_clip_audio_before_source_end_clamp"]["occupancy"]["mean"],
                6,
            ),
            "median_audio_occupancy_gap_inference_minus_training": round(
                inference["available_source_audio"]["occupancy"]["median"]
                - training["intended_clip_audio_before_source_end_clamp"]["occupancy"]["median"],
                6,
            ),
        },
        "dense_candidate": dense,
        "limits": [
            "Training speech density is a timestamp-span proxy, not frame-level VAD speech.",
            "The pod-built training WAV cache is unavailable locally; intended clip audio is before the final clamp to source-recording end.",
            "Inference emitted-word positions are model outputs, not ground-truth speech, and are descriptive only.",
            "Construction feasibility does not imply a WER or deletion improvement.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote aggregate-only result: {args.out}")
    print(f"training rows={training['n_rows']}, raw mean={training['raw_speech_span_proxy']['seconds']['mean']:.3f}s, "
          f"clip mean={training['intended_clip_audio_before_source_end_clamp']['seconds']['mean']:.3f}s")
    print(f"inference windows={inference['n_reconstructed_windows']}, "
          f"mean occupancy={inference['available_source_audio']['occupancy']['mean']:.3f}")
    print(f"dense packs={dense['n_packs']}, feasible={dense['construction_feasible']}")


if __name__ == "__main__":
    main()
