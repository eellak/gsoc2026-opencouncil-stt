"""Cut the overlap-free contiguous packs into a training arm the pod can read.

Frozen criteria: docs/specs/2026-08-19-overlap-clean-selection.md. Each pack is ONE
continuous span of a single speaker, already accepted by the overlap predicate, with its
two boundaries placed acoustically (quietest frame inside the space diarization says
belongs to nobody else). The target text is the meeting's own utterance text, in order.

Output: pack wavs plus a JSONL with `audio` / `text_p` / `text_pn`, the format
notebooks/train_runpod.py already reads through PACK_MANIFEST. Audio and text stay under
~/.cache, never in git.

    .venv-eval/bin/python -m eval.controlled_eval.build_clean_pack_arm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from eval.controlled_eval.boundary_snap import snap
from eval.controlled_eval.overlap_clip_census import (
    census, contiguous_packs, load_jobs, transcript_rows,
)

DIAR = Path.home() / ".cache/oc-public/training-diarization-2026-08"
TRANSCRIPTS = Path.home() / ".cache/oc-public/training-transcripts-2026-08"
OUT = Path.home() / ".cache/oc-public/clean-pack-arm-2026-08"
SR = 16000
SNAP_SEARCH = 0.25


def read_window(audio: Path, start: float, duration: float) -> np.ndarray:
    """Decode a short span to mono float32 without touching the rest of the recording."""
    run = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}",
         "-i", str(audio), "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
        capture_output=True)
    if run.returncode != 0:
        raise RuntimeError(f"decode failed: {run.stderr[-200:]!r}")
    return np.frombuffer(run.stdout, dtype="float32")


def snapped_edges(audio: Path, pack: dict) -> tuple[float, float]:
    """Refine the pack's two cuts acoustically; pad_bounds already fenced the search."""
    lo = max(0.0, pack["start"] - SNAP_SEARCH)
    head = read_window(audio, lo, (pack["start"] - lo) + SNAP_SEARCH)
    start = lo + snap(head, SR, pack["start"] - lo, 0.0, len(head) / SR, "start")
    tail_lo = max(0.0, pack["end"] - SNAP_SEARCH)
    tail = read_window(audio, tail_lo, SNAP_SEARCH * 2)
    end = tail_lo + snap(tail, SR, pack["end"] - tail_lo, 0.0, len(tail) / SR, "end")
    if end - start < 5.0:                      # a degenerate snap is worse than no snap
        return pack["start"], pack["end"]
    return round(start, 3), round(end, 3)


def cut(audio: Path, start: float, end: float, out_wav: Path) -> None:
    run = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
         "-i", str(audio), "-ar", str(SR), "-ac", "1", str(out_wav)], capture_output=True)
    if run.returncode != 0:
        raise RuntimeError(f"cut failed: {run.stderr[-200:]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diar", type=Path, default=DIAR)
    parser.add_argument("--transcripts", type=Path, default=TRANSCRIPTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    jobs = load_jobs(args.diar)
    rows, _ = census(transcript_rows(args.transcripts, jobs), jobs)
    packs = contiguous_packs(rows, {u: j["turns"] for u, j in jobs.items()},
                             {u: j["audio_end"] for u, j in jobs.items()})
    if args.limit:
        packs = packs[:args.limit]

    texts: dict[tuple[str, str], dict[str, str]] = {}
    clips = args.out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "packs.jsonl"
    written, failed, seconds = 0, 0, 0.0
    with open(manifest, "w") as handle:
        for pack in packs:
            key = (pack["city_id"], pack["meeting_id"])
            if key not in texts:
                payload = json.loads((args.transcripts / f"{key[0]}__{key[1]}.json").read_text())
                texts[key] = {u["id"]: (u.get("text") or "")
                              for s in payload["transcript"] for u in s.get("utterances", [])}
            text = " ".join(texts[key].get(i, "") for i in pack["utterance_ids"]).strip()
            if not text:
                failed += 1
                continue
            audio = args.diar / "audio" / (hashlib.sha256(
                pack["audio_url"].encode()).hexdigest()[:16] + ".mp3")
            name = hashlib.sha256(f"{pack['audio_url']}|{pack['utterance_ids'][0]}"
                                  .encode()).hexdigest()[:16]
            wav = clips / f"{name}.wav"
            try:
                start, end = snapped_edges(audio, pack)
                if not wav.is_file():
                    cut(audio, start, end, wav)
            except RuntimeError:
                failed += 1
                continue
            seconds += end - start
            written += 1
            handle.write(json.dumps({
                "audio": f"clips/{wav.name}",
                "text_pn": text,
                "text_p": text,
                "city_id": pack["city_id"],
                "meeting_id": pack["meeting_id"],
                "person_id": pack["person_id"],
                "speech_sec": pack["speech_sec"],
                "span_sec": round(end - start, 3),
                "n_utterances": pack["n_utterances"],
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(manifest), "packs": written, "failed": failed,
                      "audio_hours": round(seconds / 3600, 3),
                      "mean_span_sec": round(seconds / written, 2) if written else None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
