"""Cut the overlap-free SPLICED packs into a training arm the pod can read.

Same frozen acceptance predicate as the contiguous arm
(docs/specs/2026-08-19-overlap-clean-selection.md), different assembly. A contiguous
pack is one continuous span of a single speaker and therefore requires the *pauses
between* its utterances to be clean too. A spliced pack keeps only the accepted
utterances and joins them with a short faded silence, so it can use speech whose
surrounding pauses are contaminated. That is where the extra hours come from, and it is
also the cost: the joins are synthetic, and a real 30 s inference window contains real
pauses.

Output schema matches the contiguous arm's manifest.jsonl exactly, so
notebooks/train_runpod.py reads it through PACK_MANIFEST unchanged.

    .venv-eval/bin/python -m eval.controlled_eval.build_spliced_pack_arm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.controlled_eval.boundary_snap import snap
from eval.controlled_eval.overlap_clip_census import (
    build_packs, census, load_jobs, pack_speech_sec, transcript_rows,
)

DIAR = Path.home() / ".cache/oc-public/training-diarization-2026-08"
TRANSCRIPTS = Path.home() / ".cache/oc-public/training-transcripts-2026-08"
OUT = Path.home() / ".cache/oc-public/spliced-pack-arm-2026-08"
SR = 16000
SEPARATOR_SEC = 0.4
TARGET_SPEECH_SEC = 22.0
PAD_SEC = 0.25          # search room on each side for the acoustic snap
FADE_SEC = 0.01         # click guard at every join

# A join that is always 0.4 s with always the same fade is a signature the model can
# learn as a "reset cue" that no real meeting contains. Jitter both so the only thing
# left to learn is "speech resumes", not "0.4 s of digital silence resumes".
# Off by default: the frozen arm keeps its fixed join.
SEPARATOR_RANGE = (0.15, 0.60)
FADE_RANGE = (0.005, 0.030)


def read_window(audio: Path, start: float, duration: float) -> np.ndarray:
    run = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}",
         "-i", str(audio), "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
        capture_output=True)
    if run.returncode != 0:
        raise RuntimeError(f"decode failed: {run.stderr[-200:]!r}")
    return np.frombuffer(run.stdout, dtype="float32")


def utterance_audio(audio: Path, start: float, end: float,
                    rng: "np.random.Generator | None" = None) -> np.ndarray:
    """Decode one utterance with padding, then place both cuts in the quietest frame.

    One ffmpeg call per utterance, not per cut: the padded window already contains
    everything the snap is allowed to look at, so the second cut costs no extra decode.
    """
    lo = max(0.0, start - PAD_SEC)
    samples = read_window(audio, lo, (end + PAD_SEC) - lo)
    if samples.size == 0:
        raise RuntimeError("empty decode")
    span = samples.size / SR
    a = snap(samples, SR, start - lo, 0.0, min(start - lo + PAD_SEC, span), "start")
    b = snap(samples, SR, end - lo, max(0.0, end - lo - PAD_SEC), span, "end")
    if b - a < 0.2:                                  # a degenerate snap is worse than none
        a, b = start - lo, end - lo
    clip = samples[int(a * SR):int(b * SR)].copy()
    fade = int((float(rng.uniform(*FADE_RANGE)) if rng is not None else FADE_SEC) * SR)
    if clip.size > 2 * fade:
        clip[:fade] *= np.linspace(0.0, 1.0, fade, dtype="float32")
        clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype="float32")
    return clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diar", type=Path, default=DIAR)
    parser.add_argument("--transcripts", type=Path, default=TRANSCRIPTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jitter", action="store_true",
                        help="randomise separator length and fade shape at every join")
    parser.add_argument("--jitter-seed", type=int, default=13)
    args = parser.parse_args()

    jobs = load_jobs(args.diar)
    rows, _ = census(transcript_rows(args.transcripts, jobs), jobs)
    packs = build_packs(rows, target_speech_sec=TARGET_SPEECH_SEC,
                        separator_sec=SEPARATOR_SEC)
    # Same 80%-of-target rule the yield estimate used, so the arm and the estimate that
    # justified building it count the same packs.
    packs = [p for p in packs if pack_speech_sec(p) >= TARGET_SPEECH_SEC * 0.8]
    if args.limit:
        packs = packs[:args.limit]

    clips = args.out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.jitter_seed) if args.jitter else None
    separator = np.zeros(int(SEPARATOR_SEC * SR), dtype="float32")

    def gap() -> np.ndarray:
        if rng is None:
            return separator
        return np.zeros(int(float(rng.uniform(*SEPARATOR_RANGE)) * SR), dtype="float32")
    texts: dict[tuple[str, str], dict[str, str]] = {}
    written = failed = 0
    seconds = 0.0

    with open(args.out / "manifest.jsonl", "w") as handle:
        for pack in packs:
            head = pack[0]
            key = (head["city_id"], head["meeting_id"])
            if key not in texts:
                payload = json.loads(
                    (args.transcripts / f"{key[0]}__{key[1]}.json").read_text())
                texts[key] = {u["id"]: (u.get("text") or "")
                              for s in payload["transcript"] for u in s.get("utterances", [])}
            text = " ".join(texts[key].get(r["utterance_id"], "") for r in pack).strip()
            if not text:
                failed += 1
                continue
            audio = args.diar / "audio" / (hashlib.sha256(
                head["audio_url"].encode()).hexdigest()[:16] + ".mp3")
            pack_id = hashlib.sha256(
                f"{head['audio_url']}|{head['utterance_id']}|spliced".encode()).hexdigest()[:16]
            wav = clips / f"{pack_id}.wav"
            try:
                pieces = [utterance_audio(audio, r["start"], r["end"], rng) for r in pack]
            except RuntimeError:
                failed += 1
                continue
            waveform = pieces[0]
            for piece in pieces[1:]:
                waveform = np.concatenate([waveform, gap(), piece])
            if not wav.is_file():
                sf.write(str(wav), waveform, SR)
            dur = round(waveform.size / SR, 3)
            seconds += dur
            written += 1
            handle.write(json.dumps({
                "pack_id": pack_id,
                "audio": f"clips/{wav.name}",
                "text_p": text,
                "text_pn": text,
                "dur_sec": dur,
                "city_id": head["city_id"],
                "meeting_id": head["meeting_id"],
                "person_id": head["person_id"],
                "speech_sec": pack_speech_sec(pack),
                "n_utterances": len(pack),
            }, ensure_ascii=False) + "\n")

    print(json.dumps({"manifest": str(args.out / "manifest.jsonl"), "packs": written,
                      "jitter": bool(args.jitter), "jitter_seed": args.jitter_seed,
                      "failed": failed, "audio_hours": round(seconds / 3600, 3),
                      "mean_dur_sec": round(seconds / written, 2) if written else None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
