"""Arm X: the seam-repairing decode policy, as a reusable configuration.

This is the deployable form of `exp-2026-08-20-seam-repair`. `eval/seam_repair.py`
is the experiment harness -- it knows about frozen windows, caches, bootstrap
intervals and pilot verdicts. This module knows none of that: it takes audio and
returns text, so the same policy can be pointed at a whole benchmark, a meeting, or
one file.

    from serve.decode_x import DecodeX
    decoder = DecodeX.load()
    result = decoder.transcribe("meeting.wav")
    print(result.text)

or from the shell::

    python -m serve.decode_x transcribe audio.wav
    python -m serve.decode_x config          # print the frozen policy and its hash

WHAT THE POLICY DOES, in one pass over the audio:

1. Silero VAD marks the silences.
2. The audio is cut into CORE pieces of at most 27.0 s. A boundary is chosen by a
   three-step ladder: a reported silence of at least 0.5 s where one exists;
   otherwise the lowest-mean Silero probability valley of at least 128 ms below
   0.35 inside the last 3.0 s before the ceiling; otherwise, blind.
3. Each piece is decoded from `[core_start - 0.5 s, core_end + 2.0 s]`, so every
   seam is heard with context on both sides. 27.0 + 0.5 + 2.0 = 29.5 s, still one
   Whisper window.
4. The previous pieces' core text is passed as `initial_prompt`, capped at 64
   tokens of the model's own tokenizer.
5. Every decoded word is owned by the core containing its own midpoint. The margins
   are context the decoder hears and never text it contributes.

Everything outside step 3's margins is the frozen CONTROL decode config, unchanged.

STATUS: this policy is EXPLORATORY. See the ledger record for what has and has not
been measured. It is packaged so it CAN be run over a benchmark, not because a
benchmark has endorsed it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval import seam_repair as SR      # noqa: E402
from eval import chunking_decode as CD  # noqa: E402

POLICY_ID = "decode-policy-x-seam-repair"
POLICY_VERSION = "1.0.0"
EXPERIMENT = SR.EXPERIMENT

#: The frozen policy. Changing any value here is a new policy, not a tweak.
POLICY: dict[str, Any] = {
    "policy_id": POLICY_ID,
    "version": POLICY_VERSION,
    "experiment": EXPERIMENT,
    "decode": dict(SR.CONTROL, word_timestamps=True),
    "segmentation": {
        "max_core_seconds": SR.X_MAX_CHUNK_SECONDS,
        "min_core_seconds": SR.MIN_CHUNK_SECONDS,
        "min_speech_seconds": SR.MIN_SPEECH_SECONDS,
        "min_silence_seconds": SR.MIN_SILENCE_SECONDS,
        "valley_slack_seconds": SR.X_SLACK_SECONDS,
        "valley_prob_threshold": SR.VALLEY_PROB_THRESHOLD,
        "valley_min_frames": SR.MIN_VALLEY_FRAMES,
        "vad_hop_samples": SR.VAD_HOP_SAMPLES,
    },
    "context": {
        "lookback_seconds": SR.X_LOOKBACK_SECONDS,
        "lookahead_seconds": SR.X_LOOKAHEAD_SECONDS,
        "prompt_max_tokens": SR.PROMPT_MAX_TOKENS,
    },
    "ownership": "word_core_ownership@1",
}


def policy_sha256() -> str:
    return hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def _assert_fits_one_window() -> None:
    total = (POLICY["segmentation"]["max_core_seconds"]
             + POLICY["context"]["lookback_seconds"]
             + POLICY["context"]["lookahead_seconds"])
    if total > SR.WINDOW_SECONDS + 1e-9:
        raise AssertionError(
            f"core + margins is {total} s, which does not fit one "
            f"{SR.WINDOW_SECONDS} s Whisper window"
        )


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Boundary:
    at: float
    kind: str


@dataclass
class Transcript:
    text: str
    words: list[Word]
    boundaries: list[Boundary]
    audio_seconds: float
    policy_sha256: str
    counters: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "words": [{"text": w.text, "start": w.start, "end": w.end}
                      for w in self.words],
            "boundaries": [{"at": b.at, "kind": b.kind} for b in self.boundaries],
            "audio_seconds": self.audio_seconds,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "policy_sha256": self.policy_sha256,
            "counters": self.counters,
        }


class DecodeX:
    """The arm X decode policy, bound to one loaded model."""

    def __init__(self, model: Any) -> None:
        _assert_fits_one_window()
        self.model = model

    @classmethod
    def load(cls, model_dir: str | Path | None = None,
             device: str | None = None,
             compute_type: str | None = None,
             cpu_threads: int | None = None,
             verify: bool = True) -> "DecodeX":
        from faster_whisper import WhisperModel

        model_dir = Path(model_dir or SR.MODEL_DIR)
        if verify and model_dir == Path(SR.MODEL_DIR):
            CD.verify_model()          # an output whose model is unknown is not evidence
        model = WhisperModel(
            str(model_dir),
            device=device or SR.DEVICE,
            compute_type=compute_type or SR.COMPUTE,
            cpu_threads=cpu_threads if cpu_threads is not None else SR.THREADS,
        )
        return cls(model)

    # ---------------------------------------------------------------- the policy

    def plan(self, audio: Any, duration: float) -> tuple[list[SR.RPiece], dict]:
        """Where the cuts go, and why. Model-free apart from the VAD."""
        silences, _, _ = CD.detect_silences(audio, duration, self.model)
        probs = SR.speech_probabilities(audio)
        energies = SR.frame_energies(audio, probs.shape[0])
        return SR.split_with_valley_repair(
            duration, silences, probs, energies,
            max_chunk=POLICY["segmentation"]["max_core_seconds"],
            slack=POLICY["segmentation"]["valley_slack_seconds"],
        )

    def transcribe(self, audio: str | Path | Any,
                   seed: int | None = None) -> Transcript:
        """Decode any audio under the frozen policy."""
        import ctranslate2

        if isinstance(audio, (str, Path)):
            audio = CD._load_audio(Path(audio))
        duration = len(audio) / SR.SAMPLE_RATE
        pieces, counters = self.plan(audio, duration)
        SR.sample_boundaries(pieces, len(audio))      # asserts exact integer tiling
        cores = [(p.start, p.end) for p in pieces]
        lookback = POLICY["context"]["lookback_seconds"]
        lookahead = POLICY["context"]["lookahead_seconds"]
        config = POLICY["decode"]
        tokenizer = self.model.hf_tokenizer

        if seed is not None:
            ctranslate2.set_random_seed(seed)

        candidates: list[dict] = []
        carried = ""
        for index, piece in enumerate(pieces):
            begin = max(0, int(round((piece.start - lookback) * SR.SAMPLE_RATE)))
            end = min(len(audio), int(round((piece.end + lookahead) * SR.SAMPLE_RATE)))
            offset = begin / SR.SAMPLE_RATE
            ids = SR.prompt_ids_for(
                tokenizer, carried,
                max_tokens=POLICY["context"]["prompt_max_tokens"]) if index else []
            extra = {"initial_prompt": ids} if ids else {}
            records = SR._decode(self.model, audio[begin:end], config, **extra)
            for segment_index, segment in enumerate(records):
                for word_index, word in enumerate(segment.get("words") or []):
                    if word.get("start") is None or word.get("end") is None:
                        raise ValueError("the policy requires a timestamp on every word")
                    start = float(word["start"]) + offset
                    stop = float(word["end"]) + offset
                    candidates.append({
                        "text": str(word.get("word", "")),
                        "start": start, "end": stop,
                        "midpoint": (start + stop) / 2.0,
                        "piece_index": index,
                        "segment_index": segment_index,
                        "word_index": word_index,
                    })
            core_text = "".join(
                c["text"] for c in candidates
                if c["piece_index"] == index
                and SR.owning_core_index(c["midpoint"], cores) == index
            )
            carried = " ".join(p for p in (carried.strip(), core_text.strip()) if p)

        merged = SR.merge_words_by_core(candidates, cores)
        counters = dict(counters)
        counters.update({
            "candidates": len(candidates),
            "kept": merged["kept"],
            "dropped_as_context": merged["non_owner"],
            "outside_audio": merged["outside_audio"],
        })
        return Transcript(
            text=merged["text"],
            words=[Word(w["text"], w["start"], w["end"]) for w in merged["words"]],
            boundaries=[Boundary(round(p.end, 3), p.end_kind) for p in pieces[:-1]],
            audio_seconds=duration,
            policy_sha256=policy_sha256(),
            counters=counters,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="The arm X seam-repair decode policy")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="print the frozen policy and its hash")
    t = sub.add_parser("transcribe", help="decode one audio file")
    t.add_argument("audio")
    t.add_argument("--json", action="store_true", help="emit the full record")
    t.add_argument("--model-dir", default=None)
    t.add_argument("--device", default=None)
    t.add_argument("--compute-type", default=None)
    t.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()

    if args.command == "config":
        print(json.dumps({**POLICY, "policy_sha256": policy_sha256()},
                         ensure_ascii=False, indent=1))
        return
    decoder = DecodeX.load(args.model_dir, args.device, args.compute_type,
                           args.cpu_threads)
    result = decoder.transcribe(args.audio)
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=1))
    else:
        print(result.text)


if __name__ == "__main__":
    main()
