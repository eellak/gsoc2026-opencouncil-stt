"""Arm P: cut the audio at silences, decode each piece, concatenate.

This is the deployable form of arm P of `exp-2026-08-17-chunking-aware-decoding`.
`eval/chunking_decode.py` is the experiment harness -- it knows about frozen
windows, caches, bootstrap intervals and preregistered gates. This module knows
none of that: it takes audio and returns text.

    from serve.decode_p import DecodeP
    decoder = DecodeP.load()
    result = decoder.transcribe("meeting.mp3")
    print(result.text)

or from the shell::

    python -m serve.decode_p transcribe audio.wav
    python -m serve.decode_p config          # print the frozen policy and its hash

WHAT THE POLICY DOES

1. Silero VAD marks the silences (>= 0.5 s, no speech padding).
2. The audio is accumulated up to 29.5 s and then cut at the LATEST silence
   midpoint that leaves a piece of at least 5.0 s containing at least 2.5 s of
   voiced audio. If no such candidate exists, the latest in-range candidate is
   used; if there is none at all, the cut is forced blind at the 29.5 s ceiling.
   A short final piece is merged into its predecessor, or the shared boundary is
   moved, so no piece is under 5.0 s.
3. Every piece is decoded independently with the frozen CONTROL config. No
   context margins, no `initial_prompt`, no seam de-duplication -- those are arm
   X, a different and unendorsed policy.
4. The piece texts are concatenated in order and the segment timestamps are
   shifted by each piece's offset.

WHAT THE EVIDENCE SAYS, EXACTLY

On the 39 frozen evaluation windows, same decoder and same normalizer:

    arm A (control, blind cutting)   WER 0.15893   del 0.06003   sub 0.07875
    arm V (vad_filter)               WER 0.15246   del 0.05911   sub 0.07690
    arm P (this policy)              WER 0.14751   del 0.05575   sub 0.07220

P's WER delta against the control is -0.01142, cluster-bootstrap CI
[-0.02025, -0.00195], which excludes zero. **P nevertheless FAILED the
preregistered gate of that experiment**, because the gate was on DELETIONS and
P's deletion delta has a CI that includes zero. The win comes from
SUBSTITUTIONS. P is the best measured arm by WER; it is not an arm that passed a
gate. Any text written about this policy has to say so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval import chunking_decode as CD  # noqa: E402

POLICY_ID = "decode-policy-p-silence-cut"
POLICY_VERSION = "1.0.0"

VAD_SILENCE = "vad_silence"
FORCED = "forced"

#: The frozen policy. Changing any value here is a new policy, not a tweak.
#: Every number is read from the experiment module, never retyped, so that a
#: change to the measured arm shows up as a changed hash rather than as silence.
_POLICY_DATA: dict[str, Any] = {
    "policy_id": POLICY_ID,
    "version": POLICY_VERSION,
    "experiment": CD.EXPERIMENT,
    # the harness string above is baked into the on-disk caches and predates the
    # ledger record, whose id is a day later. Both name the same experiment.
    "ledger_experiment_id": "exp-2026-08-18-chunking-aware-decoding",
    "arm": "P",
    "model_artifact_id": CD.MODEL_ARTIFACT_ID,
    "decode": dict(CD.CONTROL),
    "vad": {
        "detector": "faster_whisper.vad.silero",
        "min_silence_duration_ms": int(CD.MINIMUM_SILENCE_SECONDS * 1000),
        "speech_pad_ms": 0,
    },
    "segmentation": {
        "splitter": "accumulate_then_cut@1",
        "max_chunk_seconds": CD.MAX_CHUNK_SECONDS,
        "min_chunk_seconds": CD.MIN_CHUNK_SECONDS,
        "min_speech_seconds": CD.MIN_SPEECH_SECONDS,
        "min_silence_seconds": CD.MIN_SILENCE_SECONDS,
    },
    "combine": "plain_concat_offset_segments@1",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, (dict, MappingProxyType)):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


POLICY: Mapping[str, Any] = _freeze(_POLICY_DATA)


def _digest(policy: Any) -> str:
    return hashlib.sha256(
        json.dumps(_thaw(policy), sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def policy_sha256() -> str:
    return _digest(POLICY)


#: Pinned so that a test can fail on a policy change rather than merely notice
#: that the hash moved. Recorded in the ledger as `artifact-decode-policy-p`.
EXPECTED_POLICY_SHA256 = (
    "5ae98472227696e0976eeb80390d392106e80ea6485fa96e13b78f3ec010186b"
)


# ------------------------------------------------------------------ the planner

def plan_pieces(duration: float,
                silences: Iterable[CD.Silence]) -> tuple[list[CD.Piece], dict]:
    """Where the cuts go. Model-free apart from the VAD that produced ``silences``.

    Calls the experiment's own splitter with the policy's constants, so the
    served segmentation cannot differ from the measured one by construction.
    """
    seg = POLICY["segmentation"]
    pieces, counters = CD.split_accumulating(
        duration, silences,
        max_chunk=seg["max_chunk_seconds"],
        min_chunk=seg["min_chunk_seconds"],
        min_speech=seg["min_speech_seconds"],
        min_silence=seg["min_silence_seconds"],
    )
    longest = max(p.end - p.start for p in pieces)
    if longest > seg["max_chunk_seconds"] + 1e-6:
        raise AssertionError(
            f"a piece is {longest:.3f} s, over the {seg['max_chunk_seconds']} s "
            "ceiling; it would be truncated inside one Whisper window"
        )
    return pieces, counters


@dataclass(frozen=True)
class Boundary:
    """One internal cut and the reason it can be defended.

    ``kind`` is a *description of where the boundary landed*, computed after the
    fact: `vad_silence` means the cut time falls inside a reported silence of at
    least ``min_silence_seconds``. It is deliberately not read from the splitter's
    own ``cut_silence_duration`` field, which the short-tail repair carries over
    onto a boundary that has since MOVED -- see the `stale_silence_provenance`
    case in the test file.
    """

    at: float
    kind: str
    silence_seconds: float | None


def _normalized_silences(silences: Iterable[CD.Silence]) -> list[CD.Silence]:
    ordered = sorted((CD._as_silence(s) for s in silences), key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        if b.start < a.end - 1e-9:
            raise ValueError(
                f"reported silences overlap: [{a.start}, {a.end}) and "
                f"[{b.start}, {b.end}); the boundary reason would be a guess"
            )
    return ordered


def classify_boundaries(pieces: Sequence[CD.Piece],
                        silences: Iterable[CD.Silence],
                        min_silence: float | None = None) -> list[Boundary]:
    """Label every internal boundary so a bad piece is traceable to its cut."""
    floor = (POLICY["segmentation"]["min_silence_seconds"]
             if min_silence is None else min_silence)
    quiet = _normalized_silences(silences)
    out: list[Boundary] = []
    for piece in list(pieces)[:-1]:
        at = piece.end
        found = None
        for silence in quiet:
            if silence.start > at:
                break
            # half-open [start, end): a cut exactly at the end of a silence is
            # the first sample of speech again, so it is not a silent boundary
            if silence.start <= at < silence.end and silence.duration >= floor:
                found = silence
                break
        out.append(Boundary(round(at, 6), VAD_SILENCE, round(found.duration, 6))
                   if found else Boundary(round(at, 6), FORCED, None))
    return out


# ------------------------------------------------------------------ the decoder

@dataclass
class Transcript:
    text: str
    segments: list[dict]
    boundaries: list[Boundary]
    audio_seconds: float
    policy_sha256: str
    seed: int | None = None
    counters: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": self.segments,
            "boundaries": [{"at": b.at, "kind": b.kind,
                            "silence_seconds": b.silence_seconds}
                           for b in self.boundaries],
            "audio_seconds": self.audio_seconds,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "policy_sha256": self.policy_sha256,
            "model_artifact_id": CD.MODEL_ARTIFACT_ID,
            "seed": self.seed,
            "counters": self.counters,
        }


def verify_model_dir(model_dir: Path) -> str:
    """Refuse to load anything that is not `artifact-ct2-fixed`.

    Verifies the directory that will actually be loaded, not a default path.
    The hash covers ``model.bin`` only, which is what the ledger records; the
    tokenizer and config files beside it are not covered.
    """
    model_bin = model_dir / "model.bin"
    if not model_bin.exists():
        raise FileNotFoundError(
            f"CT2 model directory is absent: {model_dir}; "
            f"the policy needs {CD.MODEL_ARTIFACT_ID}"
        )
    actual = CD.sha16(model_dir)
    if actual != CD.MODEL_SHA256_16:
        raise ValueError(
            f"{model_bin} is {actual}, expected {CD.MODEL_SHA256_16} "
            f"({CD.MODEL_ARTIFACT_ID}); an output whose producing model is "
            "unknown is not evidence"
        )
    return actual


def _as_audio(audio: Any) -> Any:
    """Freeze the input contract: mono, float32, 16 kHz, one dimension."""
    import numpy as np

    if isinstance(audio, (str, Path)):
        audio = CD._load_audio(Path(audio))
    array = np.asarray(audio)
    if array.ndim != 1:
        raise ValueError(
            f"audio must be a mono sample array, got shape {array.shape}")
    if array.dtype != np.float32:
        raise ValueError(
            f"audio must be float32 at {CD.SAMPLE_RATE} Hz, got {array.dtype}; "
            "resampling silently would change what was measured")
    if array.size == 0:
        raise ValueError("audio is empty; there is nothing to segment or decode")
    return array


class DecodeP:
    """The arm P decode policy, bound to one loaded model."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def load(cls, model_dir: str | Path | None = None,
             device: str | None = None,
             compute_type: str | None = None,
             cpu_threads: int | None = None,
             verify: bool = True,
             _model_factory: Callable[..., Any] | None = None) -> "DecodeP":
        model_dir = Path(model_dir or CD.MODEL_DIR)
        if verify:
            verify_model_dir(model_dir)          # before anything is constructed
        if _model_factory is None:
            from faster_whisper import WhisperModel
            _model_factory = WhisperModel
        model = _model_factory(
            str(model_dir),
            device=device or CD.DEVICE,
            compute_type=compute_type or CD.COMPUTE,
            cpu_threads=cpu_threads if cpu_threads is not None else CD.THREADS,
        )
        return cls(model)

    # ---------------------------------------------------------------- the policy

    def _detect(self, audio: Any, duration: float):
        return CD.detect_silences(audio, duration, self.model)

    def plan(self, audio: Any) -> dict:
        """Segmentation only: no decode, no GPU, seconds even on a whole meeting."""
        audio = _as_audio(audio)
        duration = len(audio) / CD.SAMPLE_RATE
        silences, source, error = self._detect(audio, duration)
        pieces, counters = plan_pieces(duration, silences)
        boundaries = classify_boundaries(pieces, silences)
        return {
            "audio_seconds": duration,
            "pieces": pieces,
            "boundaries": boundaries,
            "counters": self._counters(counters, boundaries, source, error),
        }

    @staticmethod
    def _counters(split_counters: dict, boundaries: Sequence[Boundary],
                  source: str, error: str | None) -> dict:
        counters = dict(split_counters)
        counters.update({
            "n_boundaries": len(boundaries),
            "vad_silence_boundaries": sum(1 for b in boundaries
                                          if b.kind == VAD_SILENCE),
            "forced_boundaries": sum(1 for b in boundaries if b.kind == FORCED),
            "detection_source": source,
            "detection_error": error,
        })
        return counters

    def transcribe(self, audio: Any, seed: int | None = None) -> Transcript:
        """Decode any audio under the frozen policy.

        ``seed`` is passed to ``ctranslate2.set_random_seed``, which is
        PROCESS-GLOBAL: it only reproduces if nothing else decodes concurrently
        in this process. With ``seed=None`` the RNG is left alone, so output
        under the temperature fallback ladder depends on process history and is
        not reproducible.
        """
        audio = _as_audio(audio)
        duration = len(audio) / CD.SAMPLE_RATE
        silences, source, error = self._detect(audio, duration)
        pieces, split_counters = plan_pieces(duration, silences)
        boundaries = classify_boundaries(pieces, silences)

        if seed is not None:
            import ctranslate2
            ctranslate2.set_random_seed(seed)

        cuts = [0] + [int(round(p.end * CD.SAMPLE_RATE)) for p in pieces[:-1]]
        cuts.append(len(audio))
        if sorted(cuts) != cuts:
            raise AssertionError("sample boundaries are not monotonic")

        piece_results: list[dict] = []
        for index, piece in enumerate(pieces):
            begin, end = cuts[index], cuts[index + 1]
            text, segments, _ = CD._decode_one(
                self.model, audio[begin:end], dict(CD.CONTROL))
            piece_results.append({
                "start_sec": piece.start,
                "end_sec": piece.end,
                "text": text,
                "segments": segments,
            })
        combined = CD.combine_piece_transcripts(piece_results)

        counters = self._counters(split_counters, boundaries, source, error)
        counters["n_segments"] = len(combined["segments"])
        return Transcript(
            text=combined["text"],
            segments=combined["segments"],
            boundaries=boundaries,
            audio_seconds=duration,
            policy_sha256=policy_sha256(),
            seed=seed,
            counters=counters,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The arm P silence-cutting decode policy")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="print the frozen policy and its hash")
    for name, helptext in (("transcribe", "decode one audio file"),
                           ("plan", "segment one audio file without decoding")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("audio")
        p.add_argument("--json", action="store_true", help="emit the full record")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--model-dir", default=None)
        p.add_argument("--device", default=None)
        p.add_argument("--compute-type", default=None)
        p.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()

    if args.command == "config":
        print(json.dumps({**_thaw(POLICY), "policy_sha256": policy_sha256()},
                         ensure_ascii=False, indent=1))
        return

    if args.command == "plan":
        # segmentation needs no model, but the VAD fallback does; verify anyway
        decoder = DecodeP.load(args.model_dir, args.device, args.compute_type,
                               args.cpu_threads)
        result = decoder.plan(args.audio)
        print(json.dumps({
            "audio_seconds": result["audio_seconds"],
            "pieces": [{"start_sec": round(p.start, 3), "end_sec": round(p.end, 3)}
                       for p in result["pieces"]],
            "boundaries": [{"at": b.at, "kind": b.kind,
                            "silence_seconds": b.silence_seconds}
                           for b in result["boundaries"]],
            "counters": result["counters"],
            "policy_sha256": policy_sha256(),
        }, ensure_ascii=False, indent=1))
        return

    decoder = DecodeP.load(args.model_dir, args.device, args.compute_type,
                           args.cpu_threads)
    result = decoder.transcribe(args.audio, seed=args.seed)
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=1))
    else:
        print(result.text)


if __name__ == "__main__":
    main()
