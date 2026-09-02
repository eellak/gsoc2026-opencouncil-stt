"""Seam repair on arm P, ``exp-2026-08-20-seam-repair``.

Three arms on top of arm P of ``exp-2026-08-18-chunking-aware-decoding``, plus one
derived free:

* ``F`` repairs the forced (blind) cut: when no VAD silence exists before the
  29.5 s deadline, cut at the lowest-mean Silero probability valley inside the last
  two seconds instead of cutting blind;
* ``C`` chains context across the seam by passing the previous pieces' tail as
  ``initial_prompt``, capped at 64 tokens of the model's own tokenizer;
* ``W`` decodes overlapping windows with ``word_timestamps=True`` and assigns each
  WORD to the window whose kept tile contains that word's own midpoint;
* ``E-WT`` re-merges W's cached decodes with the OLD segment-level rule, so the
  finer merge can be separated from merely having word timestamps on.

Arms, constants, pilot criteria and scoring are frozen in
``docs/specs/2026-08-20-seam-repair-prereg.md`` (with Codex amendment 1) and were
committed before any arm was decoded. Everything here is EXPLORATORY: the same 39
windows generated the forced-cut hypothesis.

Hypotheses and audio stay under ``$SC``; only aggregates go in the repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval import chunking_decode as CD  # noqa: E402
from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402

import decode_ablation as DA  # noqa: E402

EXPERIMENT = "exp-2026-08-20-seam-repair"
CONTROL = dict(CD.CONTROL)
SAMPLE_RATE = CD.SAMPLE_RATE

# ---- arm F, frozen -----------------------------------------------------------
SLACK_SECONDS = 2.0
VALLEY_PROB_THRESHOLD = 0.35        # Silero's own neg_threshold default (0.5 - 0.15)
MIN_VALLEY_FRAMES = 4               # 4 x 32 ms = 128 ms run, over the 100 ms floor
VAD_HOP_SAMPLES = 512               # Silero's frame hop at 16 kHz

# ---- arm C, frozen -----------------------------------------------------------
PROMPT_MAX_TOKENS = 64

# ---- arm X = F + C + W together, frozen --------------------------------------
# The three mechanisms compose only if the decoder input still fits ONE Whisper
# window: a core piece plus an overlap margin on each side must stay under 30 s.
# So X cuts at 25.5 s instead of 29.5 s and pays for the margin with core length.
# Amendment 5: the margin is ASYMMETRIC because the two sides are not used alike.
# Measured on the first X decode: the left piece emits 3.7-5.3 words past its core,
# the right piece emits 0.32-1.43 words before its core start. A symmetric margin
# spends ceiling on a lookback that is almost never transcribed.
X_LOOKBACK_SECONDS = 0.5
X_LOOKAHEAD_SECONDS = 2.0
X_MAX_CHUNK_SECONDS = 27.0          # 27.0 + 0.5 + 2.0 = 29.5, inside one window
X_SLACK_SECONDS = 3.0               # the valley search only; its definition is unchanged

# ---- arm W, frozen -----------------------------------------------------------
WINDOW_SECONDS = CD.WHISPER_WINDOW_SECONDS
STRIDE_SECONDS = CD.OVERLAP_STRIDE_SECONDS
SEAM_BAND_SECONDS = 1.0

# ---- splitter constants inherited unchanged from arm P -----------------------
MAX_CHUNK_SECONDS = CD.MAX_CHUNK_SECONDS
MIN_CHUNK_SECONDS = CD.MIN_CHUNK_SECONDS
MIN_SPEECH_SECONDS = CD.MIN_SPEECH_SECONDS
MIN_SILENCE_SECONDS = CD.MIN_SILENCE_SECONDS

BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED = CD.BOOTSTRAP_REPLICATES, CD.BOOTSTRAP_SEED
MODEL_DIR = CD.MODEL_DIR
MODEL_ARTIFACT_ID = CD.MODEL_ARTIFACT_ID
MODEL_SHA256_16 = CD.MODEL_SHA256_16
DEVICE, COMPUTE, THREADS = CD.DEVICE, CD.COMPUTE, CD.THREADS

ARMS = ("F", "C", "W", "X")
DERIVED_ARMS = ("E-WT",)

VAD_SILENCE, PROBABILITY_VALLEY, BLIND, AUDIO_END = (
    "vad_silence", "probability_valley", "blind", "audio_end",
)

# Pilot windows: the five with the most forced cuts in the cached arm P decode,
# ties broken by window id ascending. Frozen in the spec; asserted at run time.
PILOT_WINDOW_IDS = (
    "win_orestiada_dec11_2025_600052",
    "win_orestiada_feb11_2026_149776",
    "win_argos_apr7_2026_960810",
    "win_argos_aug29_2025_2731295",
    "win_argos_dec23_2025_581922",
)


ARM_MANIFESTS: dict[str, dict] = {
    "F": {
        "splitter": "split_with_valley_repair@1",
        "slack_seconds": SLACK_SECONDS,
        "valley_prob_threshold": VALLEY_PROB_THRESHOLD,
        "min_valley_frames": MIN_VALLEY_FRAMES,
        "vad_hop_samples": VAD_HOP_SAMPLES,
        "max_chunk_seconds": MAX_CHUNK_SECONDS,
        "min_chunk_seconds": MIN_CHUNK_SECONDS,
        "min_speech_seconds": MIN_SPEECH_SECONDS,
        "min_silence_seconds": MIN_SILENCE_SECONDS,
    },
    "C": {
        "splitter": "chunking_decode.split_accumulating",
        "prompt_max_tokens": PROMPT_MAX_TOKENS,
        "prompt_carry_join": "space_separated@1",
        "max_chunk_seconds": MAX_CHUNK_SECONDS,
        "min_chunk_seconds": MIN_CHUNK_SECONDS,
        "min_speech_seconds": MIN_SPEECH_SECONDS,
        "min_silence_seconds": MIN_SILENCE_SECONDS,
    },
    "X": {
        "splitter": "split_with_valley_repair@1",
        "slack_seconds": SLACK_SECONDS,
        "valley_prob_threshold": VALLEY_PROB_THRESHOLD,
        "min_valley_frames": MIN_VALLEY_FRAMES,
        "vad_hop_samples": VAD_HOP_SAMPLES,
        "prompt_max_tokens": PROMPT_MAX_TOKENS,
        "prompt_carry_join": "space_separated@1",
        "merge_x": "word_core_ownership@1",
        "max_chunk_seconds": X_MAX_CHUNK_SECONDS,
        "lookback_seconds": X_LOOKBACK_SECONDS,
        "lookahead_seconds": X_LOOKAHEAD_SECONDS,
        "x_slack_seconds": X_SLACK_SECONDS,
        "min_chunk_seconds": MIN_CHUNK_SECONDS,
        "min_speech_seconds": MIN_SPEECH_SECONDS,
        "min_silence_seconds": MIN_SILENCE_SECONDS,
    },
    "W": {
        "merge_w": "word_tile_ownership@1",
        "merge_e_wt": "chunking_decode.merge_overlap_segments",
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "seam_band_seconds": SEAM_BAND_SECONDS,
    },
}


def algorithm_manifest(arm: str) -> dict:
    """The constants and rules one arm's cache must have been written under.

    Deliberately PER ARM: a single manifest over every constant would invalidate
    F's and W's caches whenever a rule only arm C uses is corrected.
    """
    if arm not in ARM_MANIFESTS:
        raise ValueError(f"unknown seam-repair arm {arm!r}")
    return {"experiment": EXPERIMENT, "arm": arm, **ARM_MANIFESTS[arm]}


def algorithm_manifest_sha256(arm: str) -> str:
    payload = json.dumps(algorithm_manifest(arm), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def out_dir() -> Path:
    directory = DA.sc() / "seam-repair-2026-08"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def destination_for(arm: str, pilot: bool = False) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown seam-repair arm {arm!r}")
    suffix = "pilot" if pilot else "eval"
    return out_dir() / f"{suffix}-{arm}.json"


# ============================================================ arm F: valley search


def speech_probabilities(audio: Any, model: Any | None = None) -> Any:
    """Silero frame speech probabilities, one per 512 samples, flattened."""
    import numpy as np

    source = np.asarray(audio)
    if source.ndim != 1:
        raise ValueError("VAD input must be a mono sample array")
    if model is None:
        from faster_whisper.vad import get_vad_model

        model = get_vad_model()
    remainder = source.shape[0] % VAD_HOP_SAMPLES
    padded = np.pad(source, (0, VAD_HOP_SAMPLES - remainder)) if remainder else source
    probs = np.asarray(model(padded), dtype="float64").reshape(-1)
    if not np.all(np.isfinite(probs)):
        raise ValueError("Silero returned a non-finite speech probability")
    return probs


def frame_energies(audio: Any, n_frames: int) -> Any:
    """Per-frame RMS on the same 512-sample grid. Tie-break signal only."""
    import numpy as np

    source = np.asarray(audio, dtype="float64")
    needed = n_frames * VAD_HOP_SAMPLES
    if source.shape[0] < needed:
        source = np.pad(source, (0, needed - source.shape[0]))
    frames = source[:needed].reshape(n_frames, VAD_HOP_SAMPLES)
    return np.sqrt(np.mean(frames * frames, axis=1))


def choose_valley_cut(
    probs: Any,
    energies: Any,
    cursor: float,
    deadline: float,
    min_chunk: float = MIN_CHUNK_SECONDS,
    slack: float = SLACK_SECONDS,
    threshold: float = VALLEY_PROB_THRESHOLD,
    min_frames: int = MIN_VALLEY_FRAMES,
) -> tuple[float, dict] | None:
    """Lowest-mean-probability valley inside the last ``slack`` seconds.

    The frame array is restricted to the slack region FIRST and runs are found
    inside it, so a valley that straddles either slack edge contributes the quiet
    frames it has there instead of being rejected whole.  Energy is a tie-break
    only: the global RMS minimum over the whole legal range can land at 5 s, and
    low energy is not a phoneme boundary -- unvoiced consonants are quiet.
    """
    import numpy as np

    hop = VAD_HOP_SAMPLES / SAMPLE_RATE
    lower = max(cursor + min_chunk, deadline - slack)
    if lower >= deadline:
        return None
    probs = np.asarray(probs).reshape(-1)
    energies = np.asarray(energies).reshape(-1)
    first = int(math.ceil(lower / hop - 1e-9))
    last = int(math.floor(deadline / hop + 1e-9)) - 1
    last = min(last, probs.shape[0] - 1, energies.shape[0] - 1)
    if last < first:
        return None
    quiet = probs[first:last + 1] < threshold

    best: tuple[float, float, float] | None = None
    best_diag: dict | None = None
    run_start: int | None = None
    for offset in range(quiet.shape[0] + 1):
        inside = offset < quiet.shape[0] and bool(quiet[offset])
        if inside and run_start is None:
            run_start = offset
        elif not inside and run_start is not None:
            run_end = offset - 1
            length = run_end - run_start + 1
            if length >= min_frames:
                lo, hi = first + run_start, first + run_end
                mean_prob = float(np.mean(probs[lo:hi + 1]))
                mean_rms = float(np.mean(energies[lo:hi + 1]))
                start_sec, end_sec = lo * hop, (hi + 1) * hop
                key = (round(mean_prob, 6), round(mean_rms, 9), -start_sec)
                if best is None or key < best:
                    best = key
                    best_diag = {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration_sec": end_sec - start_sec,
                        "n_frames": length,
                        "mean_speech_prob": mean_prob,
                        "mean_rms": mean_rms,
                    }
            run_start = None
    if best_diag is None:
        return None
    midpoint = (best_diag["start_sec"] + best_diag["end_sec"]) / 2.0
    cut = min(max(midpoint, cursor + min_chunk), deadline)
    best_diag["cut_sec"] = cut
    best_diag["clamped"] = abs(cut - midpoint) > 1e-9
    return cut, best_diag


@dataclass(frozen=True)
class RPiece:
    """A decode piece; ``end_kind`` describes the boundary at ``end``."""

    start: float
    end: float
    end_kind: str
    detail: dict | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def split_with_valley_repair(
    duration: float,
    silences: Iterable[Any],
    probs: Any | None = None,
    energies: Any | None = None,
    max_chunk: float = MAX_CHUNK_SECONDS,
    min_chunk: float = MIN_CHUNK_SECONDS,
    min_speech: float = MIN_SPEECH_SECONDS,
    min_silence: float = MIN_SILENCE_SECONDS,
    slack: float = SLACK_SECONDS,
) -> tuple[list[RPiece], dict]:
    """Arm P's ``split_accumulating`` with the forced-cut branch repaired.

    The non-forced path is P's, character for character, up to the first repaired
    boundary; after that the cursor has moved and every later deadline differs.
    ``min_speech`` is a PREFERENCE here exactly as in P: if no candidate clears it,
    the latest in-range silence is still taken.

    With ``probs is None`` this reduces to P's behaviour and emits ``blind``.
    """
    if duration < 0:
        raise ValueError("duration must be non-negative")
    quiet = sorted((CD._as_silence(x) for x in silences), key=lambda s: s.start)
    cuts = [((s.start + s.end) / 2.0, s) for s in quiet if s.duration >= min_silence]

    pieces: list[RPiece] = []
    cursor = 0.0
    while duration - cursor > max_chunk + 1e-7:
        deadline = cursor + max_chunk
        inrange = [(c, s) for c, s in cuts if cursor + min_chunk <= c <= deadline]
        ok = [(c, s) for c, s in inrange
              if CD._voiced_between(cursor, c, quiet) >= min_speech]
        pick = (max(ok, key=lambda t: t[0]) if ok
                else max(inrange, key=lambda t: t[0]) if inrange else None)
        if pick is None:
            repaired = (
                choose_valley_cut(probs, energies, cursor, deadline, min_chunk,
                                  slack=slack)
                if probs is not None else None
            )
            if repaired is None:
                pieces.append(RPiece(cursor, deadline, BLIND, None))
                cursor = deadline
                continue
            cut, diag = repaired
            pieces.append(RPiece(cursor, cut, PROBABILITY_VALLEY, diag))
            cursor = cut
            continue
        cut, sil = pick
        pieces.append(RPiece(cursor, cut, VAD_SILENCE,
                             {"silence_duration_sec": sil.duration}))
        cursor = cut
    if duration - cursor > 1e-7 or not pieces:
        pieces.append(RPiece(cursor, duration, AUDIO_END, None))
    else:
        pieces[-1] = RPiece(pieces[-1].start, pieces[-1].end, AUDIO_END,
                            pieces[-1].detail)

    # short-tail repair, inherited from P; provenance is recomputed here because
    # this step can delete a boundary or invent a midpoint of its own.
    if len(pieces) > 1 and pieces[-1].duration < min_chunk:
        prev, last = pieces[-2], pieces[-1]
        if (last.end - prev.start) <= max_chunk:
            pieces[-2] = RPiece(prev.start, last.end, AUDIO_END, None)
            pieces.pop()
        else:
            lo = max(prev.start + min_chunk, last.end - max_chunk)
            hi = min(prev.start + max_chunk, last.end - min_chunk)
            if lo <= hi:
                inside = [c for c, _ in cuts if lo <= c <= hi]
                if inside:
                    boundary = min(inside, key=lambda c: abs(c - (lo + hi) / 2.0))
                    kind = VAD_SILENCE
                else:
                    boundary = (lo + hi) / 2.0
                    kind = BLIND
                pieces[-2] = RPiece(prev.start, boundary, kind, None)
                pieces[-1] = RPiece(boundary, last.end, AUDIO_END, None)

    for a, b in zip(pieces, pieces[1:]):
        if abs(a.end - b.start) > 1e-6:
            raise AssertionError("pieces overlap or leave a gap")
    if pieces[-1].end_kind != AUDIO_END:
        raise AssertionError("the final piece must end at the audio end")

    kinds = [p.end_kind for p in pieces[:-1]]
    counters = {
        "n_pieces": len(pieces),
        "vad_silence_cuts": kinds.count(VAD_SILENCE),
        "valley_cuts": kinds.count(PROBABILITY_VALLEY),
        "blind_cuts": kinds.count(BLIND),
        "tiny_chunks": (
            0 if (len(pieces) == 1 and duration < min_chunk)
            else sum(1 for p in pieces if p.duration < min_chunk - 1e-7)
        ),
        "overlong_chunks": sum(1 for p in pieces if p.duration > max_chunk + 1e-7),
        "durations": [round(p.duration, 3) for p in pieces],
    }
    if counters["vad_silence_cuts"] + counters["valley_cuts"] + counters["blind_cuts"] \
            != len(pieces) - 1:
        raise AssertionError("boundary kinds do not account for every boundary")
    return pieces, counters


def sample_boundaries(pieces: Sequence[RPiece], n_samples: int) -> list[tuple[int, int]]:
    """Integer sample slices that tile ``audio[0:n_samples]`` exactly once."""
    if not pieces:
        raise ValueError("no pieces to place on the sample grid")
    edges = [0]
    for piece in pieces[:-1]:
        edge = int(round(piece.end * SAMPLE_RATE))
        edge = min(max(edge, edges[-1] + 1), n_samples - 1)
        edges.append(edge)
    edges.append(n_samples)
    slices = list(zip(edges, edges[1:]))
    covered = 0
    for begin, end in slices:
        if end <= begin:
            raise AssertionError("a piece has no samples")
        covered += end - begin
    if slices[0][0] != 0 or slices[-1][1] != n_samples or covered != n_samples:
        raise AssertionError("piece slices do not tile the audio exactly once")
    return slices


# ============================================================ arm C: token prompts


def prompt_ids_for(tokenizer: Any, text: str, max_tokens: int = PROMPT_MAX_TOKENS) -> list[int]:
    """Longest whole-word suffix of ``text`` fitting in ``max_tokens`` model tokens.

    The cap is in TOKENS of the model's own tokenizer, not characters -- Greek
    characters cut words in half -- and truncation never begins mid-word.  Token
    ids are returned so faster-whisper does not strip, space-prefix and re-encode
    a string on our behalf.
    """
    words = str(text).split()
    if not words:
        return []
    best: list[int] = []
    for take in range(1, len(words) + 1):
        candidate = " " + " ".join(words[-take:])
        ids = list(tokenizer.encode(candidate, add_special_tokens=False).ids)
        if len(ids) > max_tokens:
            break
        best = ids
    return best


@contextmanager
def prompt_probe(model: Any):
    """Record every ``get_prompt`` call the decoder makes. Pilot instrumentation."""
    calls: list[dict] = []
    original = model.get_prompt

    def spy(tokenizer, previous_tokens, *args, **kwargs):
        result = original(tokenizer, previous_tokens, *args, **kwargs)
        calls.append({
            "previous_tokens": list(previous_tokens),
            "prompt": list(result),
            "sot_prev": tokenizer.sot_prev,
            "sot_sequence": list(tokenizer.sot_sequence),
        })
        return result

    model.get_prompt = spy
    try:
        yield calls
    finally:
        model.get_prompt = original


# ============================================================ arm W: word ownership


def owning_window_index(midpoint: float, n_windows: int,
                        stride: float = STRIDE_SECONDS) -> int:
    """Tile ownership under the frozen convention [0,15], (15,30], (30,45], ...

    The closed lower edge keeps arm E's rule that a tie goes to the earlier window.
    """
    index = max(0, math.ceil(midpoint / stride - 1e-9) - 1)
    return min(index, n_windows - 1)


def word_candidates(window: Any, segments: Iterable[dict]) -> list[dict]:
    """Shift one window's word timestamps into audio time, with stable ids."""
    candidates: list[dict] = []
    for segment_index, segment in enumerate(segments):
        words = segment.get("words") or []
        for word_index, word in enumerate(words):
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                raise ValueError("arm W requires a timestamp on every word")
            start = float(start) + window.start
            end = float(end) + window.start
            if end < start:
                raise ValueError("word timestamp end precedes start")
            candidates.append({
                "text": str(word.get("word", "")),
                "start": start,
                "end": end,
                "midpoint": (start + end) / 2.0,
                "window_index": window.index,
                "segment_index": segment_index,
                "word_index": word_index,
            })
    return candidates


def merge_words_by_tile(
    candidates: Iterable[dict],
    windows: Sequence[Any],
    duration: float,
) -> dict:
    """Keep a word iff its own window owns the tile its midpoint falls in.

    Exactly one owner per CANDIDATE by construction.  That is not the same as
    exactly one owner per spoken WORD: two decodes of the same boundary word can
    carry different timestamps and land on opposite sides of a tile edge, giving a
    duplicate or a hole at that word.  ``seam_disagreement`` reports that; it is
    not suppressed.
    """
    kept: list[dict] = []
    non_owner = 0
    outside_audio = 0
    for candidate in candidates:
        midpoint = float(candidate["midpoint"])
        if midpoint < -1e-7 or midpoint > duration + 1e-7:
            outside_audio += 1
            continue
        owner = owning_window_index(min(max(midpoint, 0.0), duration), len(windows))
        if owner != candidate["window_index"]:
            non_owner += 1
            continue
        kept.append(candidate)
    kept.sort(key=lambda c: (round(c["start"], 6), c["window_index"],
                             c["segment_index"], c["word_index"]))
    return {
        "text": "".join(c["text"] for c in kept).strip(),
        "words": [{"text": c["text"], "start": c["start"], "end": c["end"]} for c in kept],
        "kept": len(kept),
        "non_owner": non_owner,
        "outside_audio": outside_audio,
    }


def seam_disagreement(
    candidates: Iterable[dict],
    windows: Sequence[Any],
    duration: float,
    band: float = SEAM_BAND_SECONDS,
) -> dict:
    """How often the two decodes of a seam disagree about the words there.

    At each tile edge, take the raw candidates of the two adjacent windows whose
    midpoints fall within ``band`` seconds of the edge, normalize with the frozen
    ``ftoks``, and compare the two multisets.  Candidate conservation alone cannot
    see a hole or a duplicate; this can.
    """
    from collections import Counter

    by_window: dict[int, list[dict]] = {}
    for candidate in candidates:
        by_window.setdefault(candidate["window_index"], []).append(candidate)

    seams: list[dict] = []
    agreed = disagreed = 0
    for left in range(len(windows) - 1):
        edge = windows[left].kept_end
        if edge >= duration - 1e-7:
            continue
        def band_tokens(index: int) -> Counter:
            tokens: Counter = Counter()
            for candidate in by_window.get(index, []):
                if abs(candidate["midpoint"] - edge) <= band:
                    tokens.update(ftoks(candidate["text"]))
            return tokens
        a, b = band_tokens(left), band_tokens(left + 1)
        common = sum((a & b).values())
        only_a = sum((a - b).values())
        only_b = sum((b - a).values())
        agreed += common
        disagreed += only_a + only_b
        seams.append({
            "edge_sec": round(edge, 3),
            "tokens_left": sum(a.values()),
            "tokens_right": sum(b.values()),
            "agreed": common,
            "only_left": only_a,
            "only_right": only_b,
        })
    total = agreed + disagreed
    return {
        "seams": seams,
        "tokens_agreed": agreed,
        "tokens_disagreed": disagreed,
        "disagreement_rate": (disagreed / total) if total else 0.0,
    }


# ============================================================ decode drivers


def _segments_with_words(segments: Sequence[Any]) -> list[dict]:
    records: list[dict] = []
    for segment in segments:
        words = CD._segment_value(segment, "words", None) or []
        records.append({
            "text": str(CD._segment_value(segment, "text", "")),
            "start": CD._segment_value(segment, "start"),
            "end": CD._segment_value(segment, "end"),
            "words": [
                {
                    "word": str(CD._segment_value(w, "word", "")),
                    "start": CD._segment_value(w, "start"),
                    "end": CD._segment_value(w, "end"),
                    "probability": CD._segment_value(w, "probability"),
                }
                for w in words
            ],
        })
    return records


def _decode(model: Any, audio: Any, config: dict, **extra) -> list[dict]:
    segments, _ = model.transcribe(audio, **config, **extra)
    return _segments_with_words(list(segments))


def _piece_text(records: Sequence[dict]) -> str:
    return "".join(record["text"] for record in records)


def _window_record_f(model, wav, row, audio_loader=CD._load_audio,
                     silence_detector=CD.detect_silences, probe=None) -> dict:
    import ctranslate2

    started = time.monotonic()
    audio = audio_loader(wav)
    duration = CD._audio_duration(audio, float(row["duration_sec"]))
    silences, detection_source, detection_error = silence_detector(audio, duration, model)
    probs = speech_probabilities(audio)
    energies = frame_energies(audio, probs.shape[0])
    pieces, counters = split_with_valley_repair(duration, silences, probs, energies)
    slices = sample_boundaries(pieces, len(audio))

    ctranslate2.set_random_seed(DA.seed_for("A", row["window_id"]))
    piece_results = []
    for piece, (begin, end) in zip(pieces, slices):
        records = _decode(model, audio[begin:end], CONTROL)
        piece_results.append({
            "start_sec": piece.start, "end_sec": piece.end,
            "text": _piece_text(records), "segments": records,
        })
    combined = CD.combine_piece_transcripts(piece_results)
    return {
        "text": combined["text"],
        "segments": combined["segments"],
        "n_segments": len(combined["segments"]),
        "audio_seconds": round(duration, 2),
        "geometry_seconds": duration,
        "wall_seconds": round(time.monotonic() - started, 1),
        "seed": DA.seed_for("A", row["window_id"]),
        "config": CONTROL,
        "detection_source": detection_source,
        "detection_error": detection_error,
        "n_pieces": len(pieces),
        "counters": counters,
        "pieces": [
            {
                "start_sec": round(p.start, 3), "end_sec": round(p.end, 3),
                "duration_sec": round(p.duration, 3),
                "sample_begin": b, "sample_end": e,
                "end_kind": p.end_kind, "detail": p.detail,
            }
            for p, (b, e) in zip(pieces, slices)
        ],
    }


def _window_record_c(model, wav, row, audio_loader=CD._load_audio,
                     silence_detector=CD.detect_silences, probe=None) -> dict:
    """Arm P's segmentation exactly; only the per-piece ``initial_prompt`` is new."""
    import ctranslate2

    started = time.monotonic()
    audio = audio_loader(wav)
    duration = CD._audio_duration(audio, float(row["duration_sec"]))
    silences, detection_source, detection_error = silence_detector(audio, duration, model)
    pieces, split_counters = CD.split_accumulating(duration, silences)
    tokenizer = model.hf_tokenizer

    ctranslate2.set_random_seed(DA.seed_for("A", row["window_id"]))
    piece_results, prompts = [], []
    carried = ""
    for index, piece in enumerate(pieces):
        begin = int(round(piece.start * SAMPLE_RATE))
        end = int(round(piece.end * SAMPLE_RATE))
        ids = prompt_ids_for(tokenizer, carried) if index else []
        prompts.append({
            "piece_index": index,
            "n_prompt_tokens": len(ids),
            "prompt_token_ids": ids,
            "preceding_text_empty": not carried.strip(),
        })
        extra = {"initial_prompt": ids} if ids else {}
        call_start = len(probe) if probe is not None else None
        records = _decode(model, audio[begin:end], CONTROL, **extra)
        if probe is not None:
            prompts[-1]["get_prompt_call_start"] = call_start
            prompts[-1]["get_prompt_call_end"] = len(probe)
        text = _piece_text(records)
        piece_results.append({
            "start_sec": piece.start, "end_sec": piece.end,
            "text": text, "segments": records,
        })
        # a separator, so the last word of piece k-1 and the first of piece k
        # cannot fuse into one "word" when prompt_ids_for splits on whitespace
        carried = " ".join(part for part in (carried.strip(), text.strip()) if part)
    combined = CD.combine_piece_transcripts(piece_results)
    return {
        "text": combined["text"],
        "segments": combined["segments"],
        "n_segments": len(combined["segments"]),
        "audio_seconds": round(duration, 2),
        "geometry_seconds": duration,
        "wall_seconds": round(time.monotonic() - started, 1),
        "seed": DA.seed_for("A", row["window_id"]),
        "config": CONTROL,
        "detection_source": detection_source,
        "detection_error": detection_error,
        "n_pieces": len(pieces),
        "counters": split_counters,
        "prompts": prompts,
        "pieces": [
            {"start_sec": round(p.start, 3), "end_sec": round(p.end, 3),
             "duration_sec": round(p.duration, 3)}
            for p in pieces
        ],
    }


def _window_record_w(model, wav, row, audio_loader=CD._load_audio,
                     silence_detector=None, probe=None) -> dict:
    """Overlapping windows with word timestamps; caches raw decodes for E-WT."""
    import ctranslate2

    started = time.monotonic()
    audio = audio_loader(wav)
    duration = CD._audio_duration(audio, float(row["duration_sec"]))
    windows = CD.overlap_layout(duration)
    config = dict(CONTROL, word_timestamps=True)

    ctranslate2.set_random_seed(DA.seed_for("A", row["window_id"]))
    raw_windows, candidates = [], []
    for window in windows:
        chunk = CD._zero_padded_window(audio, window.start, window.end)
        records = _decode(model, chunk, config)
        raw_windows.append({
            "index": window.index,
            "start_sec": window.start, "end_sec": window.end,
            "centre_sec": window.centre,
            "kept_start_sec": window.kept_start, "kept_end_sec": window.kept_end,
            "segments": records,
        })
        candidates.extend(word_candidates(window, records))

    merged = merge_words_by_tile(candidates, windows, duration)
    disagreement = seam_disagreement(candidates, windows, duration)
    return {
        "text": merged["text"],
        "words": merged["words"],
        "n_words": len(merged["words"]),
        "audio_seconds": round(duration, 2),
        "geometry_seconds": duration,
        "wall_seconds": round(time.monotonic() - started, 1),
        "seed": DA.seed_for("A", row["window_id"]),
        "config": config,
        "n_windows": len(windows),
        "stride_sec": STRIDE_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "n_candidates": len(candidates),
        "kept": merged["kept"],
        "non_owner": merged["non_owner"],
        "outside_audio": merged["outside_audio"],
        "seam_disagreement": disagreement,
        "raw_windows": raw_windows,
    }


def owning_core_index(midpoint: float, cores: Sequence[tuple[float, float]]) -> int | None:
    """Half-open core ownership [s, e); the last core is closed at the audio end.

    The cores tile the window exactly, so a midpoint inside the audio has exactly
    one owner and a word can never be kept twice.
    """
    for index, (start, end) in enumerate(cores):
        last = index == len(cores) - 1
        if midpoint < start - 1e-7:
            continue
        if midpoint < end - 1e-7 or (last and midpoint <= end + 1e-7):
            return index
    return None


def merge_words_by_core(candidates: Iterable[dict],
                        cores: Sequence[tuple[float, float]]) -> dict:
    """Arm W's rule applied to arm F's pieces: one owner per word, by its own time."""
    kept, non_owner, outside_audio = [], 0, 0
    for candidate in candidates:
        owner = owning_core_index(float(candidate["midpoint"]), cores)
        if owner is None:
            outside_audio += 1
            continue
        if owner != candidate["piece_index"]:
            non_owner += 1
            continue
        kept.append(candidate)
    kept.sort(key=lambda c: (round(c["start"], 6), c["piece_index"],
                             c["segment_index"], c["word_index"]))
    return {
        "text": "".join(c["text"] for c in kept).strip(),
        "words": [{"text": c["text"], "start": c["start"], "end": c["end"]} for c in kept],
        "kept": len(kept), "non_owner": non_owner, "outside_audio": outside_audio,
    }


def _window_record_x(model, wav, row, audio_loader=CD._load_audio,
                     silence_detector=CD.detect_silences, probe=None) -> dict:
    """All three mechanisms at once: F's cut, C's prompt, W's word ownership."""
    import ctranslate2

    started = time.monotonic()
    audio = audio_loader(wav)
    duration = CD._audio_duration(audio, float(row["duration_sec"]))
    silences, detection_source, detection_error = silence_detector(audio, duration, model)
    probs = speech_probabilities(audio)
    energies = frame_energies(audio, probs.shape[0])
    pieces, counters = split_with_valley_repair(
        duration, silences, probs, energies, max_chunk=X_MAX_CHUNK_SECONDS,
        slack=X_SLACK_SECONDS)
    slices = sample_boundaries(pieces, len(audio))
    cores = [(p.start, p.end) for p in pieces]
    config = dict(CONTROL, word_timestamps=True)
    tokenizer = model.hf_tokenizer

    ctranslate2.set_random_seed(DA.seed_for("A", row["window_id"]))
    candidates, prompts, raw_pieces = [], [], []
    carried = ""
    for index, piece in enumerate(pieces):
        # the decoder sees the core PLUS a margin on each side; ownership still
        # belongs to the core, so the margin is context and never double-counted
        begin = max(0, int(round((piece.start - X_LOOKBACK_SECONDS) * SAMPLE_RATE)))
        end = min(len(audio), int(round((piece.end + X_LOOKAHEAD_SECONDS) * SAMPLE_RATE)))
        offset = begin / SAMPLE_RATE
        ids = prompt_ids_for(tokenizer, carried) if index else []
        prompts.append({"piece_index": index, "n_prompt_tokens": len(ids),
                        "prompt_token_ids": ids,
                        "preceding_text_empty": not carried.strip()})
        extra = {"initial_prompt": ids} if ids else {}
        call_start = len(probe) if probe is not None else None
        records = _decode(model, audio[begin:end], config, **extra)
        if probe is not None:
            prompts[-1]["get_prompt_call_start"] = call_start
            prompts[-1]["get_prompt_call_end"] = len(probe)
        raw_pieces.append({"piece_index": index, "input_start_sec": offset,
                           "input_end_sec": end / SAMPLE_RATE,
                           "core_start_sec": piece.start, "core_end_sec": piece.end,
                           "segments": records})
        for segment_index, segment in enumerate(records):
            for word_index, word in enumerate(segment.get("words") or []):
                if word.get("start") is None or word.get("end") is None:
                    raise ValueError("arm X requires a timestamp on every word")
                w_start = float(word["start"]) + offset
                w_end = float(word["end"]) + offset
                candidates.append({
                    "text": str(word.get("word", "")),
                    "start": w_start, "end": w_end,
                    "midpoint": (w_start + w_end) / 2.0,
                    "piece_index": index, "segment_index": segment_index,
                    "word_index": word_index,
                })
        # the prompt carries the CORE text only, never the margin's duplicate words
        core_words = [c for c in candidates
                      if c["piece_index"] == index
                      and owning_core_index(c["midpoint"], cores) == index]
        core_text = "".join(c["text"] for c in core_words)
        carried = " ".join(part for part in (carried.strip(), core_text.strip()) if part)

    merged = merge_words_by_core(candidates, cores)
    return {
        "text": merged["text"],
        "words": merged["words"],
        "n_words": len(merged["words"]),
        "audio_seconds": round(duration, 2),
        "geometry_seconds": duration,
        "wall_seconds": round(time.monotonic() - started, 1),
        "seed": DA.seed_for("A", row["window_id"]),
        "config": config,
        "detection_source": detection_source,
        "detection_error": detection_error,
        "n_pieces": len(pieces),
        "counters": counters,
        "lookback_sec": X_LOOKBACK_SECONDS,
        "lookahead_sec": X_LOOKAHEAD_SECONDS,
        "n_candidates": len(candidates),
        "kept": merged["kept"], "non_owner": merged["non_owner"],
        "outside_audio": merged["outside_audio"],
        "prompts": prompts,
        "pieces": [
            {"start_sec": round(p.start, 3), "end_sec": round(p.end, 3),
             "duration_sec": round(p.duration, 3),
             "sample_begin": b, "sample_end": e,
             "end_kind": p.end_kind, "detail": p.detail}
            for p, (b, e) in zip(pieces, slices)
        ],
        "raw_pieces": raw_pieces,
    }


RECORDERS: dict[str, Callable[..., dict]] = {
    "F": _window_record_f, "C": _window_record_c, "W": _window_record_w,
    "X": _window_record_x,
}


def _geometry_seconds(record: dict) -> float:
    """The duration the layout was actually built from, unrounded.

    `audio_seconds` is rounded to 2 dp for reading. Rebuilding `overlap_layout` from the
    rounded value moves the final tile boundary, which changes ownership at the audio end
    and makes a derived arm disagree with the decode it was derived from. Records written
    before this key existed carry only the rounded value, so it stays the fallback.
    """
    value = record.get("geometry_seconds")
    return float(record["audio_seconds"] if value is None else value)


def derive_e_wt(record: dict) -> dict:
    """Re-merge a cached W record with the OLD segment-level rule. No decoding."""
    duration = _geometry_seconds(record)
    windows = CD.overlap_layout(duration)
    if len(windows) != len(record["raw_windows"]):
        raise AssertionError("cached W layout does not match the frozen layout")
    candidates: list[dict] = []
    for window, raw in zip(windows, record["raw_windows"]):
        if raw["index"] != window.index:
            raise AssertionError("cached W window order is not the frozen order")
        candidates.extend(
            CD._window_token_candidates(window, raw["segments"], duration)
        )
    merged = CD.merge_overlap_segments(candidates, windows, duration)
    return {
        "text": merged["text"],
        "segments": merged["segments"],
        "n_segments": len(merged["segments"]),
        "audio_seconds": record["audio_seconds"],
        "geometry_seconds": duration,
        "config": record["config"],
        "derived_from": "W",
        "tokens_dropped_as_duplicates_at_seams":
            merged["tokens_dropped_as_duplicates_at_seams"],
    }


def _fresh_state(arm: str, model_sha: str, env: dict, which: str) -> dict:
    config = (dict(CONTROL, word_timestamps=True)
              if arm in ("W", "X") else dict(CONTROL))
    return {
        "experiment": EXPERIMENT,
        "arm": arm,
        "artifact_id": MODEL_ARTIFACT_ID,
        "model": str(MODEL_DIR),
        "model_sha256_16": model_sha,
        "config": config,
        "environment": env,
        "algorithm_manifest": algorithm_manifest(arm),
        "algorithm_manifest_sha256": algorithm_manifest_sha256(arm),
        "set": which,
        "windows": {},
    }


def validate_cache_identity(existing: dict, fresh: dict, path: Any = "cache") -> None:
    """Refuse a cache written under a different model, config, env OR algorithm."""
    CD.validate_cache_identity(existing, fresh, path)
    for key in ("arm", "set", "algorithm_manifest_sha256"):
        if existing.get(key) != fresh.get(key):
            raise ValueError(
                f"{path} was written under a different {key} "
                f"({existing.get(key)} != {fresh.get(key)}); "
                "delete it rather than extending it"
            )


def eval_rows() -> list[dict]:
    return CD.eval_rows()


def pilot_rows() -> list[dict]:
    """The five frozen pilot windows, verified against the cached arm P decode."""
    by_id = {row["window_id"]: row for row in eval_rows()}
    rows = []
    for window_id in PILOT_WINDOW_IDS:
        if window_id not in by_id:
            raise ValueError(f"pilot window {window_id} is not in the frozen 39")
        rows.append(by_id[window_id])
    return rows


def p_forced_cuts() -> dict[str, int]:
    path = DA.sc() / CD.SEGMENT_CACHE_DIRNAME / "eval-P.json"
    records = CD._cached_window_records(path)
    return {wid: int(r.get("whole_fallback_stretches", 0)) for wid, r in records.items()}


def decode(
    arm: str,
    pilot: bool = False,
    limit: int | None = None,
    model: Any | None = None,
    destination: Path | None = None,
    rows_override: list[dict] | None = None,
    audio_loader: Callable[[Path], Any] = CD._load_audio,
    silence_detector: Callable[..., Any] = CD.detect_silences,
) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown seam-repair arm {arm!r}")
    model_sha = CD.verify_model() if model is None else MODEL_SHA256_16
    env = CD.environment()
    which = "pilot" if pilot else "eval"
    fresh = _fresh_state(arm, model_sha, env, which)
    destination = destination or destination_for(arm, pilot)
    state = json.loads(destination.read_text()) if destination.exists() else fresh
    validate_cache_identity(state, fresh, destination)
    state.setdefault("windows", {})
    if rows_override is not None:
        rows = list(rows_override)
    else:
        rows = pilot_rows() if pilot else eval_rows()
    if limit is not None:
        rows = rows[:limit]
    todo = [row for row in rows if row["window_id"] not in state["windows"]]
    if not todo:
        return destination
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(str(MODEL_DIR), device=DEVICE, compute_type=COMPUTE,
                             cpu_threads=THREADS)
    recorder = RECORDERS[arm]
    for row in todo:
        wav = DA.sc() / "bench_windows" / f"{row['window_id']}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"missing audio for {row['window_id']}: {wav}")
        if arm in ("C", "X") and pilot:
            with prompt_probe(model) as calls:
                record = recorder(model, wav, row, audio_loader,
                                  silence_detector, calls)
            record["get_prompt_calls"] = calls
        else:
            record = recorder(model, wav, row, audio_loader, silence_detector)
        state["windows"][row["window_id"]] = record
        CD._write_state(destination, state)   # a multi-hour CPU pass must resume
    return destination


# ============================================================ pilot: MECHANISM only


def _f_verdict(state: dict, forced: dict[str, int]) -> dict:
    failures: list[str] = []
    blind = valley = vad = 0
    p_forced = 0
    for window_id, record in state["windows"].items():
        counters = record["counters"]
        blind += counters["blind_cuts"]
        valley += counters["valley_cuts"]
        vad += counters["vad_silence_cuts"]
        p_forced += forced.get(window_id, 0)
        if counters["tiny_chunks"]:
            failures.append(f"{window_id}: {counters['tiny_chunks']} pieces under 5 s")
        if counters["overlong_chunks"]:
            failures.append(f"{window_id}: {counters['overlong_chunks']} pieces over 29.5 s")
        pieces = record["pieces"]
        if pieces[0]["sample_begin"] != 0:
            failures.append(f"{window_id}: first piece does not start at sample 0")
        for a, b in zip(pieces, pieces[1:]):
            if a["sample_end"] != b["sample_begin"]:
                failures.append(f"{window_id}: sample slices do not tile exactly")
                break
        for piece in pieces:
            if piece["end_kind"] != PROBABILITY_VALLEY:
                continue
            detail = piece["detail"] or {}
            if detail.get("duration_sec", 0.0) < 0.1 - 1e-9:
                failures.append(f"{window_id}: repaired cut valley shorter than 100 ms")
            if detail.get("mean_speech_prob", 1.0) >= VALLEY_PROB_THRESHOLD:
                failures.append(f"{window_id}: repaired cut valley is not below 0.35")
    if not (blind < p_forced):
        failures.append(
            f"mechanism never fired: {blind} blind cuts vs {p_forced} forced cuts in P"
        )
    return {
        "arm": "F",
        "blind_cuts": blind,
        "valley_cuts": valley,
        "vad_silence_cuts": vad,
        "p_forced_cuts_same_windows": p_forced,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }


def _c_verdict(state: dict) -> dict:
    failures: list[str] = []
    checked = 0
    for window_id, record in state["windows"].items():
        calls = record.get("get_prompt_calls")
        if calls is None:
            failures.append(f"{window_id}: no get_prompt instrumentation recorded")
            continue
        for prompt in record["prompts"]:
            index = prompt["piece_index"]
            ids = prompt["prompt_token_ids"]
            if len(ids) > PROMPT_MAX_TOKENS:
                failures.append(f"{window_id}/{index}: prompt exceeds 64 tokens")
            if index and not ids and not prompt["preceding_text_empty"]:
                failures.append(
                    f"{window_id}/{index}: empty prompt with non-empty preceding text"
                )
            begin = prompt.get("get_prompt_call_start")
            end = prompt.get("get_prompt_call_end")
            if begin is None or end is None or end <= begin:
                failures.append(f"{window_id}/{index}: decoder made no get_prompt call")
                continue
            window_calls = calls[begin:end]
            checked += 1
            expected = ids if index else []
            if window_calls[0]["previous_tokens"] != expected:
                failures.append(
                    f"{window_id}/{index}: first get_prompt saw "
                    f"{len(window_calls[0]['previous_tokens'])} tokens, expected "
                    f"{len(expected)}"
                )
            for later in window_calls[1:]:
                if later["previous_tokens"] != []:
                    failures.append(
                        f"{window_id}/{index}: a later get_prompt call was not reset"
                    )
                    break
            if expected:
                prompt_tokens = window_calls[0]["prompt"]
                sot_prev = window_calls[0]["sot_prev"]
                sot_sequence = window_calls[0]["sot_sequence"]
                if not prompt_tokens or prompt_tokens[0] != sot_prev:
                    failures.append(f"{window_id}/{index}: prompt does not start with sot_prev")
                if prompt_tokens[1:1 + len(expected)] != expected:
                    failures.append(f"{window_id}/{index}: prompt ids are not the recorded ids")
                if prompt_tokens[1 + len(expected):1 + len(expected) + len(sot_sequence)] \
                        != sot_sequence:
                    failures.append(f"{window_id}/{index}: sot_sequence does not follow the prompt")
    return {
        "arm": "C",
        "pieces_checked": checked,
        "failures": failures[:20],
        "n_failures": len(failures),
        "verdict": "PASS" if not failures else "FAIL",
    }


def _w_verdict(state: dict) -> dict:
    failures: list[str] = []
    kept = non_owner = outside = candidates = 0
    disagreement = []
    for window_id, record in state["windows"].items():
        duration = _geometry_seconds(record)
        windows = CD.overlap_layout(duration)
        cached_windows = record["raw_windows"]
        if len(windows) != len(cached_windows):
            failures.append(f"{window_id}: cached layout has {len(cached_windows)} "
                            f"windows, the frozen layout has {len(windows)}")
            continue
        if [w.index for w in windows] != [c["index"] for c in cached_windows]:
            failures.append(f"{window_id}: cached window order is not the frozen order")
            continue
        raw: list[dict] = []
        for window, cached in zip(windows, cached_windows):
            raw.extend(word_candidates(window, cached["segments"]))
        recomputed = merge_words_by_tile(raw, windows, duration)
        if recomputed["kept"] != record["kept"]:
            failures.append(f"{window_id}: cached merge is not reproducible")
        total = recomputed["kept"] + recomputed["non_owner"] + recomputed["outside_audio"]
        if total != len(raw):
            failures.append(f"{window_id}: candidates are not conserved")
        if recomputed["kept"] == 0:
            failures.append(f"{window_id}: the merge kept no words at all")
        # two-way oracle: kept IFF this window owns the tile of the word's midpoint
        expected_kept = 0
        for candidate in raw:
            mid = candidate["midpoint"]
            if mid < -1e-7 or mid > duration + 1e-7:
                continue
            owner = owning_window_index(min(max(mid, 0.0), duration), len(windows))
            if owner == candidate["window_index"]:
                expected_kept += 1
        if expected_kept != recomputed["kept"]:
            failures.append(f"{window_id}: ownership is not an if-and-only-if")
        seen = set()
        for candidate in raw:
            key = (candidate["window_index"], candidate["segment_index"],
                   candidate["word_index"])
            if key in seen:
                failures.append(f"{window_id}: duplicate candidate id {key}")
                break
            seen.add(key)
        kept += recomputed["kept"]
        non_owner += recomputed["non_owner"]
        outside += recomputed["outside_audio"]
        candidates += len(raw)
        disagreement.append(record["seam_disagreement"]["disagreement_rate"])
    return {
        "arm": "W",
        "candidates": candidates,
        "kept": kept,
        "non_owner": non_owner,
        "outside_audio": outside,
        "mean_seam_disagreement_rate": (
            round(sum(disagreement) / len(disagreement), 4) if disagreement else None
        ),
        "failures": failures[:20],
        "n_failures": len(failures),
        "verdict": "PASS" if not failures else "FAIL",
    }


def _x_verdict(state: dict, forced: dict[str, int]) -> dict:
    """All three mechanisms must hold at once, or the combination is not what it says."""
    failures: list[str] = []
    blind = valley = 0
    p_forced = kept = non_owner = outside = candidates = context_words = 0
    for window_id, record in state["windows"].items():
        counters = record["counters"]
        blind += counters["blind_cuts"]
        valley += counters["valley_cuts"]
        p_forced += forced.get(window_id, 0)
        # --- F: legal geometry, exact integer tiling of the CORE pieces
        if counters["tiny_chunks"] or counters["overlong_chunks"]:
            failures.append(f"{window_id}: illegal core piece length")
        pieces = record["pieces"]
        if pieces[0]["sample_begin"] != 0:
            failures.append(f"{window_id}: cores do not start at sample 0")
        for a, b in zip(pieces, pieces[1:]):
            if a["sample_end"] != b["sample_begin"]:
                failures.append(f"{window_id}: cores do not tile exactly")
                break
        for piece in pieces:
            if piece["end_kind"] != PROBABILITY_VALLEY:
                continue
            detail = piece["detail"] or {}
            if detail.get("duration_sec", 0.0) < 0.1 - 1e-9 or \
                    detail.get("mean_speech_prob", 1.0) >= VALLEY_PROB_THRESHOLD:
                failures.append(f"{window_id}: a repaired cut is not a real valley")
        # --- C: the prompt reached the decoder on every piece after the first
        calls = record.get("get_prompt_calls")
        if calls is None:
            failures.append(f"{window_id}: no get_prompt instrumentation")
        else:
            for prompt in record["prompts"]:
                index, ids = prompt["piece_index"], prompt["prompt_token_ids"]
                if len(ids) > PROMPT_MAX_TOKENS:
                    failures.append(f"{window_id}/{index}: prompt over 64 tokens")
                if index and not ids and not prompt["preceding_text_empty"]:
                    failures.append(f"{window_id}/{index}: empty prompt, non-empty text")
                begin, end = prompt.get("get_prompt_call_start"), prompt.get("get_prompt_call_end")
                if begin is None or end is None or end <= begin:
                    failures.append(f"{window_id}/{index}: no get_prompt call")
                    continue
                window_calls = calls[begin:end]
                if window_calls[0]["previous_tokens"] != (ids if index else []):
                    failures.append(f"{window_id}/{index}: prompt did not reach the decoder")
                if any(c["previous_tokens"] != [] for c in window_calls[1:]):
                    failures.append(f"{window_id}/{index}: a later call was not reset")
        # --- W: one owner per word, recomputed from the cached raw pieces
        cores = [(p["core_start_sec"], p["core_end_sec"]) for p in record["raw_pieces"]]
        raw: list[dict] = []
        for piece in record["raw_pieces"]:
            offset = float(piece["input_start_sec"])
            for segment_index, segment in enumerate(piece["segments"]):
                for word_index, word in enumerate(segment.get("words") or []):
                    start = float(word["start"]) + offset
                    end = float(word["end"]) + offset
                    raw.append({"text": str(word.get("word", "")),
                                "start": start, "end": end,
                                "midpoint": (start + end) / 2.0,
                                "piece_index": piece["piece_index"],
                                "segment_index": segment_index,
                                "word_index": word_index})
        recomputed = merge_words_by_core(raw, cores)
        if recomputed["kept"] != record["kept"]:
            failures.append(f"{window_id}: cached merge is not reproducible")
        if recomputed["kept"] == 0:
            failures.append(f"{window_id}: the merge kept no words at all")
        if recomputed["kept"] + recomputed["non_owner"] + recomputed["outside_audio"] \
                != len(raw):
            failures.append(f"{window_id}: candidates are not conserved")
        expected = sum(1 for c in raw
                       if owning_core_index(c["midpoint"], cores) == c["piece_index"])
        if expected != recomputed["kept"]:
            failures.append(f"{window_id}: ownership is not an if-and-only-if")
        kept += recomputed["kept"]
        non_owner += recomputed["non_owner"]
        outside += recomputed["outside_audio"]
        candidates += len(raw)
    # --- X's OWN criterion, not F's. In X a blind CORE cut is not a blind DECODE
    # cut: both neighbours hear across the seam through the margins, and word
    # ownership then picks one side. Measured on the first X decode, blind seams
    # disagreed LESS than silence seams (0.551 vs 0.608), so "zero blind cuts" is
    # the wrong thing to require here. What X actually claims is that no seam is
    # ever decoded without context, and that is what is checked -- and it can fail.
    for window_id, record in state["windows"].items():
        raw = record["raw_pieces"]
        duration = _geometry_seconds(record)
        for index, piece in enumerate(raw):
            core_start, core_end = piece["core_start_sec"], piece["core_end_sec"]
            lookback = core_start - piece["input_start_sec"]
            lookahead = piece["input_end_sec"] - core_end
            if index > 0 and lookback < X_LOOKBACK_SECONDS - 1e-3:
                failures.append(f"{window_id}/{index}: seam decoded with "
                                f"{lookback:.2f} s of lookback, needs "
                                f"{X_LOOKBACK_SECONDS}")
            if index < len(raw) - 1 and lookahead < X_LOOKAHEAD_SECONDS - 1e-3:
                failures.append(f"{window_id}/{index}: seam decoded with "
                                f"{lookahead:.2f} s of lookahead, needs "
                                f"{X_LOOKAHEAD_SECONDS}")
            if index == 0 and piece["input_start_sec"] > 1e-6:
                failures.append(f"{window_id}: the first input does not start at 0")
            # NB: compare against the last CORE end, which is exact. The record's
            # audio_seconds is rounded to 2 dp, so a 1e-3 tolerance against it
            # fires on the rounding, not on the mechanism.
            if index == len(raw) - 1 and piece["input_end_sec"] < core_end - 1e-6:
                failures.append(f"{window_id}: the last input stops before the audio end")
        # the lookahead must be USED, not merely supplied: the left piece has to
        # actually transcribe into the next core at least sometimes
        used = 0
        for index, piece in enumerate(raw[:-1]):
            offset = piece["input_start_sec"]
            core_end = piece["core_end_sec"]
            for segment in piece["segments"]:
                for word in segment.get("words") or []:
                    mid = (float(word["start"]) + float(word["end"])) / 2.0 + offset
                    if mid > core_end + 1e-7:
                        used += 1
        if len(raw) > 1 and used == 0:
            failures.append(f"{window_id}: no piece transcribed into its lookahead, "
                            "so the margin bought nothing")
        context_words += used
    return {
        "arm": "X", "blind_cuts": blind, "valley_cuts": valley,
        "p_forced_cuts_same_windows": p_forced,
        "lookahead_words_used": context_words,
        "candidates": candidates, "kept": kept, "non_owner": non_owner,
        "outside_audio": outside,
        "failures": failures[:20], "n_failures": len(failures),
        "verdict": "PASS" if not failures else "FAIL",
    }


def pilot_verdicts() -> dict:
    """Mechanism verdicts only. No WER is computed here, by design."""
    forced = p_forced_cuts()
    result = {"experiment": EXPERIMENT, "windows": list(PILOT_WINDOW_IDS), "arms": {}}
    for arm in ARMS:
        path = destination_for(arm, pilot=True)
        if not path.exists():
            result["arms"][arm] = {"arm": arm, "verdict": "NOT RUN"}
            continue
        state = json.loads(path.read_text())
        if arm == "F":
            result["arms"][arm] = _f_verdict(state, forced)
        elif arm == "C":
            result["arms"][arm] = _c_verdict(state)
        elif arm == "X":
            result["arms"][arm] = _x_verdict(state, forced)
        else:
            result["arms"][arm] = _w_verdict(state)
    destination = out_dir() / "pilot-verdicts.json"
    CD._write_state(destination, result)
    for arm, report in result["arms"].items():
        print(f"{arm}: {report['verdict']}"
              + (f" -- {report.get('failures') or ''}" if report["verdict"] == "FAIL" else ""))
    print(f"-> {destination}")
    return result


# ============================================================ scoring, 39 windows


def _loo(arm_scores, base_scores, rows, key: str, metric: str = "deletion_rate") -> dict:
    picker = CD.METRIC_PICKERS[metric]
    wids = [row["window_id"] for row in rows]
    full = CD._pooled_delta(arm_scores, base_scores, wids, picker)
    groups = sorted({row[key] for row in rows})
    reversed_by, without = [], {}
    for group in groups:
        keep = [row["window_id"] for row in rows if row[key] != group]
        if not keep:
            continue
        value = CD._pooled_delta(arm_scores, base_scores, keep, picker)
        without[str(group)] = value
        if (full < 0) != (value < 0):
            reversed_by.append(str(group))
    return {"delta": full, "without": without, "sign_reversed_by": reversed_by,
            "stable": not reversed_by}


def _contrast(arm_scores, base_scores, rows) -> dict:
    return {
        "vs": {metric: CD.paired_interval(arm_scores, base_scores, rows, metric)
               for metric in CD.METRIC_PICKERS},
        "leave_one_window_out_deletion": _loo(arm_scores, base_scores, rows, "window_id"),
        "leave_one_meeting_out_deletion": _loo(arm_scores, base_scores, rows, "meeting_id"),
    }


def _extended_rates(scores) -> dict:
    base = CD.rates(scores)
    base["correct_error_rate_s_plus_d_over_n"] = (
        (base["substitutions"] + base["deletions"]) / base["ref_tokens"]
    )
    base["emitted_tokens"] = (
        base["ref_tokens"] - base["deletions"] + base["insertions"]
    )
    return base


def score(rows: list[dict] | None = None, pilot: bool = False) -> Path:
    if rows is None:
        rows = pilot_rows() if pilot else eval_rows()
    rows = list(rows)
    control_path = DA.sc() / "decode-ablation" / "eval-A.json"
    p_path = DA.sc() / CD.SEGMENT_CACHE_DIRNAME / "eval-P.json"
    e_path = DA.sc() / CD.SEGMENT_CACHE_DIRNAME / "eval-E.json"
    control = CD.load_json(control_path)
    control_scores = CD.per_window_scores(control, rows)
    p_scores = CD.per_window_scores(CD.load_json(p_path), rows)
    e_scores = (CD.per_window_scores(CD.load_json(e_path), rows)
                if e_path.exists() else None)

    states: dict[str, dict] = {}
    for arm in ARMS:
        path = destination_for(arm, pilot)
        if path.exists():
            states[arm] = json.loads(path.read_text())
    if "W" in states:
        states["E-WT"] = {
            "windows": {wid: derive_e_wt(rec)
                        for wid, rec in states["W"]["windows"].items()}
        }

    result = {
        "experiment": EXPERIMENT,
        "set": "pilot" if pilot else "eval",
        "exploratory": (
            "DESCRIPTION ONLY over the 5 pilot windows, which were chosen BECAUSE "
            "arm P cuts blind in them. Not representative, no gate, nothing adopted."
            if pilot else
            "These 39 windows generated the forced-cut hypothesis. Every number "
            "here is exploratory, not confirmation."
        ),
        "n_windows": len(rows),
        "n_meetings": len({row["meeting_id"] for row in rows}),
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "blocks": "meeting_id"},
        "reference": {
            "A": _extended_rates(control_scores),
            "P": _extended_rates(p_scores),
        },
        "arms": {},
    }
    if e_scores is not None:
        result["reference"]["E"] = _extended_rates(e_scores)

    for arm, state in states.items():
        arm_scores = CD.per_window_scores(state, rows)
        report = {
            "rates": _extended_rates(arm_scores),
            "vs_A": _contrast(arm_scores, control_scores, rows),
            "vs_P": _contrast(arm_scores, p_scores, rows),
        }
        if arm == "W" and "E-WT" in states:
            report["vs_E_WT"] = _contrast(
                arm_scores, CD.per_window_scores(states["E-WT"], rows), rows)
        if arm == "E-WT" and e_scores is not None:
            report["vs_E"] = _contrast(arm_scores, e_scores, rows)
        result["arms"][arm] = report

    destination = out_dir() / ("results-pilot.json" if pilot else "results-eval.json")
    CD._write_state(destination, result)
    repo_copy = ROOT / "eval" / (
        "results_seam_repair_pilot.json" if pilot else "results_seam_repair.json")
    repo_copy.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for arm, report in result["arms"].items():
        rates = report["rates"]
        delta = report["vs_A"]["vs"]["wer"]
        print(f"{arm}: WER {rates['wer']:.5f} "
              f"delta_vs_A {delta['delta']:+.5f} "
              f"CI [{delta['ci95'][0]:+.5f}, {delta['ci95'][1]:+.5f}] "
              f"(S+D)/N {rates['correct_error_rate_s_plus_d_over_n']:.5f}")
    print(f"-> {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--arm", required=True, choices=ARMS)
    d.add_argument("--pilot", action="store_true")
    d.add_argument("--limit", type=int, default=None)
    sub.add_parser("pilot-verdicts")
    sc = sub.add_parser("score")
    sc.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.command == "decode":
        decode(args.arm, pilot=args.pilot, limit=args.limit)
    elif args.command == "pilot-verdicts":
        pilot_verdicts()
    else:
        score(pilot=args.pilot)


if __name__ == "__main__":
    main()
