"""The deployable form of arm P must be the same policy the experiment measured.

Arm P is the best measured arm by WER on the 39 frozen evaluation windows
(0.14751 against the control's 0.15893). It is **not** an arm that passed a gate:
the preregistered gate of `exp-2026-08-17-chunking-aware-decoding` was on
DELETIONS, and P's deletion delta has a confidence interval containing zero. If
this file's assertions fail, the served policy has drifted away from the measured
one and no number attached to it is valid.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from serve import decode_p as DP
from eval import chunking_decode as CD


# --------------------------------------------------------------- frozen policy

def test_the_policy_matches_the_experiment_constants():
    """A drift between the served policy and the measured arm is a silent lie."""
    seg = DP.POLICY["segmentation"]
    assert seg["max_chunk_seconds"] == CD.MAX_CHUNK_SECONDS == 29.5
    assert seg["min_chunk_seconds"] == CD.MIN_CHUNK_SECONDS == 5.0
    assert seg["min_speech_seconds"] == CD.MIN_SPEECH_SECONDS == 2.5
    assert seg["min_silence_seconds"] == CD.MIN_SILENCE_SECONDS == 0.5
    vad = DP.POLICY["vad"]
    assert vad["min_silence_duration_ms"] == int(CD.MINIMUM_SILENCE_SECONDS * 1000)
    assert vad["speech_pad_ms"] == 0
    assert DP.POLICY["experiment"] == CD.EXPERIMENT
    assert DP.POLICY["model_artifact_id"] == CD.MODEL_ARTIFACT_ID
    assert DP.POLICY["ledger_experiment_id"] == "exp-2026-08-18-chunking-aware-decoding"


def test_the_policy_decodes_with_control_unchanged():
    """P's only declared change is audio geometry. The decode config is CONTROL."""
    assert DP._thaw(DP.POLICY["decode"]) == CD.CONTROL
    assert DP.POLICY["decode"]["word_timestamps"] is False
    assert DP.POLICY["decode"]["vad_filter"] is False
    assert DP.POLICY["decode"]["condition_on_previous_text"] is False
    assert DP.POLICY["decode"]["language"] == "el"
    assert list(DP.POLICY["decode"]["temperature"]) == list(CD.CONTROL["temperature"])


def test_the_policy_digest_is_pinned():
    """A test that only says 'the hash changes' would let any policy through."""
    assert DP.policy_sha256() == DP.EXPECTED_POLICY_SHA256


def test_the_policy_digest_is_stable_and_moves_with_the_policy():
    first = DP.policy_sha256()
    assert first == DP.policy_sha256()
    mutated = DP._thaw(DP.POLICY)
    mutated["segmentation"]["max_chunk_seconds"] = 27.0
    assert DP._digest(mutated) != first


def test_the_policy_cannot_be_edited_in_place():
    with pytest.raises(TypeError):
        DP.POLICY["segmentation"] = {}
    with pytest.raises(TypeError):
        DP.POLICY["segmentation"]["max_chunk_seconds"] = 1.0


# ------------------------------------------------------- frozen planner fixture
#
# Characterisation, not derivation: these are the pieces `CD.split_accumulating`
# produced on 2026-08-21. They are hard-coded so that any future edit to the
# splitter breaks this test instead of silently changing what "arm P" means.
# Every case is a segmentation regime the policy must handle.

PLANNER_FIXTURE = {
    "vad_cuts": {
        "duration": 100.0,
        "silences": [(24.0, 25.0), (50.0, 51.5), (76.0, 77.0)],
        "pieces": [(0.0, 24.5), (24.5, 50.75), (50.75, 76.5), (76.5, 100.0)],
        "forced_cuts": 0,
        "kinds": ["vad_silence", "vad_silence", "vad_silence"],
    },
    "blind_continuous_speech": {
        "duration": 70.0,
        "silences": [],
        "pieces": [(0.0, 29.5), (29.5, 59.0), (59.0, 70.0)],
        "forced_cuts": 2,
        "kinds": ["forced", "forced"],
    },
    "silences_below_the_minimum": {
        "duration": 70.0,
        "silences": [(20.0, 20.2), (45.0, 45.3)],
        "pieces": [(0.0, 29.5), (29.5, 59.0), (59.0, 70.0)],
        "forced_cuts": 2,
        "kinds": ["forced", "forced"],
    },
    "short_tail_boundary_moved": {
        "duration": 62.0,
        "silences": [],
        "pieces": [(0.0, 29.5), (29.5, 45.75), (45.75, 62.0)],
        "forced_cuts": 2,
        "kinds": ["forced", "forced"],
    },
    "voiced_ground_fallback": {
        # the only in-range candidate leaves under min_speech voiced seconds;
        # the ceiling overrides the speech minimum, so the cut is taken anyway
        "duration": 60.0,
        "silences": [(0.0, 28.0), (28.5, 29.0)],
        "pieces": [(0.0, 28.75), (28.75, 44.375), (44.375, 60.0)],
        "forced_cuts": 1,
        "kinds": ["vad_silence", "forced"],
    },
    "stale_silence_provenance": {
        # the short-tail repair moves the boundary from 25.5 to 14.8 but keeps
        # the piece's recorded `cut_silence_duration` of 1.0 s. The recorded
        # field is therefore NOT provenance; the classifier must say "forced".
        "duration": 29.6,
        "silences": [(25.0, 26.0)],
        "pieces": [(0.0, 14.8), (14.8, 29.6)],
        "forced_cuts": 0,
        "kinds": ["forced"],
    },
}


@pytest.mark.parametrize("name", sorted(PLANNER_FIXTURE))
def test_the_planner_reproduces_the_frozen_fixture(name):
    case = PLANNER_FIXTURE[name]
    silences = [CD.Silence(a, b) for a, b in case["silences"]]
    pieces, counters = DP.plan_pieces(case["duration"], silences)
    assert [(round(p.start, 6), round(p.end, 6)) for p in pieces] == \
        [(round(a, 6), round(b, 6)) for a, b in case["pieces"]]
    assert counters["forced_cuts"] == case["forced_cuts"]
    assert counters["tiny_chunks"] == 0
    assert counters["speech_dropped"] == 0.0


@pytest.mark.parametrize("name", sorted(PLANNER_FIXTURE))
def test_the_boundary_kinds_reproduce_the_frozen_fixture(name):
    case = PLANNER_FIXTURE[name]
    silences = [CD.Silence(a, b) for a, b in case["silences"]]
    pieces, _ = DP.plan_pieces(case["duration"], silences)
    boundaries = DP.classify_boundaries(pieces, silences)
    assert [b.kind for b in boundaries] == case["kinds"]
    assert len(boundaries) == len(pieces) - 1


@pytest.mark.parametrize("name", sorted(PLANNER_FIXTURE))
def test_no_piece_exceeds_the_whisper_window(name):
    """The short-tail repair moves boundaries; it must never breach the ceiling."""
    case = PLANNER_FIXTURE[name]
    silences = [CD.Silence(a, b) for a, b in case["silences"]]
    pieces, _ = DP.plan_pieces(case["duration"], silences)
    assert max(p.end - p.start for p in pieces) <= CD.MAX_CHUNK_SECONDS + 1e-6


# --------------------------------------------------------- boundary classifier

def test_a_boundary_inside_a_qualifying_silence_reports_that_silence():
    pieces = [CD.Piece(0.0, 25.5), CD.Piece(25.5, 40.0)]
    b, = DP.classify_boundaries(pieces, [CD.Silence(25.0, 26.0)])
    assert (b.at, b.kind, b.silence_seconds) == (25.5, "vad_silence", 1.0)


def test_a_boundary_at_a_silence_endpoint_uses_half_open_intervals():
    """[start, end): the opening endpoint is inside, the closing one is not."""
    opened, = DP.classify_boundaries(
        [CD.Piece(0.0, 25.0), CD.Piece(25.0, 40.0)], [CD.Silence(25.0, 26.0)])
    closed, = DP.classify_boundaries(
        [CD.Piece(0.0, 26.0), CD.Piece(26.0, 40.0)], [CD.Silence(25.0, 26.0)])
    assert opened.kind == "vad_silence"
    assert closed.kind == "forced"


def test_a_silence_below_the_minimum_does_not_justify_a_boundary():
    b, = DP.classify_boundaries(
        [CD.Piece(0.0, 25.1), CD.Piece(25.1, 40.0)], [CD.Silence(25.0, 25.2)])
    assert (b.kind, b.silence_seconds) == ("forced", None)


def test_the_final_piece_end_is_not_a_boundary():
    pieces = [CD.Piece(0.0, 25.5), CD.Piece(25.5, 40.0)]
    assert [b.at for b in DP.classify_boundaries(pieces, [])] == [25.5]


def test_overlapping_silences_are_rejected_rather_than_guessed_at():
    with pytest.raises(ValueError):
        DP.classify_boundaries(
            [CD.Piece(0.0, 10.0), CD.Piece(10.0, 20.0)],
            [CD.Silence(5.0, 12.0), CD.Silence(9.0, 15.0)])


# ------------------------------------------------------------ decode behaviour

class FakeSegment(dict):
    pass


class FakeModel:
    """Records exactly what the policy asks the decoder to do."""

    def __init__(self, per_call):
        self.per_call = list(per_call)
        self.calls = []
        self.unconsumed = 0

    def transcribe(self, audio, **config):
        index = len(self.calls)
        self.calls.append({"n_samples": len(audio), "config": config})
        segments = self.per_call[index]
        self.unconsumed += 1

        def generator():
            for segment in segments:
                yield segment
            self.unconsumed -= 1

        return generator(), None


def _audio(seconds):
    import numpy as np
    return np.zeros(int(seconds * CD.SAMPLE_RATE), dtype="float32")


def _decoder(per_call, silences):
    decoder = DP.DecodeP(FakeModel(per_call))
    decoder._detect = lambda audio, duration: (list(silences), "stub", None)
    return decoder


def test_every_piece_is_decoded_once_in_order_with_control_unchanged():
    per_call = [[{"text": f" piece{i}", "start": 0.0, "end": 1.0}] for i in range(3)]
    decoder = _decoder(per_call, [])
    result = decoder.transcribe(_audio(70.0))
    assert len(decoder.model.calls) == 3
    for call in decoder.model.calls:
        assert call["config"] == CD.CONTROL
    assert [call["n_samples"] for call in decoder.model.calls] == [
        int(round(29.5 * CD.SAMPLE_RATE)),
        int(round(29.5 * CD.SAMPLE_RATE)),
        int(round(11.0 * CD.SAMPLE_RATE)),
    ]
    assert result.text == "piece0 piece1 piece2"


def test_each_lazy_segment_generator_is_consumed_before_the_next_piece():
    per_call = [[{"text": " a", "start": 0.0, "end": 1.0}],
                [{"text": " b", "start": 0.0, "end": 1.0}],
                [{"text": " c", "start": 0.0, "end": 1.0}]]
    decoder = _decoder(per_call, [])
    decoder.transcribe(_audio(70.0))
    assert decoder.model.unconsumed == 0


def test_the_pieces_tile_the_audio_exactly_in_samples():
    decoder = _decoder([[] for _ in range(3)], [])
    decoder.transcribe(_audio(70.0))
    total = sum(call["n_samples"] for call in decoder.model.calls)
    assert total == int(70.0 * CD.SAMPLE_RATE)


def test_segment_timestamps_are_shifted_once_by_the_piece_offset():
    per_call = [[{"text": " a", "start": 1.0, "end": 2.0}],
                [{"text": " b", "start": 3.0, "end": 4.0}],
                [{"text": " c", "start": 0.5, "end": 1.5}]]
    decoder = _decoder(per_call, [])
    result = decoder.transcribe(_audio(70.0))
    assert [(s["start"], s["end"]) for s in result.segments] == [
        (1.0, 2.0), (32.5, 33.5), (59.5, 60.5)]


def test_the_transcript_carries_its_own_provenance():
    decoder = _decoder([[] for _ in range(3)], [])
    record = decoder.transcribe(_audio(70.0)).as_dict()
    assert record["policy_id"] == DP.POLICY_ID
    assert record["policy_sha256"] == DP.policy_sha256()
    assert record["model_artifact_id"] == CD.MODEL_ARTIFACT_ID
    assert [b["kind"] for b in record["boundaries"]] == ["forced", "forced"]
    assert record["counters"]["n_pieces"] == 3
    assert record["counters"]["forced_boundaries"] == 2


def test_seeding_is_recorded_and_only_happens_when_asked(monkeypatch):
    seeds = []
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "set_random_seed", seeds.append)

    decoder = _decoder([[] for _ in range(3)], [])
    assert decoder.transcribe(_audio(70.0)).seed is None
    assert seeds == []

    decoder = _decoder([[] for _ in range(3)], [])
    assert decoder.transcribe(_audio(70.0), seed=1234).seed == 1234
    assert seeds == [1234]


def test_audio_that_is_not_mono_float32_16k_is_refused():
    import numpy as np
    decoder = _decoder([[]], [])
    with pytest.raises(ValueError):
        decoder.transcribe(np.zeros((2, 1000), dtype="float32"))
    with pytest.raises(ValueError):
        decoder.transcribe(np.zeros(1000, dtype="float64"))
    with pytest.raises(ValueError, match="empty"):
        decoder.transcribe(np.zeros(0, dtype="float32"))


# ------------------------------------------------------------ model verification

def test_load_refuses_a_model_directory_that_is_not_the_expected_artifact(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"not the fine-tune")
    with pytest.raises(ValueError, match="not evidence"):
        DP.DecodeP.load(model_dir=tmp_path)


def test_load_refuses_a_missing_model_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        DP.DecodeP.load(model_dir=tmp_path / "absent")


def test_verification_covers_the_directory_that_would_be_loaded(tmp_path):
    """Overriding model_dir must not be a way around the hash check."""
    (tmp_path / "model.bin").write_bytes(b"not the fine-tune")
    seen = []
    with pytest.raises(ValueError):
        DP.DecodeP.load(model_dir=tmp_path, _model_factory=seen.append)
    assert seen == []          # the model was never constructed


# ------------------------------------------------- the measured arm, on real audio

CACHE = Path.home() / ".cache/oc-public/chunking-decode-2026-08/eval-P.json"
PINNED_WINDOW = "win_orestiada_mar23_2026_6383163"


def test_the_served_policy_reproduces_the_measured_arm_boundaries():
    """The strongest available check without a GPU: same audio, same cuts.

    Pinned to one window ID on purpose. 'Whichever window happens to exist'
    makes the coverage of this test depend on the machine it runs on.
    """
    if not CACHE.exists():
        pytest.skip("the arm P decode cache is not on this machine")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
    import decode_ablation as DA

    state = json.loads(CACHE.read_text())
    record = state["windows"][PINNED_WINDOW]
    wav = DA.sc() / "bench_windows" / f"{PINNED_WINDOW}.wav"
    if not wav.exists():
        pytest.skip("the pinned window's audio is not on this machine")

    audio = CD._load_audio(wav)
    duration = len(audio) / CD.SAMPLE_RATE
    silences, _, _ = CD.detect_silences(audio, duration, None)
    pieces, _ = DP.plan_pieces(duration, silences)
    served = [(round(p.start, 3), round(p.end, 3)) for p in pieces]
    cached = [(round(p["start_sec"], 3), round(p["end_sec"], 3))
              for p in record["pieces"]]
    assert served == cached
