"""Model-free contract tests for the chunking-aware decoding experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval import chunking_decode as M
import decode_ablation as DA


def test_arm_v_changes_only_vad_filter():
    resolved = M.config_for("V")
    assert {key for key in resolved if resolved[key] != DA.CONTROL[key]} == {"vad_filter"}
    assert resolved["vad_filter"] is True
    assert {key: value for key, value in resolved.items() if key != "vad_filter"} == {
        key: value for key, value in DA.CONTROL.items() if key != "vad_filter"
    }


def test_arm_p_piece_durations_and_cuts_are_reported_silences():
    silences = [
        M.Silence(10.0, 16.0),
        M.Silence(25.0, 26.0),
        M.Silence(36.0, 43.0),
        M.Silence(50.0, 52.0),
        M.Silence(58.0, 61.0),
    ]
    pieces, fallbacks = M.split_at_silences(65.0, silences)

    assert fallbacks == 0
    assert all(piece.duration <= 30.0 + 1e-7 for piece in pieces)
    assert all(first.end == second.start for first, second in zip(pieces, pieces[1:]))
    for piece in pieces[:-1]:
        assert piece.cut_silence_duration is not None
        cut = piece.end
        assert any(silence.start <= cut <= silence.end for silence in silences)
    assert [round(piece.cut_silence_duration, 3) for piece in pieces[:-1]] == [6.0, 7.0]


def test_piece_concatenation_preserves_order_and_offsets_timestamps():
    combined = M.combine_piece_transcripts([
        {
            "start_sec": 10.0,
            "text": " first",
            "segments": [{"text": " first", "start": 0.5, "end": 1.5}],
        },
        {
            "start_sec": 30.0,
            "text": " second",
            "segments": [{"text": " second", "start": 0.25, "end": 1.0}],
        },
    ])

    assert combined["text"] == "first second"
    assert [segment["text"] for segment in combined["segments"]] == [" first", " second"]
    assert combined["segments"][0]["start"] == pytest.approx(10.5)
    assert combined["segments"][0]["end"] == pytest.approx(11.5)
    assert combined["segments"][1]["start"] == pytest.approx(30.25)
    assert combined["segments"][1]["end"] == pytest.approx(31.0)


def test_no_qualifying_silence_is_whole_and_counted():
    pieces, fallbacks = M.split_at_silences(65.0, [M.Silence(40.0, 45.0)])

    assert len(pieces) == 1
    assert pieces[0].start == 0.0 and pieces[0].end == 65.0
    assert pieces[0].whole_fallback is True
    assert fallbacks == 1


def test_arm_pi_never_decodes_more_than_25_seconds_and_always_pads_to_30(
    tmp_path, monkeypatch
):
    rows = [
        {"window_id": "w0", "duration_sec": 40.0, "meeting_id": "m0"},
        {"window_id": "w1", "duration_sec": 12.0, "meeting_id": "m1"},
    ]
    decoder_lengths = []

    class FixtureModel:
        def transcribe(self, audio, **config):
            assert config == DA.CONTROL
            decoder_lengths.append(len(audio) / M.SAMPLE_RATE)
            return iter([SimpleNamespace(text=" piece", start=0.5, end=1.5)]), \
                SimpleNamespace(duration=len(audio) / M.SAMPLE_RATE)

    monkeypatch.setattr(M, "environment", lambda: {"fixture": True})
    monkeypatch.setattr(M.DA, "sc", lambda: tmp_path)
    for row in rows:
        path = tmp_path / "bench_windows" / f"{row['window_id']}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fixture_audio_loader(path):
        duration = 40.0 if path.stem == "w0" else 12.0
        return np.zeros(round(M.SAMPLE_RATE * duration), dtype=np.float32)

    result = M.decode(
        "PI",
        model=FixtureModel(),
        destination=tmp_path / "chunking" / "eval-PI.json",
        rows_override=rows,
        audio_loader=fixture_audio_loader,
        silence_detector=lambda audio, duration, model: ([], "fixture", None),
    )
    state = json.loads(result.read_text())

    assert decoder_lengths == [30.0, 30.0, 30.0]
    for window in state["windows"].values():
        assert all(piece["speech_seconds"] <= 25.0 for piece in window["pieces"])
        assert all(piece["padded_duration_sec"] == 30.0 for piece in window["pieces"])
        assert all(
            piece["speech_seconds"] + piece["padding_seconds"] == pytest.approx(30.0)
            for piece in window["pieces"]
        )
        assert window["speech_seconds_per_piece"] == [
            piece["speech_seconds"] for piece in window["pieces"]
        ]
        assert window["padding_seconds_per_piece"] == [
            piece["padding_seconds"] for piece in window["pieces"]
        ]


def test_arm_e_stride_and_kept_centre_tile_multiple_durations():
    for duration in (12.0, 30.0, 37.0, 65.0):
        windows = M.overlap_layout(duration)
        M.assert_exact_cover(windows, duration)
        assert sum(window.kept_centre_width for window in windows) == pytest.approx(duration)
        assert windows[0].kept_start == pytest.approx(0.0)
        assert windows[-1].kept_end == pytest.approx(duration)
        assert all(
            first.kept_end == pytest.approx(second.kept_start)
            for first, second in zip(windows, windows[1:])
        )
        assert all(window.duration == pytest.approx(30.0) for window in windows)


def test_arm_e_seam_tie_is_deterministic_and_chooses_earlier_centre():
    windows = M.overlap_layout(30.0)
    candidates = [
        {"text": " right", "start": 14.9, "end": 15.1,
         "midpoint": 15.0, "window_index": 1},
        {"text": " left", "start": 14.9, "end": 15.1,
         "midpoint": 15.0, "window_index": 0},
    ]

    first = M.merge_overlap_segments(candidates, windows, 30.0)
    second = M.merge_overlap_segments(list(reversed(candidates)), windows, 30.0)

    assert M.nearest_overlap_window_index(15.0, windows) == 0
    assert first == second
    assert first["text"] == "left"
    assert first["tokens_dropped_as_duplicates_at_seams"] == 1


def test_arm_e_kept_regions_retain_audio_at_both_outer_boundaries():
    duration = 37.0
    windows = M.overlap_layout(duration)
    candidates = [
        {"text": " start", "start": 0.0, "end": 1.0,
         "midpoint": 0.5, "window_index": 0},
        {"text": " end", "start": 36.0, "end": 37.0,
         "midpoint": 36.5, "window_index": len(windows) - 1},
    ]

    merged = M.merge_overlap_segments(candidates, windows, duration)

    assert windows[0].kept_start == pytest.approx(0.0)
    assert windows[-1].kept_end == pytest.approx(duration)
    assert merged["text"] == "start end"
    assert [segment["start"] for segment in merged["segments"]] == [0.0, 36.0]


def test_adding_pi_and_e_preserves_v_p_identity_and_gate_evaluation(tmp_path, monkeypatch):
    assert M.config_for("V") == dict(DA.CONTROL, vad_filter=True)
    assert M.config_for("P") == DA.CONTROL
    assert M.config_for("PI") == DA.CONTROL
    assert M.config_for("E") == DA.CONTROL
    assert M.destination_for("V") == M.out_dir() / "eval-V.json"
    assert M.destination_for("P") == M.out_dir() / "eval-P.json"

    rows = [
        {"window_id": "w0", "duration_sec": 1.0, "meeting_id": "m0",
         "reference_text": "alpha"},
        {"window_id": "w1", "duration_sec": 1.0, "meeting_id": "m1",
         "reference_text": "beta"},
    ]
    control = {"windows": {"w0": {"text": "alpha"}, "w1": {"text": "beta"}}}
    v = {"windows": {"w0": {"text": "alpha"}, "w1": {"text": "beta"}}}
    p = {"windows": {"w0": {"text": "alpha"}, "w1": {"text": "beta"}}}
    pi = {"windows": {"w0": {"text": "alpha"}, "w1": {"text": "beta"}}}
    e = {"windows": {"w0": {"text": "alpha"}, "w1": {"text": "beta"}}}

    baseline = M.score_states(control, {"V": v, "P": p}, rows)
    extended = M.score_states(control, {"V": v, "P": p, "PI": pi, "E": e}, rows)

    assert extended["arms"]["V"] == baseline["arms"]["V"]
    assert extended["arms"]["P"] == baseline["arms"]["P"]


def test_gate_rejects_lower_wer_that_raises_deletions():
    comparison = {
        "wer": {"delta": -0.004, "ci95": [-0.010, -0.001]},
        "deletion_rate": {"delta": 0.003, "ci95": [0.001, 0.006]},
        "insertion_rate": {"delta": -0.001, "ci95": [-0.003, 0.001]},
        "substitution_rate": {"delta": -0.006, "ci95": [-0.010, -0.002]},
    }
    gate = M.evaluate_gate(comparison, {"stable": True})

    assert len(gate["conditions"]) == 4
    assert gate["conditions"]["deletion_rate_delta_negative_and_ci95_upper_lt_0"] is False
    assert gate["overall_pass"] is False
    assert gate["verdict"] == "FAIL"


def test_decode_is_two_window_smoke_with_fixture_model(tmp_path, monkeypatch):
    rows = [
        {"window_id": "w0", "duration_sec": 12.0, "meeting_id": "m0"},
        {"window_id": "w1", "duration_sec": 12.0, "meeting_id": "m1"},
    ]

    class FixtureModel:
        def transcribe(self, audio, **config):
            duration = len(audio) / M.SAMPLE_RATE if not isinstance(audio, (str, Path)) else 12.0
            text = " A" if config["vad_filter"] else " B"
            return iter([SimpleNamespace(text=text, start=10.0, end=min(11.0, duration))]), \
                SimpleNamespace(duration=duration, transcription_options=config)

    monkeypatch.setattr(M, "environment", lambda: {"fixture": True})
    monkeypatch.setattr(M.DA, "sc", lambda: tmp_path)
    destination = tmp_path / "chunking" / "eval-V.json"
    for row in rows:
        (tmp_path / "bench_windows" / f"{row['window_id']}.wav").parent.mkdir(
            parents=True, exist_ok=True
        )
        (tmp_path / "bench_windows" / f"{row['window_id']}.wav").touch()

    result = M.decode("V", model=FixtureModel(), destination=destination, rows_override=rows)
    state = json.loads(result.read_text())
    assert set(state["windows"]) == {"w0", "w1"}
    assert all(window["text"] == "A" for window in state["windows"].values())

    def fixture_audio_loader(_path):
        return np.zeros(M.SAMPLE_RATE * 12, dtype=np.float32)

    def fixture_silence_detector(audio, duration, model):
        return [], "fixture", None

    p_destination = tmp_path / "chunking" / "eval-P.json"
    p_result = M.decode(
        "P",
        model=FixtureModel(),
        destination=p_destination,
        rows_override=rows,
        audio_loader=fixture_audio_loader,
        silence_detector=fixture_silence_detector,
    )
    p_state = json.loads(p_result.read_text())
    assert set(p_state["windows"]) == {"w0", "w1"}
    assert all(window["text"] == "B" for window in p_state["windows"].values())
    assert all(window["n_pieces"] == 1 for window in p_state["windows"].values())

    e_destination = tmp_path / "chunking" / "eval-E.json"
    e_result = M.decode(
        "E",
        model=FixtureModel(),
        destination=e_destination,
        rows_override=rows,
        audio_loader=fixture_audio_loader,
    )
    e_state = json.loads(e_result.read_text())
    assert set(e_state["windows"]) == {"w0", "w1"}
    assert all(window["text"] == "B" for window in e_state["windows"].values())
    assert all(window["n_overlapping_windows"] == 1 for window in e_state["windows"].values())


@pytest.mark.parametrize("field,value", [
    ("model", "another-model"),
    ("config", {"different": True}),
    ("environment", {"different": True}),
])
def test_decode_refuses_to_extend_cache_with_different_identity(
    tmp_path, monkeypatch, field, value
):
    monkeypatch.setattr(M, "environment", lambda: {"fixture": True})
    destination = tmp_path / "eval-V.json"
    fresh = M._fresh_state("V", M.MODEL_SHA256_16, {"fixture": True})
    existing = json.loads(json.dumps(fresh))
    existing[field] = value
    destination.write_text(json.dumps(existing))

    with pytest.raises(ValueError, match="different"):
        M.decode("V", model=object(), destination=destination, rows_override=[])


def test_real_decode_dependencies_skip_cleanly_when_absent():
    if not M.MODEL_DIR.is_dir():
        pytest.skip(f"CT2 model directory absent: {M.MODEL_DIR}")
    control = DA.sc() / "decode-ablation" / "eval-A.json"
    if not control.exists():
        pytest.skip(f"control cache absent: {control}")


@pytest.mark.parametrize(
    "duration,silences",
    [
        (3.0, []),
        (31.0, []),
        (60.0, []),
        (
            90.0,
            [
                M.Silence(24.0, 27.0),
                M.Silence(51.0, 54.0),
                M.Silence(79.0, 81.0),
            ],
        ),
        (148.0, [M.Silence(8.352, 10.400)]),
        (
            31.5,
            [M.Silence(14.0, 16.0), M.Silence(27.5, 28.5)],
        ),
    ],
)
def test_split_accumulating_piece_lengths_are_legal(duration, silences):
    pieces, counters = M.split_accumulating(duration, silences)

    if duration < M.MIN_CHUNK_SECONDS:
        assert len(pieces) == 1
        assert pieces[0].duration == pytest.approx(duration)
    else:
        assert all(
            M.MIN_CHUNK_SECONDS - 1e-7
            <= piece.duration
            <= M.MAX_CHUNK_SECONDS + 1e-7
            for piece in pieces
        )
    assert counters["tiny_chunks"] == 0


@pytest.mark.parametrize(
    "duration,silences",
    [
        (3.0, []),
        (31.0, []),
        (60.0, []),
        (90.0, [M.Silence(24.0, 27.0), M.Silence(51.0, 54.0), M.Silence(79.0, 81.0)]),
        (148.0, [M.Silence(8.352, 10.400)]),
        (31.5, [M.Silence(14.0, 16.0), M.Silence(27.5, 28.5)]),
    ],
)
def test_split_accumulating_pieces_tile_the_input_exactly(duration, silences):
    pieces, counters = M.split_accumulating(duration, silences)

    assert pieces[0].start == pytest.approx(0.0, abs=1e-6)
    assert pieces[-1].end == pytest.approx(duration, abs=1e-6)
    for first, second in zip(pieces, pieces[1:]):
        assert first.end == pytest.approx(second.start, abs=1e-6)
    assert counters["speech_dropped"] == 0


def test_split_accumulating_non_forced_cuts_are_inside_reported_silences():
    silences = [
        M.Silence(24.0, 27.0),
        M.Silence(51.0, 54.0),
        M.Silence(79.0, 81.0),
    ]
    pieces, counters = M.split_accumulating(90.0, silences)

    assert counters["forced_cuts"] == 0
    for piece in pieces[:-1]:
        assert piece.cut_silence_duration is not None
        assert piece.cut_silence_duration >= M.MIN_SILENCE_SECONDS
        assert any(silence.start <= piece.end <= silence.end for silence in silences)


@pytest.mark.parametrize(
    "duration,expected_forced_cuts",
    [(31.0, 1), (60.0, 2)],
)
def test_split_accumulating_forces_continuous_speech_cuts_without_dropping_audio(
    duration, expected_forced_cuts
):
    pieces, counters = M.split_accumulating(duration, [])

    assert counters["forced_cuts"] == expected_forced_cuts
    assert counters["speech_dropped"] == 0
    assert counters["tiny_chunks"] == 0
    assert counters["unjustified_forced_cuts"] == 0
    assert pieces[0].start == 0.0
    assert pieces[-1].end == duration


def test_split_accumulating_short_final_tail_stays_as_one_legal_piece_when_it_fits():
    pieces, counters = M.split_accumulating(28.0, [M.Silence(22.0, 27.0)])

    assert len(pieces) == 1
    assert pieces[0].start == 0.0
    assert pieces[0].end == 28.0
    assert M.MIN_CHUNK_SECONDS <= pieces[0].duration <= M.MAX_CHUNK_SECONDS
    assert counters["tiny_chunks"] == 0


def test_split_accumulating_short_final_tail_moves_shared_boundary_when_merge_would_breach_max():
    silences = [M.Silence(14.0, 16.0), M.Silence(27.5, 28.5)]
    pieces, counters = M.split_accumulating(31.5, silences)

    assert len(pieces) == 2
    assert [piece.duration for piece in pieces] == pytest.approx([15.0, 16.5])
    assert M.MIN_CHUNK_SECONDS <= pieces[0].duration <= M.MAX_CHUNK_SECONDS
    assert M.MIN_CHUNK_SECONDS <= pieces[1].duration <= M.MAX_CHUNK_SECONDS
    assert any(silence.start <= pieces[0].end <= silence.end for silence in silences)
    assert counters["tiny_chunks"] == 0


def test_split_accumulating_single_long_silence_is_not_reentered():
    pieces, counters = M.split_accumulating(
        148.0, [M.Silence(8.352, 10.400)]
    )

    assert len(pieces) < 10
    assert min(piece.duration for piece in pieces) >= M.MIN_CHUNK_SECONDS
    assert counters["tiny_chunks"] == 0
    assert counters["speech_dropped"] == 0
    assert counters["unjustified_forced_cuts"] == 0


def test_split_accumulating_counters_are_zero_on_all_contract_fixtures():
    fixtures = [
        (3.0, []),
        (31.0, []),
        (60.0, []),
        (90.0, [M.Silence(24.0, 27.0), M.Silence(51.0, 54.0), M.Silence(79.0, 81.0)]),
        (148.0, [M.Silence(8.352, 10.400)]),
        (31.5, [M.Silence(14.0, 16.0), M.Silence(27.5, 28.5)]),
    ]

    for duration, silences in fixtures:
        _, counters = M.split_accumulating(duration, silences)
        assert counters["tiny_chunks"] == 0
        assert counters["speech_dropped"] == 0
        assert counters["unjustified_forced_cuts"] == 0


def test_split_accumulating_is_deterministic():
    duration = 148.0
    silences = [M.Silence(8.352, 10.400)]

    first_pieces, first_counters = M.split_accumulating(duration, silences)
    second_pieces, second_counters = M.split_accumulating(duration, silences)

    assert first_pieces == second_pieces
    assert first_counters == second_counters


def test_segment_only_reports_cached_geometry_without_loading_a_model(tmp_path, monkeypatch, capsys):
    rows = [
        {"window_id": "w0", "duration_sec": 35.0},
        {"window_id": "w1", "duration_sec": 31.0},
    ]
    (tmp_path / "eval-P.json").write_text(json.dumps({
        "windows": {
            "w0": {
                "pieces": [{"end_sec": 8.0, "cut_silence_duration_sec": 2.0}]
            }
        }
    }))
    (tmp_path / "eval-V.json").write_text(json.dumps({
        "windows": {
            "w1": {"segments": [{"start": 0.0, "end": 31.0}]}
        }
    }))
    monkeypatch.setattr(M, "verify_model", lambda: pytest.fail("model verification called"))
    monkeypatch.setattr(M, "environment", lambda: pytest.fail("environment called"))

    result = M.segment_only(cache_root=tmp_path, rows_override=rows)
    output = capsys.readouterr().out

    assert "w0 audio_seconds=35.00 n_pieces=2" in output
    assert "w1 audio_seconds=31.00 n_pieces=2" in output
    assert "summary tiny_chunks=0 speech_dropped=0.000000 unjustified_forced_cuts=0" in output
    assert result["totals"] == {
        "tiny_chunks": 0,
        "speech_dropped": 0.0,
        "unjustified_forced_cuts": 0,
    }


def test_segment_only_subcommand_dispatches_to_model_free_report(monkeypatch):
    called = []
    monkeypatch.setattr(M, "segment_only", lambda: called.append(True))
    monkeypatch.setattr(sys, "argv", ["chunking_decode.py", "segment-only"])

    M.main()

    assert called == [True]
