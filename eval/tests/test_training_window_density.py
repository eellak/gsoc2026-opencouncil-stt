from eval.training_window_density import (
    aggregate_bin_overlaps,
    histogram,
    interval_bin_overlaps,
    training_aggregate,
    wav_header_fingerprint,
)
import wave


def test_interval_bin_overlaps_respects_frozen_five_second_bins():
    assert interval_bin_overlaps(4.0, 11.0) == [1.0, 5.0, 1.0, 0.0, 0.0, 0.0]


def test_aggregate_bin_overlaps_reports_coverage_and_seconds():
    got = aggregate_bin_overlaps([(0.0, 2.0), (0.0, 7.0)])
    assert got[0]["mean_seconds_per_window"] == 3.5
    assert got[0]["windows_with_any_share"] == 1.0
    assert got[1]["mean_seconds_per_window"] == 1.0
    assert got[1]["windows_with_any_share"] == 0.5


def test_histogram_includes_exact_30_seconds_in_last_bin():
    got = histogram([0.0, 1.5, 29.0, 30.0])
    assert [x["count"] for x in got] == [1, 1, 0, 0, 0, 0, 0, 2]


def test_training_aggregate_separates_raw_aligned_and_encoder_clip_span():
    rows = [{
        "start": 10.0,
        "end": 12.0,
        "start_adj": 9.8,
        "end_adj": 12.2,
        "source": "correction",
    }]
    got = training_aggregate(rows)
    assert got["raw_speech_span_proxy"]["seconds"]["mean"] == 2.0
    assert got["aligned_span"]["seconds"]["mean"] == 2.4
    assert got["intended_clip_audio_before_source_end_clamp"]["seconds"]["mean"] == 2.8
    assert got["raw_speech_span_proxy"]["onset_seconds"]["mean"] == 0.4
    assert got["raw_speech_span_proxy"]["offset_seconds"]["mean"] == 2.4
    assert got["by_source"]["correction"]["rows"] == 1


def test_wav_header_fingerprint_reads_only_named_evaluation_windows(tmp_path):
    for name in ("allowed", "sealed"):
        with wave.open(str(tmp_path / f"{name}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16_000)
            f.writeframes(b"\x00\x00" * 16)
    got = wav_header_fingerprint(tmp_path, ["allowed"])
    assert got["n_wav_headers"] == 1
