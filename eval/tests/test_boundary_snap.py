import numpy as np
import pytest

from eval.controlled_eval.boundary_snap import rms_frames, snap

SR = 16000


def signal(loud_spans, total_sec=2.0):
    samples = np.zeros(int(SR * total_sec), dtype=np.float32)
    for start, end in loud_spans:
        samples[int(start * SR):int(end * SR)] = 0.5
    return samples


def test_rms_frames_are_one_per_frame():
    assert rms_frames(signal([(0.0, 1.0)]), SR).shape[0] == 200
    assert rms_frames(np.zeros(3, dtype=np.float32), SR).size == 0


def test_snap_finds_the_silent_gap_between_two_words():
    samples = signal([(0.0, 0.9), (1.1, 2.0)])
    assert snap(samples, SR, 0.95, 0.8, 1.2, "end") == pytest.approx(1.0, abs=0.03)


def test_snap_moves_outward_when_everything_is_loud():
    samples = signal([(0.0, 2.0)])
    assert snap(samples, SR, 1.0, 0.8, 1.2, "start") == pytest.approx(0.8)
    assert snap(samples, SR, 1.0, 0.8, 1.2, "end") == pytest.approx(1.2)


def test_snap_never_leaves_the_allowed_window():
    samples = signal([(0.0, 0.5), (1.5, 2.0)])
    out = snap(samples, SR, 1.0, 0.9, 1.1, "start")
    assert 0.9 <= out <= 1.1


def test_degenerate_window_returns_the_target():
    assert snap(signal([(0.0, 2.0)]), SR, 1.0, 1.0, 1.0, "start") == 1.0
