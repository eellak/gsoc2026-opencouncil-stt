import math

from eval.tsfusion import coords as C


def test_page_span_is_the_advertised_299_seconds():
    assert math.isclose(C.PAGE_DURATION, 299.287, abs_tol=1e-6)
    assert math.isclose(C.T0, 1945.951)
    assert math.isclose(C.T1, 2245.238)


def test_round_trip_is_exact_enough():
    for t in (1945.951, 2000.0, 2096.065, 2096.234, 2245.238):
        assert math.isclose(C.to_absolute(C.to_page(t)), t, abs_tol=1e-9)


def test_known_boundaries():
    # window A starts the page
    assert math.isclose(C.to_page(1945.951), 0.0, abs_tol=1e-9)
    # window A ends
    assert math.isclose(C.to_page(2096.065), 150.114, abs_tol=1e-9)
    # window B starts 169 ms later in absolute time, and the page audio is continuous
    assert math.isclose(C.to_page(2096.234), 150.283, abs_tol=1e-9)
    # the page ends
    assert math.isclose(C.to_page(2245.238), 299.287, abs_tol=1e-9)


def test_window_local_zero_maps_to_the_window_start():
    a, b = C.WINDOWS
    assert math.isclose(C.window_to_absolute(a.item_id, 0.0), 1945.951)
    assert math.isclose(C.window_to_absolute(b.item_id, 0.0), 2096.234)
    # a word 10 s into window B is 160.283 s into the page, not 160.114
    assert math.isclose(C.window_to_page(b.item_id, 10.0), 160.283, abs_tol=1e-9)


def test_the_seam_is_a_real_hole_and_is_named():
    lo, hi = C.SEAM
    assert math.isclose(hi - lo, 0.169, abs_tol=1e-9)
    assert C.in_seam(2096.1)
    assert not C.in_seam(2096.0)
    assert not C.in_seam(2096.3)
    # no window claims a moment inside the seam
    assert not any(w.contains(2096.15) for w in C.WINDOWS)


def test_chunk_phase_is_measured_from_what_the_decoder_was_fed():
    a, b = C.WINDOWS
    # meeting-clock phase: the wrong quantity, kept only for contrast
    assert math.isclose(C.chunk_phase(1950.0), 0.0, abs_tol=1e-9)
    # the decoder was fed window A starting at 1945.951, so 1945.951 is phase 0
    assert math.isclose(C.chunk_phase(1945.951, a.start), 0.0, abs_tol=1e-9)
    assert math.isclose(C.whisper_phase(1945.951), 0.0, abs_tol=1e-9)
    # ... and the two answers differ by 25.951 s, which is most of a chunk
    assert not math.isclose(C.chunk_phase(1945.951), C.whisper_phase(1945.951))
    # window B restarts the phase clock at its own start, not at the page start
    assert math.isclose(C.whisper_phase(2096.234), 0.0, abs_tol=1e-9)
    assert math.isclose(C.whisper_phase(2126.234), 0.0, abs_tol=1e-9)
    assert math.isclose(C.whisper_phase(2111.234), 15.0, abs_tol=1e-9)


def test_no_decode_phase_inside_the_seam():
    assert C.whisper_phase(2096.15) is None
