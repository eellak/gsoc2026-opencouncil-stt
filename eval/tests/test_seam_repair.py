"""Tests for exp-2026-08-20-seam-repair. Written before any arm was decoded."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval import seam_repair as S
from eval import chunking_decode as CD

HOP = S.VAD_HOP_SAMPLES / S.SAMPLE_RATE   # 0.032 s


def probs_from_spec(n_frames: int, quiet: list[tuple[int, int, float]]) -> np.ndarray:
    p = np.full(n_frames, 0.9)
    for begin, end, value in quiet:
        p[begin:end] = value
    return p


# ------------------------------------------------------------------ arm F: valleys


def test_three_quiet_frames_is_not_a_valley_but_four_is():
    n = int(round(30 / HOP)) + 4
    energies = np.ones(n)
    deadline, cursor = 29.5, 0.0
    at = int(28.5 / HOP)
    short = probs_from_spec(n, [(at, at + 3, 0.05)])
    assert S.choose_valley_cut(short, energies, cursor, deadline) is None
    long = probs_from_spec(n, [(at, at + 4, 0.05)])
    result = S.choose_valley_cut(long, energies, cursor, deadline)
    assert result is not None
    _, diag = result
    assert diag["n_frames"] == 4
    assert diag["duration_sec"] == pytest.approx(4 * HOP)


def test_probability_exactly_at_the_threshold_is_not_quiet():
    n = int(round(30 / HOP)) + 4
    at = int(28.5 / HOP)
    exact = probs_from_spec(n, [(at, at + 8, S.VALLEY_PROB_THRESHOLD)])
    assert S.choose_valley_cut(exact, np.ones(n), 0.0, 29.5) is None
    below = probs_from_spec(n, [(at, at + 8, S.VALLEY_PROB_THRESHOLD - 1e-6)])
    assert S.choose_valley_cut(below, np.ones(n), 0.0, 29.5) is not None


def test_a_valley_crossing_the_slack_edge_contributes_its_frames_inside():
    """Restricting to the slack region first, not rejecting straddlers."""
    n = int(round(30 / HOP)) + 4
    lower = 29.5 - S.SLACK_SECONDS
    at = int((lower - 0.2) / HOP)          # starts before the slack region
    probs = probs_from_spec(n, [(at, at + 20, 0.05)])
    result = S.choose_valley_cut(probs, np.ones(n), 0.0, 29.5)
    assert result is not None
    cut, diag = result
    assert diag["start_sec"] >= lower - 1e-6
    assert lower - 1e-6 <= cut <= 29.5 + 1e-6


def test_no_valley_returns_none_and_the_caller_cuts_blind():
    n = int(round(30 / HOP)) + 4
    probs = np.full(n, 0.9)
    assert S.choose_valley_cut(probs, np.ones(n), 0.0, 29.5) is None
    duration = 50.0          # one boundary: 50 - 29.5 = 20.5, under the ceiling
    pieces, counters = S.split_with_valley_repair(duration, [], probs, np.ones(n))
    assert counters["blind_cuts"] == 1
    assert counters["valley_cuts"] == 0
    assert [p.end_kind for p in pieces] == [S.BLIND, S.AUDIO_END]


def test_lowest_mean_probability_wins_over_a_later_shallower_valley():
    n = int(round(30 / HOP)) + 4
    early = int(28.0 / HOP)
    late = int(29.0 / HOP)
    probs = probs_from_spec(n, [(early, early + 6, 0.02), (late, late + 6, 0.30)])
    cut, diag = S.choose_valley_cut(probs, np.ones(n), 0.0, 29.5)
    assert diag["mean_speech_prob"] == pytest.approx(0.02)


def test_energy_only_breaks_a_probability_tie():
    n = int(round(30 / HOP)) + 4
    a, b = int(28.0 / HOP), int(29.0 / HOP)
    probs = probs_from_spec(n, [(a, a + 6, 0.10), (b, b + 6, 0.10)])
    energies = np.ones(n)
    energies[b:b + 6] = 0.001            # the later valley is quieter
    cut, diag = S.choose_valley_cut(probs, energies, 0.0, 29.5)
    assert diag["start_sec"] == pytest.approx(b * HOP)
    energies[b:b + 6] = 1.0
    energies[a:a + 6] = 0.001            # now the earlier one is quieter
    cut, diag = S.choose_valley_cut(probs, energies, 0.0, 29.5)
    assert diag["start_sec"] == pytest.approx(a * HOP)


def test_a_full_probability_and_energy_tie_breaks_to_the_later_valley():
    n = int(round(30 / HOP)) + 4
    a, b = int(28.0 / HOP), int(29.0 / HOP)
    probs = probs_from_spec(n, [(a, a + 6, 0.10), (b, b + 6, 0.10)])
    cut, diag = S.choose_valley_cut(probs, np.ones(n), 0.0, 29.5)
    assert diag["start_sec"] == pytest.approx(b * HOP)


def test_the_cut_never_leaves_a_piece_shorter_than_the_minimum():
    n = int(round(10 / HOP)) + 4
    probs = np.full(n, 0.05)
    cut, _ = S.choose_valley_cut(probs, np.ones(n), 0.0, 4.0, min_chunk=5.0) or (None, None)
    assert cut is None            # the whole legal range is below the minimum


def test_nonzero_cursor_searches_the_slack_before_its_own_deadline():
    n = int(round(90 / HOP)) + 4
    at = int(58.0 / HOP)
    probs = probs_from_spec(n, [(at, at + 8, 0.05)])
    cut, diag = S.choose_valley_cut(probs, np.ones(n), 29.5, 59.0)
    assert 57.0 <= cut <= 59.0


# ------------------------------------------------- arm F: splitter and provenance


def test_split_with_no_probs_reproduces_arm_p_pieces_exactly():
    """The differential: with no forced branch, F is P."""
    duration = 148.7
    silences = [(20.0, 21.5), (44.0, 46.0), (70.0, 71.0), (95.0, 97.0), (120.0, 121.0)]
    p_pieces, _ = CD.split_accumulating(duration, silences)
    f_pieces, counters = S.split_with_valley_repair(duration, silences)
    assert [round(p.end, 6) for p in p_pieces] == [round(p.end, 6) for p in f_pieces]
    assert counters["blind_cuts"] == 0


def test_boundary_kinds_account_for_every_boundary_and_are_recomputed():
    duration = 148.7
    silences = [(20.0, 21.5), (44.0, 46.0), (70.0, 71.0), (95.0, 97.0), (120.0, 121.0)]
    pieces, counters = S.split_with_valley_repair(duration, silences)
    assert (counters["vad_silence_cuts"] + counters["valley_cuts"]
            + counters["blind_cuts"]) == len(pieces) - 1
    assert pieces[-1].end_kind == S.AUDIO_END


def test_repaired_cut_replaces_the_forced_cut_and_is_recorded_as_such():
    duration = 50.0
    n = int(round(duration / HOP)) + 4
    at = int(28.6 / HOP)
    probs = probs_from_spec(n, [(at, at + 8, 0.05)])
    pieces, counters = S.split_with_valley_repair(duration, [], probs, np.ones(n))
    assert counters["blind_cuts"] == 0
    assert counters["valley_cuts"] == 1
    assert pieces[0].end_kind == S.PROBABILITY_VALLEY
    assert pieces[0].detail["mean_speech_prob"] < S.VALLEY_PROB_THRESHOLD
    assert pieces[0].detail["duration_sec"] >= 0.1


@pytest.mark.parametrize("duration", [31.0, 59.0, 60.5, 90.0, 136.8, 148.7, 149.9])
def test_pieces_are_legal_and_tile_the_sample_grid_exactly(duration):
    n = int(round(duration / HOP)) + 4
    rng = np.random.default_rng(11)
    probs = rng.random(n)
    silences = [(t, t + 0.8) for t in np.arange(17.0, duration, 23.0)]
    pieces, counters = S.split_with_valley_repair(duration, silences, probs, np.ones(n))
    assert counters["tiny_chunks"] == 0
    assert counters["overlong_chunks"] == 0
    n_samples = int(round(duration * S.SAMPLE_RATE))
    slices = S.sample_boundaries(pieces, n_samples)
    assert slices[0][0] == 0 and slices[-1][1] == n_samples
    assert sum(e - b for b, e in slices) == n_samples
    for a, b in zip(slices, slices[1:]):
        assert a[1] == b[0]


def test_audio_length_exactly_divisible_by_the_vad_hop_is_accepted():
    n_samples = S.VAD_HOP_SAMPLES * 3000
    audio = np.zeros(n_samples, dtype="float32")
    fake = lambda padded: np.full(len(padded) // S.VAD_HOP_SAMPLES, 0.1)
    probs = S.speech_probabilities(audio, model=fake)
    assert probs.shape[0] == 3000


def test_non_finite_probabilities_are_refused():
    audio = np.zeros(S.VAD_HOP_SAMPLES * 10, dtype="float32")
    fake = lambda padded: np.array([np.nan] * (len(padded) // S.VAD_HOP_SAMPLES))
    with pytest.raises(ValueError):
        S.speech_probabilities(audio, model=fake)


# ------------------------------------------------------------------ arm C: prompts


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """One token per three characters, so a cap is reachable in tests."""

    def encode(self, text, add_special_tokens=False):
        return FakeEncoding(list(range(max(1, (len(text) + 2) // 3))))


def test_prompt_is_empty_when_there_is_no_preceding_text():
    assert S.prompt_ids_for(FakeTokenizer(), "") == []
    assert S.prompt_ids_for(FakeTokenizer(), "   ") == []


def test_prompt_never_exceeds_the_token_cap():
    text = " ".join(["λέξη"] * 200)
    ids = S.prompt_ids_for(FakeTokenizer(), text, max_tokens=64)
    assert 0 < len(ids) <= 64


def test_prompt_truncation_lands_on_whole_words():
    words = [f"λέξη{i}" for i in range(40)]
    text = " ".join(words)
    tokenizer = FakeTokenizer()
    ids = S.prompt_ids_for(tokenizer, text, max_tokens=64)
    # the chosen suffix must itself re-encode to the same ids
    for take in range(1, len(words) + 1):
        candidate = " " + " ".join(words[-take:])
        if len(tokenizer.encode(candidate).ids) == len(ids):
            assert tokenizer.encode(candidate).ids == ids
            break
    else:
        pytest.fail("the prompt is not the encoding of a whole-word suffix")


def test_exactly_at_and_one_over_the_cap():
    class Fixed:
        def __init__(self, per_word):
            self.per_word = per_word

        def encode(self, text, add_special_tokens=False):
            return FakeEncoding([0] * (self.per_word * len(text.split())))

    text = " ".join(["w"] * 100)
    assert len(S.prompt_ids_for(Fixed(1), text, max_tokens=64)) == 64
    assert len(S.prompt_ids_for(Fixed(65), text, max_tokens=64)) == 0
    assert len(S.prompt_ids_for(Fixed(2), text, max_tokens=64)) == 64


def test_real_greek_prompt_uses_the_model_tokenizer_when_present():
    tokenizer_file = Path(S.MODEL_DIR) / "tokenizer.json"
    if not tokenizer_file.exists():
        pytest.skip("artifact-ct2-fixed is not on this machine")
    import tokenizers

    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    text = "Το λόγο έχει ο κύριος δήμαρχος για το θέμα της ημερήσιας διάταξης. " * 8
    ids = S.prompt_ids_for(tokenizer, text)
    assert 0 < len(ids) <= S.PROMPT_MAX_TOKENS
    decoded = tokenizer.decode(ids)
    assert decoded.strip()
    assert decoded.strip().split()[0] in text.split()


def test_prompt_probe_records_every_get_prompt_call_and_restores_the_method():
    class FakeModel:
        def get_prompt(self, tokenizer, previous_tokens, **kwargs):
            return [tokenizer.sot_prev, *previous_tokens, *tokenizer.sot_sequence]

    model = FakeModel()
    original = model.get_prompt
    tok = SimpleNamespace(sot_prev=50361, sot_sequence=[50258, 50288, 50360])
    with S.prompt_probe(model) as calls:
        model.get_prompt(tok, [1, 2, 3])
        model.get_prompt(tok, [])
    assert [c["previous_tokens"] for c in calls] == [[1, 2, 3], []]
    assert calls[0]["prompt"][0] == 50361
    assert model.get_prompt.__func__ is original.__func__


# ------------------------------------------------------------- arm W: ownership


def test_tile_edges_use_the_frozen_closed_lower_convention():
    windows = CD.overlap_layout(60.0)
    assert S.owning_window_index(0.0, len(windows)) == 0
    assert S.owning_window_index(14.99, len(windows)) == 0
    assert S.owning_window_index(15.00, len(windows)) == 0     # tie -> earlier window
    assert S.owning_window_index(15.01, len(windows)) == 1
    assert S.owning_window_index(30.00, len(windows)) == 1
    assert S.owning_window_index(30.01, len(windows)) == 2
    assert S.owning_window_index(60.0, len(windows)) == 3


def _fake_segment(words):
    return {"text": "".join(w[0] for w in words), "start": words[0][1],
            "end": words[-1][2],
            "words": [{"word": w[0], "start": w[1], "end": w[2]} for w in words]}


def test_word_ownership_is_an_if_and_only_if_and_keeps_something():
    duration = 45.0
    windows = CD.overlap_layout(duration)
    candidates = []
    for window in windows:
        # each window transcribes its whole 30 s span, in window-local time
        words = [(" w%d" % i, i * 1.0, i * 1.0 + 0.5) for i in range(30)]
        candidates.extend(S.word_candidates(window, [_fake_segment(words)]))
    merged = S.merge_words_by_tile(candidates, windows, duration)
    assert merged["kept"] > 0
    assert merged["kept"] + merged["non_owner"] + merged["outside_audio"] == len(candidates)
    for candidate in candidates:
        mid = candidate["midpoint"]
        inside = -1e-7 <= mid <= duration + 1e-7
        owned = inside and S.owning_window_index(
            min(max(mid, 0.0), duration), len(windows)) == candidate["window_index"]
        present = any(
            abs(w["start"] - candidate["start"]) < 1e-9
            and w["text"] == candidate["text"] for w in merged["words"]
        )
        assert present == owned          # outside-audio words are absent too


def test_words_outside_the_audio_are_counted_separately_from_seam_drops():
    duration = 45.0
    windows = CD.overlap_layout(duration)
    first = windows[0]
    # the first window starts at -7.5 s; a word at local 0.1 s is before the audio
    hallucination = _fake_segment([(" ghost", 0.05, 0.15)])
    candidates = S.word_candidates(first, [hallucination])
    merged = S.merge_words_by_tile(candidates, windows, duration)
    assert merged["outside_audio"] == 1
    assert merged["non_owner"] == 0
    assert merged["kept"] == 0


def test_zero_duration_and_equal_timestamp_words_keep_source_order():
    duration = 45.0
    windows = CD.overlap_layout(duration)
    window = windows[1]
    local = 20.0 - window.start          # audio time 20 s, owned by window 1
    words = [(" beta", local, local), (" alpha", local, local)]
    candidates = S.word_candidates(window, [_fake_segment(words)])
    merged = S.merge_words_by_tile(candidates, windows, duration)
    assert [w["text"] for w in merged["words"]] == [" beta", " alpha"]


def test_seam_diagnostic_sees_a_duplicate_and_a_hole():
    duration = 45.0
    windows = CD.overlap_layout(duration)
    left, right = windows[0], windows[1]
    left_words = _fake_segment([
        (" ενα", 15.0 - left.start - 0.6, 15.0 - left.start - 0.3),
        (" δυο", 15.0 - left.start - 0.2, 15.0 - left.start + 0.1),
    ])
    right_words = _fake_segment([
        (" δυο", 15.0 - right.start - 0.2, 15.0 - right.start + 0.1),
        (" τρια", 15.0 - right.start + 0.3, 15.0 - right.start + 0.6),
    ])
    candidates = (S.word_candidates(left, [left_words])
                  + S.word_candidates(right, [right_words]))
    report = S.seam_disagreement(candidates, windows, duration)
    seam = report["seams"][0]
    assert seam["agreed"] >= 1            # "δυο" is seen by both
    assert seam["only_left"] >= 1         # "ενα" only on the left
    assert seam["only_right"] >= 1        # "τρια" only on the right
    assert report["disagreement_rate"] > 0


def test_a_merge_that_drops_everything_fails_the_pilot_oracle():
    """The original criterion was not red-capable; this proves the new one is."""
    duration = 45.0
    windows = CD.overlap_layout(duration)
    raw = []
    for window in windows:
        raw.append({
            "index": window.index,
            # window-local 0.05 s; every window's own start pushes this before
            # its kept tile, and window 0's lands before the audio entirely
            "segments": [_fake_segment([(" w", 0.02, 0.05)])],
        })
    record = {"audio_seconds": duration, "raw_windows": raw,
              "kept": 0, "seam_disagreement": {"disagreement_rate": 0.0}}
    verdict = S._w_verdict({"windows": {"win": record}})
    assert verdict["verdict"] == "FAIL"
    assert any("kept no words" in f for f in verdict["failures"])


def test_derived_e_wt_costs_no_decoding_and_reuses_the_same_raw_windows():
    duration = 45.0
    windows = CD.overlap_layout(duration)
    raw = []
    for window in windows:
        raw.append({
            "index": window.index,
            "start_sec": window.start, "end_sec": window.end,
            "centre_sec": window.centre,
            "kept_start_sec": window.kept_start, "kept_end_sec": window.kept_end,
            "segments": [
                {"text": " kappa", "start": 15.0, "end": 15.5,
                 "words": [{"word": " kappa", "start": 15.0, "end": 15.5}]}
            ],
        })
    record = {"audio_seconds": duration, "raw_windows": raw,
              "config": dict(S.CONTROL, word_timestamps=True)}
    derived = S.derive_e_wt(record)
    assert derived["derived_from"] == "W"
    assert isinstance(derived["text"], str)


def test_a_derived_arm_rebuilds_the_layout_from_the_unrounded_duration():
    """`audio_seconds` is rounded for reading. Rebuilding the layout from it moves the
    last tile, so the geometry is carried at full precision and the rounded value is
    only a fallback for records written before the key existed."""
    duration = 45.004                      # rounds to 45.0: one tile fewer
    assert len(CD.overlap_layout(duration)) != len(CD.overlap_layout(round(duration, 2)))
    windows = CD.overlap_layout(duration)
    raw = [{"index": w.index,
            "start_sec": w.start, "end_sec": w.end, "centre_sec": w.centre,
            "kept_start_sec": w.kept_start, "kept_end_sec": w.kept_end,
            "segments": [{"text": " kappa", "start": 15.0, "end": 15.5,
                          "words": [{"word": " kappa", "start": 15.0, "end": 15.5}]}]}
           for w in windows]
    record = {"audio_seconds": round(duration, 2), "geometry_seconds": duration,
              "raw_windows": raw, "config": dict(S.CONTROL, word_timestamps=True)}
    derived = S.derive_e_wt(record)        # would raise on a layout of the wrong length
    assert derived["geometry_seconds"] == duration
    assert S._geometry_seconds({"audio_seconds": 45.0}) == 45.0


def test_joining_a_segments_words_reproduces_that_segments_own_text():
    """Guards the P-vs-W text-assembly asymmetry Codex raised."""
    from eval.controlled_eval.eval_freeze import ftoks

    segment = _fake_segment([(" καλησπέρα", 1.0, 1.6), (" σας", 1.7, 1.9)])
    assert ftoks("".join(w["word"] for w in segment["words"])) == ftoks(segment["text"])


# ---------------------------------------------------------------- cache identity


def test_cache_with_a_different_algorithm_manifest_is_refused(monkeypatch, tmp_path):
    env = {"python": "3.12.3", "device": "cpu"}
    fresh = S._fresh_state("F", S.MODEL_SHA256_16, env, "eval")
    stale = dict(fresh, algorithm_manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="algorithm_manifest_sha256"):
        S.validate_cache_identity(stale, fresh, tmp_path / "eval-F.json")


def test_cache_written_for_another_arm_or_set_is_refused(tmp_path):
    env = {"python": "3.12.3", "device": "cpu"}
    fresh = S._fresh_state("F", S.MODEL_SHA256_16, env, "eval")
    with pytest.raises(ValueError, match="arm"):
        S.validate_cache_identity(
            S._fresh_state("C", S.MODEL_SHA256_16, env, "eval"), fresh, tmp_path)
    with pytest.raises(ValueError, match="set"):
        S.validate_cache_identity(
            S._fresh_state("F", S.MODEL_SHA256_16, env, "pilot"), fresh, tmp_path)


def test_each_arms_manifest_covers_that_arms_frozen_constants():
    for key in ("slack_seconds", "valley_prob_threshold", "min_valley_frames",
                "max_chunk_seconds", "min_chunk_seconds"):
        assert key in S.algorithm_manifest("F")
    for key in ("prompt_max_tokens", "prompt_carry_join"):
        assert key in S.algorithm_manifest("C")
    for key in ("merge_w", "stride_seconds", "window_seconds", "seam_band_seconds"):
        assert key in S.algorithm_manifest("W")
    for arm in S.ARMS:
        assert S.algorithm_manifest_sha256(arm) == S.algorithm_manifest_sha256(arm)
    assert len({S.algorithm_manifest_sha256(a) for a in S.ARMS}) == len(S.ARMS)


def test_one_arms_rule_change_does_not_invalidate_another_arms_cache():
    """Arm C's prompt-join rule must not appear in F's or W's identity."""
    assert "prompt_carry_join" not in S.algorithm_manifest("F")
    assert "prompt_carry_join" not in S.algorithm_manifest("W")


def test_carried_prompt_text_never_fuses_two_pieces_into_one_word():
    tokenizer = FakeTokenizer()
    carried = ""
    for text in ("γεια σου", "κόσμε"):     # no leading space on the second piece
        carried = " ".join(p for p in (carried.strip(), text.strip()) if p)
    assert carried.split() == ["γεια", "σου", "κόσμε"]
    assert S.prompt_ids_for(tokenizer, carried)


# --------------------------------------------------------------------- scoring


def test_an_arm_scored_against_itself_is_exactly_zero():
    rows = [
        {"window_id": "a", "meeting_id": "m1", "reference_text": "ένα δύο τρία"},
        {"window_id": "b", "meeting_id": "m2", "reference_text": "τέσσερα πέντε"},
    ]
    state = {"windows": {"a": {"text": "ένα δύο τέσσερα"}, "b": {"text": "τέσσερα"}}}
    scores = CD.per_window_scores(state, rows)
    contrast = S._contrast(scores, scores, rows)
    for metric, value in contrast["vs"].items():
        assert value["delta"] == pytest.approx(0.0)
    assert contrast["leave_one_meeting_out_deletion"]["delta"] == pytest.approx(0.0)


def test_leave_one_meeting_out_removes_whole_meetings():
    rows = [
        {"window_id": "a", "meeting_id": "m1", "reference_text": "ένα δύο"},
        {"window_id": "b", "meeting_id": "m1", "reference_text": "τρία τέσσερα"},
        {"window_id": "c", "meeting_id": "m2", "reference_text": "πέντε έξι"},
    ]
    arm = {"windows": {"a": {"text": "ένα"}, "b": {"text": "τρία"}, "c": {"text": "πέντε έξι"}}}
    base = {"windows": {"a": {"text": "ένα δύο"}, "b": {"text": "τρία τέσσερα"},
                        "c": {"text": "πέντε έξι"}}}
    loo = S._loo(CD.per_window_scores(arm, rows), CD.per_window_scores(base, rows),
                 rows, "meeting_id")
    assert set(loo["without"]) == {"m1", "m2"}


def test_extended_rates_report_s_plus_d_over_n_and_emitted_tokens():
    rows = [{"window_id": "a", "meeting_id": "m1", "reference_text": "ένα δύο τρία"}]
    scores = CD.per_window_scores({"windows": {"a": {"text": "ένα δύο τρία τέσσερα"}}}, rows)
    rates = S._extended_rates(scores)
    assert rates["emitted_tokens"] == 4
    assert rates["correct_error_rate_s_plus_d_over_n"] == pytest.approx(0.0)


def test_pilot_windows_are_the_five_with_the_most_forced_cuts_in_arm_p():
    forced = S.p_forced_cuts()
    if not forced:
        pytest.skip("the arm P cache is not on this machine")
    ranked = sorted(((-count, wid) for wid, count in forced.items() if count))[:5]
    assert [wid for _, wid in ranked] == list(S.PILOT_WINDOW_IDS)


# --------------------------------------------- arm X: all three mechanisms at once


def test_x_core_ownership_is_half_open_with_a_closed_final_edge():
    cores = [(0.0, 25.5), (25.5, 51.0), (51.0, 60.0)]
    assert S.owning_core_index(0.0, cores) == 0
    assert S.owning_core_index(25.49, cores) == 0
    assert S.owning_core_index(25.5, cores) == 1      # half-open: the later core
    assert S.owning_core_index(51.0, cores) == 2
    assert S.owning_core_index(60.0, cores) == 2      # the last edge is closed
    assert S.owning_core_index(-0.5, cores) is None
    assert S.owning_core_index(61.0, cores) is None


def test_x_margin_words_are_dropped_not_double_counted():
    """A word inside a neighbour's margin belongs to the neighbour's core."""
    cores = [(0.0, 25.5), (25.5, 50.0)]
    candidates = [
        # piece 0 also transcribes 2 s past its core; that word is core 1's
        {"text": " ξένη", "start": 26.0, "end": 26.4, "midpoint": 26.2,
         "piece_index": 0, "segment_index": 0, "word_index": 0},
        {"text": " ξένη", "start": 26.0, "end": 26.4, "midpoint": 26.2,
         "piece_index": 1, "segment_index": 0, "word_index": 0},
        {"text": " δική", "start": 10.0, "end": 10.4, "midpoint": 10.2,
         "piece_index": 0, "segment_index": 0, "word_index": 1},
    ]
    merged = S.merge_words_by_core(candidates, cores)
    assert merged["kept"] == 2
    assert merged["non_owner"] == 1
    assert [w["text"] for w in merged["words"]] == [" δική", " ξένη"]


def test_x_decoder_input_always_fits_one_whisper_window():
    """core + lookback + lookahead must stay under 30 s, or X is impossible."""
    total = (S.X_MAX_CHUNK_SECONDS + S.X_LOOKBACK_SECONDS + S.X_LOOKAHEAD_SECONDS)
    assert total <= S.WINDOW_SECONDS


def test_x_margin_is_asymmetric_because_the_two_sides_are_not_used_alike():
    """Measured: the left piece over-runs its core, the right barely looks back."""
    assert S.X_LOOKAHEAD_SECONDS > S.X_LOOKBACK_SECONDS


def test_x_last_input_check_does_not_fire_on_the_rounded_audio_seconds():
    """audio_seconds is stored rounded to 2 dp; the check must use the exact core end."""
    exact = 148.688
    piece = {"piece_index": 0, "core_start_sec": 0.0, "core_end_sec": exact,
             "input_start_sec": 0.0, "input_end_sec": exact,
             "segments": [{"text": " x", "start": 0.1, "end": 0.2,
                           "words": [{"word": " x", "start": 0.1, "end": 0.2}]}]}
    state = {"windows": {"win": {
        "audio_seconds": round(exact, 2),          # 148.69, larger than the truth
        "counters": {"blind_cuts": 0, "valley_cuts": 0, "tiny_chunks": 0,
                     "overlong_chunks": 0},
        "pieces": [{"sample_begin": 0, "sample_end": 1, "end_kind": S.AUDIO_END,
                    "detail": None}],
        "prompts": [], "get_prompt_calls": [], "kept": 1, "raw_pieces": [piece],
    }}}
    verdict = S._x_verdict(state, {})
    assert not any("stops before the audio end" in f for f in verdict["failures"])


def test_x_verdict_fails_when_a_seam_is_decoded_without_lookahead():
    """The new criterion must be able to fail, or it proves nothing."""
    piece = lambda i, lb, la: {
        "piece_index": i, "core_start_sec": i * 27.0, "core_end_sec": (i + 1) * 27.0,
        "input_start_sec": i * 27.0 - lb, "input_end_sec": (i + 1) * 27.0 + la,
        "segments": [{"text": " x", "start": 0.1, "end": 0.2,
                      "words": [{"word": " x", "start": 0.1, "end": 0.2}]}],
    }
    starved = {"windows": {"win": {
        "audio_seconds": 54.0, "counters": {"blind_cuts": 0, "valley_cuts": 0,
                                            "tiny_chunks": 0, "overlong_chunks": 0},
        "pieces": [{"sample_begin": 0, "sample_end": 1, "end_kind": S.AUDIO_END,
                    "detail": None}],
        "prompts": [], "get_prompt_calls": [], "kept": 0,
        "raw_pieces": [piece(0, 0.0, 0.0), piece(1, 0.5, 2.0)],
    }}}
    verdict = S._x_verdict(starved, {})
    assert verdict["verdict"] == "FAIL"
    assert any("lookahead" in f for f in verdict["failures"])


def test_x_split_uses_the_shorter_ceiling_and_still_repairs_blind_cuts():
    duration = 42.0
    n = int(round(duration / HOP)) + 4
    at = int(26.1 / HOP)
    probs = probs_from_spec(n, [(at, at + 8, 0.05)])
    pieces, counters = S.split_with_valley_repair(
        duration, [], probs, np.ones(n), max_chunk=S.X_MAX_CHUNK_SECONDS)
    assert counters["blind_cuts"] == 0
    assert counters["valley_cuts"] == 1
    assert all(p.duration <= S.X_MAX_CHUNK_SECONDS + 1e-7 for p in pieces)


def test_x_manifest_names_all_three_mechanisms():
    manifest = S.algorithm_manifest("X")
    assert manifest["splitter"] == "split_with_valley_repair@1"     # F
    assert manifest["prompt_max_tokens"] == S.PROMPT_MAX_TOKENS     # C
    assert manifest["merge_x"] == "word_core_ownership@1"           # W
    assert manifest["lookahead_seconds"] == S.X_LOOKAHEAD_SECONDS
    assert manifest["lookback_seconds"] == S.X_LOOKBACK_SECONDS
    assert S.algorithm_manifest_sha256("X") != S.algorithm_manifest_sha256("F")
