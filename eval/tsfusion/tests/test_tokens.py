import math

from eval.controlled_eval.scoring import wtoks
from eval.tsfusion.tokens import (raw_word_to_tokens, soniox_timed_tokens,
                                  soniox_words, transfer_timestamps,
                                  whisper_timed_tokens)


def piece(text, s, e, c=1.0):
    return {"text": text, "start_ms": s, "end_ms": e, "confidence": c,
            "is_final": True}


# ------------------------------------------------------------- subword grouping
def test_subword_pieces_regroup_into_words():
    # verbatim shape from win_vrilissia_apr1_2_2026_1945951
    # subword shapes of the kind stt-rt-v4 emits; the words are invented
    pieces = [piece("Έ", 120, 180), piece("νας", 240, 300), piece(" απο", 360, 420),
              piece(" τ", 420, 480), piece("ους", 540, 600), piece("δυο", 600, 660)]
    words = soniox_words(pieces)
    assert [w["raw"] for w in words] == ["Ένας", "απο", "τουςδυο"]
    assert math.isclose(words[0]["start"], 0.120)
    assert math.isclose(words[0]["end"], 0.300)
    assert math.isclose(words[2]["start"], 0.420)
    assert math.isclose(words[2]["end"], 0.660)


def test_word_confidence_is_the_minimum_over_its_pieces():
    pieces = [piece(" θ", 0, 100, 0.99), piece("αλασ", 100, 200, 0.41),
              piece("σα", 200, 300, 1.0)]
    assert math.isclose(soniox_words(pieces)[0]["conf"], 0.41)


def test_token_count_equals_wtoks_of_the_joined_text():
    pieces = [piece("Έ", 0, 60), piece("νας", 60, 120), piece(" με", 120, 180),
              piece(" θαλασσα", 180, 300), piece(".", 300, 320),
              piece(" Το", 400, 460)]
    text = "".join(p["text"] for p in pieces)
    toks = soniox_timed_tokens(pieces)
    assert [t.token for t in toks] == wtoks(text)


# ------------------------------------------------------- normalisation edge cases
def test_a_punctuation_only_word_produces_no_token_and_no_shift():
    assert raw_word_to_tokens(".", 1.0, 1.1) == []
    assert raw_word_to_tokens("...", 1.0, 1.1) == []
    assert raw_word_to_tokens(" ,", 1.0, 1.1) == []


def test_a_word_that_expands_splits_its_interval_monotonically():
    # Greek apostrophe elision: one decoder word, two \w+ tokens
    out = raw_word_to_tokens("απ'το", 10.0, 10.5)
    assert [t.token for t in out] == ["απ", "το"]
    assert out[0].start == 10.0 and out[-1].end == 10.5
    assert out[0].end == out[1].start
    assert out[0].split_of == 2 and out[1].split_index == 1
    # the typographic apostrophe behaves the same way
    assert [t.token for t in raw_word_to_tokens("απ’το", 0, 1)] == ["απ", "το"]


def test_tonos_and_case_are_folded_and_final_sigma_is_positional():
    (t,) = raw_word_to_tokens(" Θάλασσα.", 0.0, 1.0)
    assert t.token == "θαλασσα"
    # `scoring.norm` does not FOLD final sigma (ς and σ stay distinct tokens), but
    # str.lower() places it by position, so an all-caps word and its mixed-case twin
    # normalise to the same token. Checked because a mismatch here would silently
    # cost a match in `transfer_timestamps`.
    assert raw_word_to_tokens("ΔΗΜΟΣ", 0, 1)[0].token == "δημος"
    assert raw_word_to_tokens("δήμος", 0, 1)[0].token == "δημος"
    # a lone capital sigma has no following letter, so it lowercases to σ, not ς
    assert raw_word_to_tokens("Σ", 0, 1)[0].token == "σ"


def test_a_trailing_punctuation_mark_does_not_become_its_own_token():
    out = raw_word_to_tokens(" πρωινο:", 0.0, 1.0)
    assert len(out) == 1 and out[0].token == "πρωινο"


def test_whisper_words_carry_probability_through():
    segs = [{"words": [{"w": "Ένας", "s": 0.0, "e": 0.24, "p": 0.67},
                       {"w": " θάλασσα.", "s": 1.36, "e": 2.2, "p": 0.99}]}]
    toks = whisper_timed_tokens(segs, t_offset=1945.951)
    assert [t.token for t in toks] == ["ενας", "θαλασσα"]
    assert math.isclose(toks[0].start, 1945.951)
    assert math.isclose(toks[1].end, 1948.151)
    assert toks[1].conf == 0.99


# ------------------------------------------------------------ timestamp transfer
def test_transfer_only_lands_on_identical_tokens():
    timed = whisper_timed_tokens(
        [{"words": [{"w": "το", "s": 0.0, "e": 0.2},
                    {"w": "καλό", "s": 0.2, "e": 0.6},
                    {"w": "σπίτι", "s": 0.6, "e": 1.0}]}])
    # the benchmark decode of the SAME audio substituted the middle word
    got = transfer_timestamps(timed, ["το", "κακο", "σπιτι"])
    assert got.ops == ["stable", "unmatched", "stable"]
    assert got.intervals[1] is None
    assert got.n_stable == 2
    assert math.isclose(got.intervals[2].start, 0.6)


def test_transfer_leaves_an_inserted_target_token_unobserved():
    timed = whisper_timed_tokens(
        [{"words": [{"w": "ένα", "s": 0.0, "e": 0.3},
                    {"w": "πρωινό", "s": 0.3, "e": 0.9}]}])
    got = transfer_timestamps(timed, ["ενα", "μεγαλο", "πρωινο"])
    assert got.intervals[1] is None
    assert [t.token if t else None for t in got.intervals] == \
        ["ενα", None, "πρωινο"]


def test_repeated_words_are_reported_ambiguous_not_tie_broken():
    # the case a single Levenshtein backtrace gets silently wrong
    timed = whisper_timed_tokens(
        [{"words": [{"w": "ναι", "s": 0.0, "e": 0.2},
                    {"w": "όχι", "s": 0.2, "e": 0.5},
                    {"w": "ναι", "s": 0.5, "e": 0.8}]}])
    got = transfer_timestamps(timed, ["ναι"])
    assert got.ops == ["ambiguous"]
    assert got.intervals[0] is None
    # both possible partners are kept, so the page can show the spread
    assert [round(c.start, 3) for c in got.candidates[0]] == [0.0, 0.5]


def test_an_unambiguous_repeat_still_transfers():
    timed = whisper_timed_tokens(
        [{"words": [{"w": "ναι", "s": 0.0, "e": 0.2},
                    {"w": "όχι", "s": 0.2, "e": 0.5},
                    {"w": "ναι", "s": 0.5, "e": 0.8}]}])
    got = transfer_timestamps(timed, ["ναι", "οχι", "ναι"])
    assert got.ops == ["stable"] * 3
    assert math.isclose(got.intervals[2].start, 0.5)


def test_transfer_is_index_safe_when_a_timed_word_expanded():
    # "απ'το" is one timed word but two MSA tokens; the target has them separately
    timed = whisper_timed_tokens(
        [{"words": [{"w": "και", "s": 0.0, "e": 0.2},
                    {"w": " απ'το", "s": 0.2, "e": 0.8},
                    {"w": " σπίτι", "s": 0.8, "e": 1.2}]}])
    got = transfer_timestamps(timed, ["και", "απ", "το", "σπιτι"])
    assert got.ops == ["stable"] * 4
    assert math.isclose(got.intervals[3].start, 0.8)
    # the split token keeps the whole raw word as its uncertainty envelope
    assert got.intervals[1].provenance == "derived_within_raw_word"
    assert got.intervals[1].envelope == (0.2, 0.8)
    assert got.intervals[3].provenance == "observed_word"


def test_empty_streams_do_not_crash():
    assert transfer_timestamps([], ["α", "β"]).n_stable == 0
    assert transfer_timestamps([], ["α", "β"]).ops == ["unmatched", "unmatched"]
    assert transfer_timestamps(whisper_timed_tokens([]), []).n_target == 0


def test_punctuation_only_piece_does_not_drag_word_confidence_down():
    pieces = [piece(" ναι", 0, 200, 0.98), piece(",", 200, 220, 0.11)]
    (w,) = soniox_words(pieces)
    assert math.isclose(w["conf"], 0.98)
    assert math.isclose(w["conf_all"], 0.11)
