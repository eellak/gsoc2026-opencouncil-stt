"""Capture mechanics and oracle DP of the arm D N-best screen, model-free.

What must hold before any decode is trusted:
- the proxy injects num_hypotheses ONLY on the beam path and returns the CT2
  result object unchanged (top-1 path untouched by construction);
- the side channel records all hypotheses;
- the consumed-portion rule replicates `_split_segments_by_timestamps`;
- the chunk-skip rule replicates `generate_segments`;
- the oracle DP is exact: never worse than any fixed choice, equal to the
  frozen sdi total on the text it reconstructs;
- decode state is resumable.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from scripts.serving_stack.nbest_arm import (  # noqa: E402
    NUM_HYPS, CaptureSink, CapturingModel, consumed_segments, oracle_window,
    pending, predicts_skip, transduce, used_text, window_oracle)

TSB = 100          # fake timestamp_begin
EOT = 50           # fake end-of-text


class FakeTokenizer:
    """Maps token id t (< EOT) to ' w<t>'; filters specials like the real one."""
    timestamp_begin = TSB
    eot = EOT

    def decode(self, tokens):
        return "".join(f" w{t}" for t in tokens if t < EOT)


# --------------------------------------------------- consumed-portion rule
def test_unfinished_tail_is_dropped():
    # <|0|> w1 w2 <|5|><|5|> w3 w4          (no closing timestamp on the tail)
    tokens = [TSB + 0, 1, 2, TSB + 5, TSB + 5, 3, 4]
    segs = consumed_segments(tokens, TSB)
    assert len(segs) == 1
    assert segs[0]["tokens"] == [TSB + 0, 1, 2, TSB + 5]
    text, end = used_text(tokens, FakeTokenizer())
    assert text == " w1 w2"
    assert end == 5 * 0.02


def test_single_timestamp_ending_consumes_everything():
    # <|0|> w1 <|5|><|5|> w3 <|9|>          (single trailing timestamp)
    tokens = [TSB + 0, 1, TSB + 5, TSB + 5, 3, TSB + 9]
    segs = consumed_segments(tokens, TSB)
    assert [s["tokens"] for s in segs] == [[TSB + 0, 1, TSB + 5],
                                          [TSB + 5, 3, TSB + 9]]
    text, end = used_text(tokens, FakeTokenizer())
    assert text == " w1 w3"
    assert end == 9 * 0.02


def test_no_timestamp_pairs_keeps_whole_chunk():
    tokens = [TSB + 0, 1, 2, 3]
    segs = consumed_segments(tokens, TSB)
    assert len(segs) == 1 and segs[0]["start_pos"] is None
    text, end = used_text(tokens, FakeTokenizer())
    assert text == " w1 w2 w3" and end is None


def test_zero_length_and_empty_segments_contribute_nothing():
    # <|3|><|3|> is a start==end segment; then an empty-text pair
    tokens = [TSB + 3, TSB + 3, TSB + 4, TSB + 7, TSB + 7, 1, TSB + 9, TSB + 9]
    text, _ = used_text(tokens, FakeTokenizer())
    assert text == " w1"


# ----------------------------------------------------------- skip replication
def test_skip_rule():
    cfg = {"no_speech_threshold": 0.6, "log_prob_threshold": -1.0}
    # below the no-speech threshold: never skip
    assert not predicts_skip(0.5, -50.0, 10, cfg)
    # above it, but confident decode (avg_logprob > -1.0): the override holds
    assert not predicts_skip(0.9, -0.5, 10, cfg)   # avg = -0.5*10/11 ~ -0.45
    # above it and unconfident: skip
    assert predicts_skip(0.9, -2.0, 10, cfg)       # avg = -2*10/11 ~ -1.82
    # threshold disabled: never skip
    assert not predicts_skip(0.99, -30.0, 10, {"no_speech_threshold": None})


# ------------------------------------------------------------- capture proxy
class FakeResult:
    def __init__(self, n):
        self.sequences_ids = [[TSB + 0, k, TSB + 2] for k in range(n)]
        self.scores = [-0.1 * (k + 1) for k in range(n)]
        self.no_speech_prob = 0.01


class FakeInner:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, features, prompts, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(kwargs.get("num_hypotheses", 1))]


def test_proxy_injects_only_on_beam_path_and_passes_result_through():
    inner, sink = FakeInner(), CaptureSink()
    proxy = CapturingModel(inner, sink, NUM_HYPS)

    out = proxy.generate(None, [[1]], beam_size=8, patience=1)
    assert inner.calls[-1]["num_hypotheses"] == NUM_HYPS
    assert isinstance(out[0], FakeResult)                  # unchanged object
    assert len(sink.calls[-1]["sequences_ids"]) == NUM_HYPS
    assert sink.calls[-1]["beam_path"]

    proxy.generate(None, [[1]], sampling_temperature=0.2, sampling_topk=0)
    assert "num_hypotheses" not in inner.calls[-1]         # sampling untouched
    assert not sink.calls[-1]["beam_path"]

    assert proxy.device == "cpu"                           # forwarding


def test_proxy_refuses_beam_smaller_than_nbest():
    proxy = CapturingModel(FakeInner(), CaptureSink(), NUM_HYPS)
    try:
        proxy.generate(None, [[1]], beam_size=2)
    except AssertionError:
        return
    raise AssertionError("beam_size=2 with 8 hypotheses must be rejected")


# ---------------------------------------------------------------- oracle DP
def test_transduce_matches_plain_edit_distance_over_full_ref():
    ref = "a b c d".split()
    hyp = "a x d".split()
    v0 = [0] + [10 ** 6] * len(ref)     # forced to start at position 0
    v, _ = transduce(v0, hyp, ref)
    assert v[len(ref)] == sum(sdi(ref, hyp))


def test_oracle_recovers_deletion_split_across_chunks():
    ref = "a b c d".split()
    chunks = [[["a"], ["a", "b"]],       # top-1 dropped "b"
              [["d"], ["c", "d"]]]       # top-1 dropped "c"
    total, chosen = oracle_window(chunks, ref)
    assert total == 0 and chosen == [1, 1]


def test_oracle_never_worse_than_any_fixed_choice():
    ref = "a b c d e f".split()
    chunks = [[["a", "x"], ["a", "b"], ["q"]],
              [["c", "c", "c"], ["c", "d"], []],
              [["e", "f"], ["f"], ["e", "f", "g"]]]
    total, chosen = oracle_window(chunks, ref)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                fixed = chunks[0][i] + chunks[1][j] + chunks[2][k]
                assert total <= sum(sdi(ref, fixed))
    picked = chunks[0][chosen[0]] + chunks[1][chosen[1]] + chunks[2][chosen[2]]
    assert total == sum(sdi(ref, picked))


def test_oracle_handles_empty_hypothesis_and_trailing_ref():
    ref = "a b c".split()
    # one chunk, and the best hypothesis is empty: everything is deleted
    total, chosen = oracle_window([[[], ["z", "z", "z", "z"]]], ref)
    assert total == 3 and chosen == [0]


def test_window_oracle_agrees_with_frozen_sdi_and_skips_skipped_chunks():
    rec = {"chunks": [
        {"skipped": False,
         "hyps": [{"text": " καλημέρα σας", "score": -1.0, "end": None},
                  {"text": " καλημέρα σε όλους σας", "score": -1.2, "end": None}]},
        {"skipped": True,        # a skipped chunk is not a choice point
         "hyps": [{"text": " ΔΕΝ ΠΡΕΠΕΙ ΝΑ ΜΠΕΙ", "score": -1.0, "end": None}]},
        {"skipped": False,
         "hyps": [{"text": " ευχαριστώ πολύ", "score": -1.0, "end": None},
                  {"text": " ευχαριστώ", "score": -1.1, "end": None}]},
    ]}
    ref = ftoks("καλημέρα σε όλους σας ευχαριστώ πολύ")
    (s, d, i, n), text, dp = window_oracle(rec, ref, nbest=8)
    assert (s, d, i) == (0, 0, 0) and dp == 0
    assert "ΔΕΝ ΠΡΕΠΕΙ" not in text
    # 1-best restriction = the pipeline's own output (pipeline-style join)
    (s1, d1, i1, _), text1, _ = window_oracle(rec, ref, nbest=1)
    assert text1 == " καλημέρα σας ευχαριστώ πολύ".strip()
    assert (s1, d1, i1) == sdi(ref, ftoks(" καλημέρα σας ευχαριστώ πολύ"))
    assert d1 == 2                      # the two dropped words are deletions


def test_reconstruction_uses_pipeline_join_including_boundary_merges():
    # faster-whisper joins segment texts with plain concatenation: a chunk
    # text without a leading space merges with the previous word. Observed in
    # the real pipeline output; the reconstruction must replicate it.
    rec = {"chunks": [
        {"skipped": False, "hyps": [{"text": " το σώμα ότι", "score": -1.0,
                                     "end": None}]},
        {"skipped": False, "hyps": [{"text": "Επειδή έχει", "score": -1.0,
                                     "end": None}]},
    ]}
    ref = ftoks("το σώμα ότι επειδή έχει")
    (s, d, i, _), text, dp = window_oracle(rec, ref, nbest=1)
    assert "ότιΕπειδή" in text
    assert (s, d, i) == sdi(ref, ftoks(text))       # frozen scoring, merged
    assert s + d + i != dp                          # DP saw the split tokens


# ------------------------------------------------------------- resumability
def test_pending_skips_done_windows_and_respects_limit():
    rows = [{"window_id": f"w{i}"} for i in range(5)]
    state = {"windows": {"w0": {}, "w3": {}}}
    assert [r["window_id"] for r in pending(state, rows, None)] == \
        ["w1", "w2", "w4"]
    assert [r["window_id"] for r in pending(state, rows, 2)] == ["w1"]
    assert pending({"windows": {r["window_id"]: {} for r in rows}},
                   rows, None) == []
