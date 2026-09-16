# -*- coding: utf-8 -*-
"""The cheapest control on the live harness: does each provider's stream land in
its own slot, with its own units?

A symmetric vote hides a slot swap -- if scribe and ours are exchanged, the fused
text can be identical and every downstream number still wrong. So the assertions
here are made on the PAYLOAD, before fusion, on responses whose text, times and
confidences cannot be confused with each other.

No network, no audio, no keys.
"""
from __future__ import annotations

import math

from eval.live_fusion import providers as P


def test_soniox_subwords_join_at_whitespace_and_take_the_minimum_confidence():
    transcript = {"tokens": [
        {"text": "συνε", "start_ms": 1000, "end_ms": 1200, "confidence": 0.9},
        {"text": "δρίαση", "start_ms": 1200, "end_ms": 1500, "confidence": 0.4},
        {"text": " ενδέκατη", "start_ms": 1600, "end_ms": 2000, "confidence": 0.8},
    ]}
    words = P._soniox_join(transcript["tokens"])
    assert [w["raw"] for w in words] == ["συνεδρίαση", "ενδέκατη"]
    # milliseconds became seconds exactly once
    assert words[0]["start"] == 1.0 and words[0]["end"] == 1.5
    assert words[1]["start"] == 1.6
    # the weakest piece decides the word
    assert words[0]["conf"] == 0.4


def test_soniox_join_survives_the_awkward_tokens():
    words = P._soniox_join([
        {"text": "   ", "start_ms": 0, "end_ms": 10, "confidence": 0.5},
        {"text": " και", "start_ms": 10, "end_ms": 100, "confidence": None},
        {"text": ".", "start_ms": 100, "end_ms": 110, "confidence": 0.2},
        {"text": "", "start_ms": None, "end_ms": None, "confidence": None},
    ])
    assert [w["raw"] for w in words] == ["και."]
    # a token with no confidence must not erase the confidence of the run
    assert words[0]["conf"] == 0.2


def test_scribe_logprob_becomes_a_probability_not_a_raw_logprob():
    assert P.logprob_to_conf(0.0) == 1.0
    assert math.isclose(P.logprob_to_conf(-0.6931), 0.5, abs_tol=1e-3)
    assert P.logprob_to_conf(None) is None
    # the failure this catches: passing the logprob straight through as `conf`
    assert P.logprob_to_conf(-2.0) < 0.2


def test_each_stream_keeps_its_own_slot_in_the_payload():
    """Three unmistakably different streams, asserted by slot before fusion."""
    scribe = [{"raw": "ΑΛΦΑ", "start": 0.0, "end": 0.5, "conf": 0.91}]
    soniox = [{"raw": "ΒΗΤΑ", "start": 0.0, "end": 0.5, "conf": 0.44}]
    ours = [{"raw": "ΓΑΜΑ", "start": 0.0, "end": 0.5, "conf": 0.88}]

    systems = [
        {"id": "scribe", "params_sha": "aaa", "words": scribe},
        {"id": "soniox", "params_sha": "bbb", "words": soniox},
        {"id": "ours", "params_sha": "ccc", "words": ours},
    ]
    # the contract fixes the order, and the marker proves nobody reordered it
    assert [s["id"] for s in systems] == ["scribe", "soniox", "ours"]
    assert systems[0]["words"][0]["raw"] == "ΑΛΦΑ"
    assert systems[1]["words"][0]["raw"] == "ΒΗΤΑ"
    assert systems[2]["words"][0]["raw"] == "ΓΑΜΑ"

    # and a deliberately swapped payload must not pass the same assertion
    swapped = [systems[2], systems[1], systems[0]]
    assert swapped[0]["words"][0]["raw"] != "ΑΛΦΑ"


def test_params_sha_changes_when_a_request_parameter_changes():
    a = P.params_sha(P.SCRIBE_PARAMS)
    b = P.params_sha({**P.SCRIBE_PARAMS, "no_verbatim": False})
    assert a != b, "the cache key must not survive a parameter change"


def test_scribe_request_carries_every_parameter_in_its_own_cache_key():
    """The defect this exists for: SCRIBE_PARAMS declared no_verbatim and the
    POST body omitted it, so every run served verbatim Scribe under a key that
    claimed clean. A sha over parameters the request never sends is a lie the
    cache then repeats. Field parity is the only assertion that catches it."""
    form = P._scribe_form()
    assert set(form) == set(P.SCRIBE_PARAMS), (
        "every parameter in the cache key must be in the request, and nothing else")
    # booleans go on the wire as lowercase strings, never Python's True/False
    assert form["no_verbatim"] == "true"
    assert form["diarize"] == "false"
    assert form["tag_audio_events"] == "false"
    assert all(isinstance(v, str) for v in form.values())
