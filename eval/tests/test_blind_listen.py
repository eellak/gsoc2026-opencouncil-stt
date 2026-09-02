"""Tests for the blind listening page's core (spec §4.1, listening-protocol.md v1).

`chooser/listen_server.py:29-31` hands the judge the left/right context and the deleted
words. That page cannot be used here: seeing the witness first anchors the label and
turns Soniox into a teacher, which would make every row it produced a copy of the
witness rather than evidence about it.

Section 4.1 says the page must ENFORCE the order rather than rely on discipline, so the
enforcement lives in functions that can be tested without a socket, and again at the
wire. The session rules -- uncertainty and exclusion states, the three-listen limit, the
five hidden repeats -- come from the FROZEN protocol of 2026-08-09, which governs every
blind label this project has taken. They are not optional for this round.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import blind_listen as BL  # noqa: E402
from eval.controlled_eval.blind_listen import (  # noqa: E402
    EXCLUSION_REASONS, MAX_LISTENS, N_REPEATS, LabelError, RevealDenied,
    page_payload, present_token, reveal, save_label, self_consistency_pairs,
    session_order, training_rows)

QUEUE = [
    {"clip_id": "athens__mar23_2026__97000", "clip_dur": 9.6, "city": "athens",
     "meeting_id": "mar23_2026", "audio_url": "https://x/y.mp3",
     "origin_row_id": "abc", "pool": "deletion_hard", "witness": "soniox-stt-rt-v4"},
    {"clip_id": "chania__apr1_2026__12000", "clip_dur": 7.2, "city": "chania",
     "meeting_id": "apr1_2026", "audio_url": "https://x/z.mp3",
     "origin_row_id": "def", "pool": "deletion_hard", "witness": "soniox-stt-rt-v4"},
]
BIG = [{"clip_id": f"athens__m_2026__{i}000", "clip_dur": 6.0} for i in range(40)]
WITNESS = {"athens__mar23_2026__97000": "WITNESS TEXT"}
TARGET = {"athens__mar23_2026__97000": "CURRENT TRAINING TARGET"}

P0 = present_token(QUEUE[0]["clip_id"], 0)


def order(queue=QUEUE, repeats=0):
    return session_order(queue, n_repeats=repeats)


# ------------------------------------------------------------------ the gate

def test_the_witness_cannot_be_revealed_before_a_label_exists():
    with pytest.raises(RevealDenied):
        reveal(P0, order(), {}, WITNESS, TARGET)


def test_the_witness_is_revealed_once_the_label_is_in():
    labels = {P0: {"text": "ό,τι άκουσα", "state": "transcribed"}}
    got = reveal(P0, order(), labels, WITNESS, TARGET)
    assert got["witness_text"] == "WITNESS TEXT"
    assert got["current_target"] == "CURRENT TRAINING TARGET"
    assert got["label"] == "ό,τι άκουσα"


def test_an_empty_label_does_not_open_the_gate():
    """Otherwise the gate is opened by pressing save, which is not a label."""
    with pytest.raises(RevealDenied):
        reveal(P0, order(), {P0: {"text": "   ", "state": "transcribed"}},
               WITNESS, TARGET)


def test_an_excluded_span_still_opens_the_gate():
    """'I could not hear it' is a committed answer under the frozen protocol, and the
    listener has to be able to see afterwards what the witness claimed was there."""
    labels = {P0: {"text": "", "state": "excluded", "reason": "overlap"}}
    assert reveal(P0, order(), labels, WITNESS, TARGET)["witness_text"]


def test_revealing_an_unknown_presentation_is_denied():
    with pytest.raises(RevealDenied):
        reveal("deadbeefdeadbeef", order(), {"deadbeefdeadbeef": {"text": "x"}},
               WITNESS, TARGET)


# --------------------------------------------------------------- the payload

def test_the_page_payload_carries_no_text_of_any_kind():
    """Blindness must survive someone reading the page source."""
    blob = repr(page_payload(order()))
    for banned in ("WITNESS", "TARGET", "soniox", "deletion_hard", "https://"):
        assert banned not in blob, banned


def test_the_payload_does_not_carry_the_meeting_even_inside_the_clip_id():
    """A clip id names its meeting, and the meeting has a published transcript. That
    is the same anchoring the witness would supply, so the browser gets a token."""
    blob = repr(page_payload(order()))
    assert "mar23_2026" not in blob
    assert "athens" not in blob


def test_the_payload_keeps_only_what_the_listener_needs():
    for c in page_payload(order()):
        # `mode` joined the payload with the blind/assisted split of 2026-08-28; the
        # browser has to know whether to prefill. It says nothing about the meeting.
        assert set(c) == {"token", "clip_dur", "mode"}


def test_the_payload_cannot_tell_a_repeat_from_a_first_hearing():
    """A repeat the listener can recognise measures memory, not consistency."""
    pres = session_order(BIG, n_repeats=N_REPEATS)
    pay = page_payload(pres)
    assert len(pay) == len(BIG) + N_REPEATS
    assert len({c["token"] for c in pay}) == len(pay)
    for c in pay:
        assert "repeat" not in c


# ----------------------------------------------------------------- the session

def test_the_session_is_deterministic():
    a = session_order(BIG, n_repeats=N_REPEATS)
    b = session_order(BIG, n_repeats=N_REPEATS)
    assert [p["token"] for p in a] == [p["token"] for p in b]


def test_the_repeats_come_last_and_are_drawn_from_what_was_already_heard():
    pres = session_order(BIG, n_repeats=N_REPEATS)
    head, tail = pres[:len(BIG)], pres[len(BIG):]
    assert all(p["repeat"] is False for p in head)
    assert all(p["repeat"] is True for p in tail)
    heard = {p["clip_id"] for p in head}
    assert {p["clip_id"] for p in tail} <= heard
    assert len({p["clip_id"] for p in tail}) == N_REPEATS


def test_a_queue_shorter_than_the_repeat_count_repeats_what_it_has():
    pres = session_order(QUEUE, n_repeats=N_REPEATS)
    assert sum(p["repeat"] for p in pres) == len(QUEUE)


def test_a_repeat_never_becomes_a_training_row():
    pres = session_order(BIG, n_repeats=N_REPEATS)
    labels = {p["token"]: {"text": "κείμενο", "state": "transcribed"} for p in pres}
    rows = training_rows(labels, pres)
    assert len(rows) == len(BIG)
    assert len({r["clip_id"] for r in rows}) == len(BIG)


def test_uncertain_and_excluded_answers_are_not_training_rows_either():
    pres = order()
    labels = {
        pres[0]["token"]: {"text": "καλό", "state": "transcribed"},
        pres[1]["token"]: {"text": "", "state": "excluded", "reason": "overlap"},
    }
    assert [r["clip_id"] for r in training_rows(labels, pres)] == [QUEUE[0]["clip_id"]]


def test_the_consistency_pairs_are_what_the_noise_floor_is_computed_from():
    pres = session_order(BIG, n_repeats=N_REPEATS)
    labels = {p["token"]: {"text": f"κείμενο {i}", "state": "transcribed"}
              for i, p in enumerate(pres)}
    pairs = self_consistency_pairs(labels, pres)
    assert len(pairs) == N_REPEATS
    for first, again, clip in pairs:
        assert first != again          # the fixture varies them on purpose
        assert clip in {p["clip_id"] for p in pres}


# ----------------------------------------------------------------- labelling

def test_saving_a_label_records_when_and_keeps_the_others():
    pres = order()
    labels = {pres[1]["token"]: {"text": "παλιό", "state": "transcribed"}}
    out = save_label(labels, pres, pres[0]["token"], "νέο", now="T0")
    assert out[pres[0]["token"]]["text"] == "νέο"
    assert out[pres[0]["token"]]["saved_at"] == "T0"
    assert out[pres[0]["token"]]["state"] == "transcribed"
    assert out[pres[1]["token"]]["text"] == "παλιό"


def test_an_empty_transcription_is_refused_but_an_exclusion_is_not():
    """'Nothing was said' is not a label. 'I could not make it out' is."""
    pres = order()
    for bad in ("", "   ", "\n"):
        with pytest.raises(LabelError):
            save_label({}, pres, pres[0]["token"], bad, now="T0")
    out = save_label({}, pres, pres[0]["token"], "", now="T0",
                     state="excluded", reason="unintelligible")
    assert out[pres[0]["token"]]["state"] == "excluded"


def test_an_exclusion_must_name_one_of_the_frozen_reasons():
    """The share of exclusions per reason is itself a finding, so the reason cannot be
    free text that nobody can count."""
    pres = order()
    with pytest.raises(LabelError):
        save_label({}, pres, pres[0]["token"], "", now="T0", state="excluded")
    with pytest.raises(LabelError):
        save_label({}, pres, pres[0]["token"], "", now="T0", state="excluded",
                   reason="βαριέμαι")
    for good in EXCLUSION_REASONS:
        assert save_label({}, pres, pres[0]["token"], "", now="T0",
                          state="excluded", reason=good)


def test_an_unknown_state_is_refused():
    pres = order()
    with pytest.raises(LabelError):
        save_label({}, pres, pres[0]["token"], "κάτι", now="T0", state="maybe")


def test_a_label_for_a_presentation_outside_the_session_is_refused():
    """A label with no clip is a row with no audio; it cannot become training data."""
    with pytest.raises(LabelError):
        save_label({}, order(), "0000000000000000", "κείμενο", now="T0")


def test_relabelling_keeps_the_first_answer_alongside_the_new_one():
    """§4.2 asks for repeated corrections of the same span to be deduplicated, which
    needs the history rather than a silent overwrite."""
    pres = order()
    tok = pres[0]["token"]
    a = save_label({}, pres, tok, "πρώτο", now="T0")
    b = save_label(a, pres, tok, "δεύτερο", now="T1")
    assert b[tok]["text"] == "δεύτερο"
    assert b[tok]["previous"] == [{"text": "πρώτο", "saved_at": "T0",
                                   "state": "transcribed"}]


def test_the_listen_count_is_carried_so_the_three_listen_rule_can_be_applied():
    pres = order()
    out = save_label({}, pres, pres[0]["token"], "κείμενο", now="T0", listens=4)
    assert out[pres[0]["token"]]["listens"] == 4
    assert MAX_LISTENS == 3


# --------------------------------------------------------------- clip serving

def _server(tmp_path, queue=QUEUE, repeats=0, labels_path=None):
    import socketserver
    import threading

    from eval.controlled_eval.blind_listen import make_handler

    clips = tmp_path / "clips"
    clips.mkdir()
    for c in queue:
        (clips / f"{c['clip_id']}.wav").write_bytes(b"RIFFfake")
    pres = session_order(queue, n_repeats=repeats)
    handler = make_handler(pres, clips, labels_path or (tmp_path / "labels.json"),
                           WITNESS, TARGET)
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv.shutdown, pres


def test_the_http_reveal_endpoint_refuses_before_the_label_and_answers_after(tmp_path):
    """The enforcement claim of §4.1, checked at the wire rather than in a helper: a
    listener who opens /reveal in another tab gets 403, not the witness."""
    import json as _json
    import urllib.error
    import urllib.request

    base, stop, pres = _server(tmp_path)
    try:
        tok = pres[0]["token"]
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{base}/reveal/{tok}")
        assert e.value.code == 403

        req = urllib.request.Request(
            f"{base}/label", method="POST",
            data=_json.dumps({"token": tok, "text": "ό,τι άκουσα"}).encode(),
            headers={"Content-Type": "application/json"})
        assert urllib.request.urlopen(req).status == 200

        got = _json.loads(urllib.request.urlopen(f"{base}/reveal/{tok}").read())
        assert got["witness_text"] == "WITNESS TEXT"

        saved = _json.loads((tmp_path / "labels.json").read_text())
        assert saved[tok]["text"] == "ό,τι άκουσα"
    finally:
        stop()


def test_the_served_page_source_contains_no_witness_text(tmp_path):
    import urllib.request
    base, stop, _ = _server(tmp_path)
    try:
        html = urllib.request.urlopen(base + "/").read().decode()
        for banned in ("WITNESS TEXT", "CURRENT TRAINING TARGET", "mar23_2026",
                       "https://x/y.mp3"):
            assert banned not in html, banned
    finally:
        stop()


def test_a_clip_outside_the_session_is_not_served(tmp_path):
    import urllib.error
    import urllib.request
    base, stop, _ = _server(tmp_path)
    try:
        for bad in ("nope__x__1.wav", "..%2F..%2Fetc%2Fpasswd.wav",
                    "0000000000000000.wav", "athens__mar23_2026__97000.wav"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(f"{base}/clip/{bad}")
            assert e.value.code == 404
    finally:
        stop()


def test_a_malformed_post_is_a_400_and_not_a_stack_trace(tmp_path):
    """A listener's browser is not the only thing that can reach this port."""
    import urllib.error
    import urllib.request
    base, stop, _ = _server(tmp_path)
    try:
        for body in (b"[]", b"not json", b'{"token": null}', b"\xff\xfe"):
            req = urllib.request.Request(
                f"{base}/label", method="POST", data=body,
                headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            assert e.value.code == 400
    finally:
        stop()


# ---------------------------------------------------------------- resuming

def test_the_page_only_offers_what_is_still_owed_an_answer():
    """528 presentations is several sittings. Without this the page restarts at the
    first clip every time it is opened and the listener re-hears finished work."""
    from eval.controlled_eval.blind_listen import remaining
    pres = session_order(BIG, n_repeats=N_REPEATS)
    labels = {p["token"]: {"text": "κείμενο", "state": "transcribed"}
              for p in pres[:10]}
    todo = remaining(pres, labels)
    assert len(todo) == len(pres) - 10
    assert [p["token"] for p in todo] == [p["token"] for p in pres[10:]]


def test_an_uncertain_or_excluded_answer_also_counts_as_done():
    from eval.controlled_eval.blind_listen import remaining
    pres = order()
    labels = {pres[0]["token"]: {"text": "", "state": "uncertain"}}
    assert [p["token"] for p in remaining(pres, labels)] == [pres[1]["token"]]


def test_reopening_the_page_after_every_answer_offers_nothing(tmp_path):
    import json as _json
    import urllib.request
    base, stop, pres = _server(tmp_path)
    try:
        for p in pres:
            req = urllib.request.Request(
                f"{base}/label", method="POST",
                data=_json.dumps({"token": p["token"], "text": "x"}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req)
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "const Q = []" in html
    finally:
        stop()


def test_an_oversized_body_is_refused_without_being_read(tmp_path):
    import urllib.error
    import urllib.request
    from eval.controlled_eval.blind_listen import MAX_BODY_BYTES
    base, stop, pres = _server(tmp_path)
    try:
        req = urllib.request.Request(
            f"{base}/label", method="POST", data=b"{}",
            headers={"Content-Type": "application/json",
                     "Content-Length": str(MAX_BODY_BYTES + 1)})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 400
    finally:
        stop()


def test_a_label_that_cannot_be_written_does_not_become_visible(tmp_path):
    """`/reveal` opens on a committed answer, so a label held in server state but not on
    disk would spend the blindness of a clip whose answer was lost."""
    import json
    import urllib.error
    import urllib.request

    base, stop, pres = _server(tmp_path, labels_path=tmp_path / "gone" / "labels.json")
    tok = pres[0]["token"]
    try:
        req = urllib.request.Request(
            f"{base}/label", method="POST",
            data=json.dumps({"token": tok, "text": "καλησπέρα σας"}).encode(),
            headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 500
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{base}/reveal/{tok}", timeout=10)
        assert e.value.code == 403, "an unwritten answer must not open the witness"
    finally:
        stop()


def test_only_loopback_and_the_tailnet_may_be_bound():
    """There is no authentication here; the binding is what keeps council audio off
    other networks, so it is checked rather than trusted."""
    from eval.controlled_eval.blind_listen import _bindable
    held = frozenset({"100.123.255.50"})
    for ok in ("127.0.0.1", "::1", "100.123.255.50"):
        assert _bindable(ok, tailnet=held) is True, ok
    for bad in ("0.0.0.0", "::", "", "192.168.1.10", "8.8.8.8", "example.com"):
        assert _bindable(bad, tailnet=held) is False, bad


def test_a_cgnat_address_this_machine_does_not_hold_is_refused():
    """100.64.0.0/10 is carrier-grade NAT as well as the tailnet, so an ISP-assigned
    address on a real interface passes a range test. Membership is asked of tailscale,
    and with no answer the check fails closed."""
    from eval.controlled_eval.blind_listen import _bindable
    assert _bindable("100.64.7.7", tailnet=frozenset({"100.123.255.50"})) is False
    assert _bindable("100.123.255.50", tailnet=frozenset()) is False
    assert _bindable("127.0.0.1", tailnet=frozenset()) is True


# ==========================================================================
# The blind core and the assisted remainder (user decision, 2026-08-28)
# ==========================================================================

def test_the_blind_core_is_a_fixed_size_spread_over_the_whole_queue():
    """Harold labels 120 clips from scratch and corrects Soniox on the rest. The core
    is taken at an even step over the queue, which is ordered by city and meeting, so
    it inherits the queue's city mix instead of being 120 clips of one municipality."""
    core = BL.blind_core(BIG, n=8)
    assert len(core) == 8
    assert core == BL.blind_core(BIG, n=8)
    ids = [c["clip_id"] for c in BIG]
    pos = sorted(ids.index(c) for c in core)
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    assert max(gaps) - min(gaps) <= 1


def test_a_queue_smaller_than_the_core_is_entirely_blind():
    assert len(BL.blind_core(QUEUE, n=120)) == len(QUEUE)


def test_blind_presentations_come_first_and_carry_their_mode():
    pres = BL.session_order(BIG, n_repeats=3, n_blind=8)
    modes = [p["mode"] for p in pres]
    assert set(modes) == {"blind", "assisted"}
    first_assisted = modes.index("assisted")
    assert all(m == "blind" for m in modes[:first_assisted])
    assert sum(1 for p in pres if p["mode"] == "blind" and not p["repeat"]) == 8


def test_every_repeat_is_drawn_from_the_blind_core():
    """The noise floor has to be measured under the condition that decides. A repeat of
    an assisted clip would measure how consistently he edits Soniox, not how
    consistently he hears."""
    pres = BL.session_order(BIG, n_repeats=3, n_blind=8)
    core = set(BL.blind_core(BIG, n=8))
    reps = [p for p in pres if p["repeat"]]
    assert len(reps) == 3
    assert all(p["clip_id"] in core and p["mode"] == "blind" for p in reps)


def test_the_witness_is_served_as_a_prefill_only_for_assisted_clips():
    pres = BL.session_order(BIG, n_repeats=1, n_blind=2)
    witness = {c["clip_id"]: "SONIOX TEXT" for c in BIG}
    blind = next(p for p in pres if p["mode"] == "blind" and not p["repeat"])
    assisted = next(p for p in pres if p["mode"] == "assisted")
    assert BL.prefill(assisted["token"], pres, witness)["text"] == "SONIOX TEXT"
    with pytest.raises(RevealDenied):
        BL.prefill(blind["token"], pres, witness)


def test_a_saved_label_records_which_condition_produced_it():
    pres = BL.session_order(BIG, n_repeats=1, n_blind=2)
    assisted = next(p for p in pres if p["mode"] == "assisted")
    labels = save_label({}, pres, assisted["token"], "κάτι", "2026-08-28T00:00:00Z")
    assert labels[assisted["token"]]["mode"] == "assisted"


def test_measurement_can_ask_for_the_blind_rows_alone():
    """Blind and assisted labels are not one set: an assisted label inherits Soniox's
    errors, so pooling them would contaminate every number the round reports."""
    pres = BL.session_order(BIG, n_repeats=1, n_blind=2)
    labels = {}
    for p in pres[:6]:
        labels = save_label(labels, pres, p["token"], "κάτι", "2026-08-28T00:00:00Z")
    blind_only = training_rows(labels, pres, mode="blind")
    assert blind_only and all(
        r["mode"] == "blind" for r in blind_only)
    assert len(training_rows(labels, pres)) > len(blind_only)


def test_the_page_payload_says_the_mode_and_still_nothing_else():
    pres = BL.session_order(BIG, n_repeats=1, n_blind=2)
    payload = page_payload(pres)
    assert set(payload[0]) == {"token", "clip_dur", "mode"}


def test_the_wire_refuses_a_prefill_for_a_blind_clip(tmp_path):
    """The HTTP-level twin of the function gate: the browser is not the thing that has
    to be trustworthy."""
    import json as _json
    import socketserver
    import threading
    import urllib.error
    import urllib.request

    from eval.controlled_eval.blind_listen import make_handler

    queue = [{"clip_id": f"athens__m_2026__{i}000", "clip_dur": 6.0} for i in range(6)]
    witness = {c["clip_id"]: "SONIOX TEXT" for c in queue}
    clips = tmp_path / "clips"
    clips.mkdir()
    for c in queue:
        (clips / f"{c['clip_id']}.wav").write_bytes(b"RIFFfake")
    pres = BL.session_order(queue, n_repeats=1, n_blind=2)
    blind = next(p for p in pres if p["mode"] == "blind" and not p["repeat"])
    assisted = next(p for p in pres if p["mode"] == "assisted")
    handler = make_handler(pres, clips, tmp_path / "labels.json", witness, {})
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{base}/prefill/{blind['token']}")
        assert e.value.code == 403
        got = _json.loads(urllib.request.urlopen(
            f"{base}/prefill/{assisted['token']}").read())
        assert got["text"] == "SONIOX TEXT"
    finally:
        srv.shutdown()


def test_an_uncertain_label_keeps_the_half_heard_text():
    """Half a sentence heard is evidence: it says the audio was not empty and shows how
    far the listener got. The page used to discard it, which threw away exactly the hard
    clips this round exists to collect."""
    pres = session_order(QUEUE, n_repeats=0)
    out = save_label({}, pres, pres[0]["token"], "να πω κάτι για", now="T0",
                     state="uncertain")
    assert out[pres[0]["token"]]["text"] == "να πω κάτι για"
    assert out[pres[0]["token"]]["state"] == "uncertain"


def test_an_uncertain_label_may_still_be_empty():
    """Nothing intelligible at all is a legitimate answer, and it must not be forced
    into an exclusion, which claims a reason it does not have."""
    pres = session_order(QUEUE, n_repeats=0)
    out = save_label({}, pres, pres[0]["token"], "", now="T0", state="uncertain")
    assert out[pres[0]["token"]]["state"] == "uncertain"


def test_uncertain_text_never_reaches_training():
    """It is a measurement, not a target: a partial transcript as a training label would
    teach the model to stop early -- the exact failure this round is fixing."""
    pres = session_order(QUEUE, n_repeats=0)
    labels = save_label({}, pres, pres[0]["token"], "μισή πρόταση", now="T0",
                        state="uncertain")
    assert training_rows(labels, pres) == []
