"""Tests for the blind preference page and its frozen decision rule.

Record: exp-2026-08-26-training-round-2. Design frozen 2026-09-06 (wayfinder #59).

Three failures this file exists to prevent, in order of cost:

1. A number computed on the CHALLENGE sample, or on a mixture, or on a partial set,
   being quoted as "the model is better". The rule is a conjunction and every clause is
   tested, including the ones that are easy to satisfy by accident.
2. The page leaking which transcript belongs to which arm -- in the payload, in a URL,
   or by revealing after each clip instead of after the session.
3. A win manufactured by silence: a lower error rate paid for by dropping speech. The
   guard is a between-arm comparison and cannot be computed from an unattributed flag,
   so the code refuses rather than guesses.

No audio and no real transcript text appears here; the candidate texts are placeholders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import blind_preference as BP  # noqa: E402
from eval.controlled_eval.blind_preference import (  # noqa: E402
    CHOICES, TIE_RULE, CandidateError, JudgementError, MixedSampleError,
    RevealDenied, SampleKindError, challenge_table, check_complete, city_weights,
    decide, load_candidates, lost_speech_counts, page_payload, present_token, records,
    remaining, reveal, save_judgement, session_order, side_assignment, win_rate_ci)

INC = "artifact-adapter-cleanpack-cont-s47"
NEW = "artifact-adapter-round3-candidate"


# --------------------------------------------------------------------- fixtures

def clip(sample="ordinary", city="athens", meeting="m1_2026", start=1000, dur=20.0):
    prefix = "ord" if sample == "ordinary" else "chl"
    return {"clip_id": f"{prefix}__{city}__{meeting}__{start}", "sample": sample,
            "city": city, "meeting_id": meeting, "meeting_date": "2026-03-23 15:00:00",
            "audio_url": f"https://data.opencouncil.gr/audio/{city}.mp3",
            "clip_start": start / 1000, "clip_end": start / 1000 + dur,
            "clip_dur": dur}


def clips_over(spec, sample="ordinary"):
    """`{(city, meeting): n}` -> sample clips."""
    out = []
    for (city, meeting), n in spec.items():
        for i in range(n):
            out.append(clip(sample, city, meeting, 1000 * (i + 1)))
    return out


def candidates_for(clips, incumbent=INC, new=NEW):
    return {"incumbent_arm": incumbent, "new_arm": new,
            "clips": {c["clip_id"]: {incumbent: "CAND-INCUMBENT", new: "CAND-NEW"}
                      for c in clips}}


def judged(clips, choose, lost=None, sample="ordinary"):
    """Committed judgement rows straight from `save_judgement`, via the session."""
    pres = session_order(sample, clips, candidates_for(clips))
    js: dict = {}
    for i, p in enumerate(pres):
        arm = choose(i, p)                     # INC, NEW or None (tie)
        if arm is None:
            choice = "tie"
        else:
            choice = "A" if p["sides"]["A"] == arm else "B"
        flags = (lost or (lambda i, p: {"A": False, "B": False}))(i, p)
        js = save_judgement(js, pres, p["token"], choice, flags, "2026-09-06T00:00:00Z")
    return records(js, pres), pres, clips


ALL_NEW = lambda i, p: NEW            # noqa: E731
ALL_INC = lambda i, p: INC            # noqa: E731
ALL_TIE = lambda i, p: None           # noqa: E731
COINFLIP = lambda i, p: NEW if i % 2 else INC   # noqa: E731


# ==========================================================================
# Blindness at the wire
# ==========================================================================

def test_the_browser_never_receives_a_clip_id_a_meeting_a_url_or_an_arm_name():
    clips = clips_over({("athens", "m1_2026"): 2, ("chania", "m2_2026"): 1})
    pres = session_order("ordinary", clips, candidates_for(clips))
    blob = json.dumps(page_payload(pres), ensure_ascii=False)
    for banned in ("ord__", "athens", "chania", "m1_2026", "://", INC, NEW,
                   "incumbent", "new_arm", "sides", "clip_id", "meeting"):
        assert banned not in blob, banned
    assert set(page_payload(pres)[0]) == {"token", "clip_dur", "a", "b"}


def test_the_side_assignment_is_deterministic_hidden_and_not_always_the_same_way():
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})
    sides = [side_assignment("ordinary", c["clip_id"], INC, NEW) for c in clips]
    assert sides == [side_assignment("ordinary", c["clip_id"], INC, NEW) for c in clips]
    a_is_new = sum(1 for s in sides if s["A"] == NEW)
    assert 0 < a_is_new < len(sides), "the new arm must not always be on one side"


def test_the_presentation_order_is_not_the_meeting_order():
    clips = clips_over({("chania", f"m{i:02d}_2026"): 1 for i in range(30)})
    pres = session_order("ordinary", clips, candidates_for(clips))
    assert [p["clip_id"] for p in pres] != [c["clip_id"] for c in clips]
    assert {p["clip_id"] for p in pres} == {c["clip_id"] for c in clips}


def test_the_token_is_stable_and_opaque():
    t = present_token("ordinary", "ord__athens__m1_2026__1000")
    assert t == present_token("ordinary", "ord__athens__m1_2026__1000")
    assert len(t) == 16 and all(ch in "0123456789abcdef" for ch in t)
    assert t != present_token("challenge", "ord__athens__m1_2026__1000")


def test_the_page_writes_candidate_text_with_textcontent_not_innerhtml():
    """A transcript is untrusted text as far as the page is concerned."""
    assert "$('#ta').textContent" in BP.PAGE
    assert "$('#tb').textContent" in BP.PAGE
    assert "innerHTML" not in BP.PAGE


def test_a_transcript_cannot_close_the_script_block_it_is_embedded_in():
    """`json.dumps` leaves `<` alone, and the payload is substituted INTO a <script>.
    A candidate containing `</script>` would end the block (CodeRabbit, 2026-09-06)."""
    payload = [{"token": "0" * 16, "clip_dur": 20.0,
                "a": "</script><img onerror=alert(1)>", "b": "κείμενο & <b>"}]
    out = BP.script_json(payload)
    assert "<" not in out and ">" not in out and "&" not in out
    assert json.loads(out) == payload, "the escaping must be transparent to JSON"


def test_the_rendered_page_carries_no_raw_angle_bracket_from_a_transcript():
    clips = clips_over({("athens", "m1_2026"): 1})
    cands = candidates_for(clips)
    cands["clips"][clips[0]["clip_id"]][INC] = "</script><b>x</b>"
    pres = session_order("ordinary", clips, cands)
    # Rendered exactly the way `make_handler`'s `page()` renders it.
    body = BP.PAGE.replace("__DATA__", BP.script_json(BP.page_payload(pres)))
    assert "</script><b>" not in body
    assert body.count("</script>") == 1, "only the page's own closing tag"


def test_the_server_binds_loopback_or_a_tailnet_address_it_actually_holds():
    tailnet = {"100.123.255.50"}
    assert BP._bindable("127.0.0.1", tailnet) is True
    assert BP._bindable("100.123.255.50", tailnet) is True
    assert BP._bindable("0.0.0.0", tailnet) is False
    assert BP._bindable("192.168.1.10", tailnet) is False
    assert BP._bindable("minipc.local", tailnet) is False
    # inside 100.64.0.0/10 but not an address this machine holds: carrier-grade NAT
    assert BP._bindable("100.99.1.1", tailnet) is False
    assert BP._bindable("100.123.255.50", set()) is False


# ==========================================================================
# Judgements
# ==========================================================================

def test_a_judgement_is_stored_resolved_to_arms_and_keeps_its_history():
    # Two clips, so the session is still incomplete after the first is relabelled.
    clips = clips_over({("athens", "m1_2026"): 2})
    pres = session_order("ordinary", clips, candidates_for(clips))
    tok = pres[0]["token"]
    js = save_judgement({}, pres, tok, "A", {"A": True, "B": False}, "t1")
    row = js[tok]
    assert row["winner_arm"] == pres[0]["sides"]["A"]
    assert row["loser_arm"] == pres[0]["sides"]["B"]
    assert row["lost_speech"][pres[0]["sides"]["A"]] is True
    assert row["lost_speech"][pres[0]["sides"]["B"]] is False
    js2 = save_judgement(js, pres, tok, "tie", None, "t2")
    assert js2[tok]["choice"] == "tie" and js2[tok]["winner_arm"] is None
    assert len(js2[tok]["previous"]) == 1


def test_an_unknown_choice_an_unknown_token_or_an_unknown_side_is_refused():
    clips = clips_over({("athens", "m1_2026"): 1})
    pres = session_order("ordinary", clips, candidates_for(clips))
    tok = pres[0]["token"]
    assert set(CHOICES) == {"A", "B", "tie"}
    with pytest.raises(JudgementError):
        save_judgement({}, pres, tok, "maybe", None, "t")
    with pytest.raises(JudgementError):
        save_judgement({}, pres, "0" * 16, "A", None, "t")
    with pytest.raises(JudgementError):
        save_judgement({}, pres, tok, "A", {"C": True}, "t")


def test_the_witness_is_revealed_only_after_the_whole_session_is_judged():
    """Not after that clip. Per-clip reveal teaches the listener each arm's style and
    unblinds every clip after it (Codex review 0201f130)."""
    clips = clips_over({("athens", "m1_2026"): 2})
    pres = session_order("ordinary", clips, candidates_for(clips))
    js: dict = {}
    with pytest.raises(RevealDenied):
        reveal(pres[0]["token"], pres, js)
    js = save_judgement(js, pres, pres[0]["token"], "A", None, "t")
    with pytest.raises(RevealDenied, match="still unjudged"):
        reveal(pres[0]["token"], pres, js)
    js = save_judgement(js, pres, pres[1]["token"], "tie", None, "t")
    assert reveal(pres[0]["token"], pres, js)["sides"] == pres[0]["sides"]
    assert remaining(pres, js) == []


def test_a_judgement_cannot_be_changed_once_the_session_is_complete():
    """After completion the identities are revealable, so an edit is an informed edit."""
    clips = clips_over({("athens", "m1_2026"): 2})
    pres = session_order("ordinary", clips, candidates_for(clips))
    js = save_judgement({}, pres, pres[0]["token"], "A", None, "t")
    js = save_judgement(js, pres, pres[0]["token"], "B", None, "t")   # still incomplete
    js = save_judgement(js, pres, pres[1]["token"], "tie", None, "t")
    with pytest.raises(JudgementError, match="already complete"):
        save_judgement(js, pres, pres[0]["token"], "A", None, "t")


def test_candidates_must_name_two_different_arms_and_cover_every_clip(tmp_path):
    clips = clips_over({("athens", "m1_2026"): 2})
    root = tmp_path / "pref"
    (root / "ordinary").mkdir(parents=True)
    p = root / "ordinary" / "candidates.json"

    p.write_text(json.dumps({"incumbent_arm": INC, "new_arm": INC,
                             "clips": {clips[0]["clip_id"]: {INC: "x"}}}))
    with pytest.raises(CandidateError, match="must both be set and differ"):
        load_candidates("ordinary", root)

    p.write_text(json.dumps({"incumbent_arm": INC, "new_arm": NEW,
                             "clips": {clips[0]["clip_id"]: {INC: "x"}}}))
    with pytest.raises(CandidateError, match="expected exactly"):
        load_candidates("ordinary", root)

    p.write_text(json.dumps(candidates_for(clips[:1])))
    cands = load_candidates("ordinary", root)
    with pytest.raises(CandidateError, match="no candidate pair"):
        session_order("ordinary", clips, cands)


# ==========================================================================
# The frozen rule: what it refuses
# ==========================================================================

def test_the_challenge_sample_may_not_decide_anything():
    clips = clips_over({("athens", f"m{i}_2026"): 1 for i in range(12)},
                       sample="challenge")
    rows, _pres, _ = judged(clips, ALL_NEW, sample="challenge")
    with pytest.raises(SampleKindError, match="may not decide"):
        decide(rows, INC, NEW, clips)
    table = challenge_table(rows, INC, NEW, clips)
    assert table["decides"] is False
    assert "verdict" not in table and "new_arm_wins" not in table


def test_the_ordinary_sample_may_not_go_through_the_challenge_table():
    clips = clips_over({("athens", f"m{i}_2026"): 1 for i in range(12)})
    rows, _pres, _ = judged(clips, ALL_NEW)
    with pytest.raises(SampleKindError, match="decide"):
        challenge_table(rows, INC, NEW, clips)


def test_a_mixed_judgement_set_is_refused_rather_than_summed():
    ordn = clips_over({("athens", f"m{i}_2026"): 1 for i in range(6)})
    chal = clips_over({("athens", f"c{i}_2026"): 1 for i in range(6)},
                      sample="challenge")
    a, _, _ = judged(ordn, ALL_NEW)
    b, _, _ = judged(chal, ALL_NEW, sample="challenge")
    with pytest.raises(MixedSampleError, match="never summed"):
        decide(a + b, INC, NEW, ordn + chal)


def test_a_partial_judgement_set_is_refused_so_the_rule_cannot_be_watched():
    clips = clips_over({("athens", f"m{i}_2026"): 1 for i in range(20)})
    rows, _pres, _ = judged(clips, ALL_NEW)
    with pytest.raises(JudgementError, match="not the frozen sample"):
        decide(rows[:15], INC, NEW, clips)
    with pytest.raises(JudgementError, match="not the frozen sample"):
        check_complete(rows, clips[:-1])


def test_a_duplicated_clip_is_refused():
    clips = clips_over({("athens", f"m{i}_2026"): 1 for i in range(6)})
    rows, _pres, _ = judged(clips, ALL_NEW)
    with pytest.raises(JudgementError, match="judged twice"):
        decide(rows + rows[:1], INC, NEW, clips)


def test_a_missing_lost_speech_flag_refuses_rather_than_assuming_false():
    clips = clips_over({("athens", f"m{i}_2026"): 1 for i in range(6)})
    rows, _pres, _ = judged(clips, ALL_NEW)
    rows[0]["lost_speech"] = {}
    with pytest.raises(JudgementError, match="cannot be computed"):
        lost_speech_counts(rows, INC, NEW)


# ==========================================================================
# The frozen rule: the truth table
# ==========================================================================

def test_a_clean_sweep_by_the_new_arm_wins():
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})
    rows, _pres, _ = judged(clips, ALL_NEW)
    out = decide(rows, INC, NEW, clips, n_bootstrap=400)
    assert out["win_rate"] == 1.0
    assert out["ci_lo"] > 0.5
    assert out["verdict"] == "NEW ARM WINS" and out["new_arm_wins"] is True


def test_a_clean_sweep_by_the_incumbent_is_a_loss_not_a_win():
    """An interval entirely below 50% also 'excludes 50%'. The rule is the LOWER bound."""
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})
    rows, _pres, _ = judged(clips, ALL_INC)
    out = decide(rows, INC, NEW, clips, n_bootstrap=400)
    assert out["ci_excludes_50"] is True
    assert out["ci_lower_above_50"] is False
    assert out["verdict"] == "NO WIN"


def test_all_ties_score_exactly_half_and_never_win():
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})
    rows, _pres, _ = judged(clips, ALL_TIE)
    out = decide(rows, INC, NEW, clips, n_bootstrap=400)
    assert TIE_RULE == "half"
    assert out["win_rate"] == 0.5 and out["tie_share"] == 1.0
    assert out["decisive_win_rate"] is None and out["n_decisive"] == 0
    assert out["ci_lo"] == out["ci_hi"] == 0.5
    assert out["verdict"] == "NO WIN"


def test_a_coin_flip_does_not_win():
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})
    rows, _pres, _ = judged(clips, COINFLIP)
    out = decide(rows, INC, NEW, clips, n_bootstrap=600)
    assert out["ci_lo"] <= 0.5 <= out["ci_hi"]
    assert out["verdict"] == "NO WIN"


def test_a_sweep_paid_for_with_silence_does_not_win():
    """Lower error bought by dropping speech is a failed arm, not a trade-off."""
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})

    def lost(i, p):
        return {s: p["sides"][s] == NEW and i < 10 for s in ("A", "B")}

    rows, _pres, _ = judged(clips, ALL_NEW, lost=lost)
    out = decide(rows, INC, NEW, clips, n_bootstrap=400)
    assert out["ci_lower_above_50"] is True
    assert out["lost_speech"] == {"incumbent": 0, "new": 10, "both": 0,
                                  "new_only": 10, "incumbent_only": 0, "n": 40,
                                  "rose": True}
    assert out["lost_speech_guard_passed"] is False
    assert out["verdict"] == "NO WIN"


def test_equal_lost_speech_does_not_block_a_win():
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})
    rows, _pres, _ = judged(clips, ALL_NEW,
                            lost=lambda i, p: {"A": i < 5, "B": i < 5})
    out = decide(rows, INC, NEW, clips, n_bootstrap=400)
    assert out["lost_speech"]["rose"] is False
    assert out["verdict"] == "NEW ARM WINS"


# ==========================================================================
# The bootstrap
# ==========================================================================

def test_the_cluster_key_is_city_and_meeting_not_the_meeting_slug():
    clips = (clips_over({("athens", "mar23_2026"): 2}) +
             clips_over({("chania", "mar23_2026"): 2}))
    rows, _pres, _ = judged(clips, ALL_NEW)
    ci = win_rate_ci(rows, NEW, n_bootstrap=50)
    assert ci["cluster_key"] == "(city, meeting_id)"
    assert ci["n_meetings"] == 2
    assert ci["meetings_by_city"] == {"athens": 1, "chania": 1}


def test_two_clips_of_one_meeting_are_one_cluster_and_widen_the_interval():
    """40 judgements over 20 meetings must not be treated as 40 independent trials.

    The outcome is keyed on the MEETING, so the two clips of one meeting agree -- which
    is the correlation clustering exists to price. With the same 40 judgements, 20
    clusters must give a wider interval than 40.
    """
    def by_meeting(i, p):
        n = int(p["meeting"].rsplit("/m", 1)[1].split("_")[0])
        return NEW if n % 2 else INC

    spread = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})
    packed = clips_over({("chania", f"m{i}_2026"): 2 for i in range(20)})
    a, _, _ = judged(spread, by_meeting)
    b, _, _ = judged(packed, by_meeting)
    wide = win_rate_ci(b, NEW, n_bootstrap=2000)
    narrow = win_rate_ci(a, NEW, n_bootstrap=2000)
    assert wide["n_judgements"] == narrow["n_judgements"] == 40
    assert wide["n_meetings"] == 20 and narrow["n_meetings"] == 40
    assert (wide["ci_hi"] - wide["ci_lo"]) > (narrow["ci_hi"] - narrow["ci_lo"])


def test_the_bootstrap_is_stratified_by_city_and_keeps_the_frozen_mixture():
    clips = (clips_over({("chania", f"c{i}_2026"): 1 for i in range(20)}) +
             clips_over({("vrilissia", f"v{i}_2026"): 1 for i in range(3)}))
    rows, _pres, _ = judged(clips, ALL_NEW)
    ci = win_rate_ci(rows, NEW, n_bootstrap=100)
    assert ci["bootstrap"].startswith("city-stratified")
    assert ci["meetings_by_city"] == {"chania": 20, "vrilissia": 3}
    assert ci["city_weights"] == {"chania": 20, "vrilissia": 3}
    assert city_weights(rows) == {"chania": 20, "vrilissia": 3}


def test_the_interval_is_reproducible_and_independent_of_row_order():
    """More than one city on purpose: the bootstrap draws city by city, so an iteration
    that followed row order would make the interval depend on the order the listener
    happened to answer in."""
    clips = (clips_over({("chania", f"m{i}_2026"): 1 for i in range(20)}) +
             clips_over({("athens", f"a{i}_2026"): 1 for i in range(10)}))
    rows, _pres, _ = judged(clips, COINFLIP)
    a = win_rate_ci(rows, NEW, n_bootstrap=500)
    b = win_rate_ci(list(reversed(rows)), NEW, n_bootstrap=500)
    assert (a["ci_lo"], a["ci_hi"], a["win_rate"]) == (b["ci_lo"], b["ci_hi"],
                                                       b["win_rate"])


def test_the_decisive_only_rate_is_reported_but_cannot_be_the_deciding_number():
    """Half the clips a tie, the rest won by the new arm: decisive rate 1.0, frozen
    rate 0.75. The verdict must be computed from the frozen one."""
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(40)})
    rows, _pres, _ = judged(clips, lambda i, p: NEW if i % 2 else None)
    out = decide(rows, INC, NEW, clips, n_bootstrap=400)
    assert out["decisive_win_rate"] == 1.0
    assert out["win_rate"] == 0.75
    assert out["tie_rule"] == "half"
    import inspect
    params = inspect.signature(decide).parameters
    assert "tie_rule" not in params, "the tie rule is frozen, not an argument"
    assert not hasattr(BP, "win_rate_ties_as_half"), \
        "there must be no second scorer that decide() could be pointed at"


def test_the_rule_the_estimand_and_the_precision_cost_travel_with_every_verdict():
    """The 15-cluster cost has to ride on the number. A caveat that lives only in a
    report is a caveat nobody re-reads next to the result."""
    clips = clips_over({("chania", f"m{i}_2026"): 1 for i in range(12)})
    rows, _pres, _ = judged(clips, ALL_NEW)
    out = decide(rows, INC, NEW, clips, n_bootstrap=100)
    assert "lower bound" in out["rule"] and "lost-speech" in out["rule"]
    assert "meetings drawn evenly" in out["estimand"]
    assert "NOT a uniformly random second of council audio" in out["estimand"]
    assert "15 meeting clusters" in out["precision_note"]
    assert "different order of precision" in out["precision_note"]
    assert out["incumbent_arm"] == INC and out["new_arm"] == NEW


def test_the_challenge_table_is_labelled_a_training_side_diagnostic():
    clips = clips_over({("athens", f"m{i}_2026"): 1 for i in range(12)},
                       sample="challenge")
    rows, _pres, _ = judged(clips, ALL_NEW, sample="challenge")
    table = challenge_table(rows, INC, NEW, clips)
    assert table["role"] == "training-side diagnostic"
    assert "TRAINING-SIDE DIAGNOSTIC" in table["estimand"]
    assert "never summed with the ordinary sample" in table["may_support"]
    assert "verdict" not in table
