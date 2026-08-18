"""Derived render-time facts. Every word here is invented, never council speech."""
from eval.tsfusion import derive

T0 = 100.0


def row(i, **kw):
    r = {
        "i": i, "window": "win_test", "scribe": None, "soniox": None,
        "whisper": None, "w": None, "w_reason": "unanimous", "ref": None,
        "agree": False, "occupancy": 0, "time_start": None, "time_end": None,
        "time_method": "unplaced", "time_uncertainty": None,
        "time_conflict": False, "conflict_gap": None, "unresolved": False,
        "unresolved_reason": "", "sources": {}, "page_t": None,
        "speaker_state": "named", "speaker": "SPEAKER_01",
        "overlap_fraction": 1.0, "speaker_runner_up": None, "multiplicity": 1,
        "phase30": None, "phase30_meeting": None, "in_seam": False,
        "sys_op": {},
    }
    r.update(kw)
    return r


def src(start, end, conf=0.9):
    return {"start": start, "end": end, "provenance": "observed_word",
            "match": "stable", "envelope": [start, end], "conf": conf}


# ------------------------------------------------------------------ anchors
def test_anchor_takes_the_earliest_observed_source_not_the_midpoint():
    # A conflicted column: soniox heard the word early, whisper places it late.
    r = row(0, sources={"soniox": src(101.0, 101.4), "whisper": src(104.0, 104.1)},
            time_start=101.0, time_end=104.1, time_method="observed",
            time_uncertainty=3.1, time_conflict=True, conflict_gap=2.6,
            page_t=2.55)
    a = derive.anchor_of(r, T0)
    assert a["kind"] == "observed_source"
    assert a["source"] == "soniox"
    assert a["page_start"] == 1.0
    assert abs(a["page_end"] - 1.4) < 1e-9
    assert a["duration"] == 0.4
    assert a["reliable"] is True
    # the midpoint the old page used sits a full 1.5 s later
    assert r["page_t"] > a["page_start"] + 1.0


def test_anchor_falls_back_to_the_bracket_representative_then_to_nothing():
    b = derive.anchor_of(row(1, time_method="bracketed", page_t=4.0), T0)
    assert b["kind"] == "bracketed" and b["page_start"] == 4.0
    assert b["reliable"] is False and b["duration"] is None
    x = derive.anchor_of(row(2, time_method="extrapolated", page_t=5.0), T0)
    assert x["kind"] == "extrapolated" and x["reliable"] is False
    n = derive.anchor_of(row(3), T0)
    assert n["kind"] == "none" and n["page_start"] is None


def test_anchor_ignores_a_reversed_or_missing_end():
    a = derive.anchor_of(row(0, sources={"soniox": src(101.0, 100.5)}), T0)
    assert a["page_start"] == 1.0 and a["page_end"] is None
    assert a["reliable"] is False


def test_displayed_anchors_are_non_decreasing_except_where_flagged():
    rows = [
        row(0, sources={"soniox": src(101.0, 101.2)}),
        row(1, sources={"soniox": src(101.3, 101.5),
                        "whisper": src(104.0, 104.1)},
            time_conflict=True, page_t=2.55),
        row(2, sources={"soniox": src(101.6, 101.8)}),
        row(3, sources={"whisper": src(101.4, 101.5)}),   # genuinely backwards
        row(4, sources={"soniox": src(102.0, 102.2)}),
    ]
    anchors = derive.anchors(rows, T0)
    assert [a["backwards"] for a in anchors] == [False, False, False, True, False]
    assert anchors[3]["regression"] == 0.2
    running = None
    for a in anchors:
        p = a["page_start"]
        if p is None:
            continue
        if running is not None and not a["backwards"]:
            assert p >= running - 1e-9
        running = p if running is None else max(running, p)


def test_the_regression_flag_is_against_the_running_maximum():
    # Row 2 is later than row 1 but still earlier than row 0: still a regression.
    rows = [
        row(0, sources={"soniox": src(105.0, 105.1)}),
        row(1, sources={"soniox": src(101.0, 101.1)}),
        row(2, sources={"soniox": src(102.0, 102.1)}),
    ]
    a = derive.anchors(rows, T0)
    assert [x["backwards"] for x in a] == [False, True, True]


# ------------------------------------------------------------------ overlap
def turn(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end}


def overlap_row(**kw):
    r = row(0, speaker_state="overlap", speaker="SPEAKER_01", multiplicity=2,
            sources={"soniox": src(101.0, 102.0)})
    r.update(kw)
    return r


def call(r, regular):
    return derive.overlap_call(r, derive.anchor_of(r, T0), regular)


def test_a_grazing_second_turn_is_not_overlap():
    r = overlap_row()
    c = call(r, [turn("SPEAKER_01", 90.0, 105.0), turn("SPEAKER_02", 101.9, 102.05)])
    assert c["evidence"] == "not_confirmed"
    assert c["seconds"] == 0.1                      # the near miss is kept
    assert c["fraction"] == 0.1
    assert c["speaker"] == "SPEAKER_02"


def test_a_real_second_speaker_is_confirmed():
    r = overlap_row()
    c = call(r, [turn("SPEAKER_02", 101.0, 101.6)])
    assert c["evidence"] == "confirmed"
    assert c["seconds"] == 0.6 and c["fraction"] == 0.6


def test_both_thresholds_must_be_met():
    # 0.35 s of a 2 s word: over the absolute floor, under the fraction floor.
    r = overlap_row(sources={"soniox": src(101.0, 103.0)})
    assert call(r, [turn("SPEAKER_02", 101.0, 101.35)])["evidence"] \
        == "not_confirmed"
    # 0.2 s of a 0.4 s word: half the word, but below the absolute floor.
    r = overlap_row(sources={"soniox": src(101.0, 101.4)})
    assert call(r, [turn("SPEAKER_02", 101.0, 101.2)])["evidence"] \
        == "not_confirmed"


def test_fragments_of_one_speaker_are_merged_before_measuring():
    r = overlap_row()
    frags = [turn("SPEAKER_02", 101.0, 101.5), turn("SPEAKER_02", 101.2, 101.4)]
    c = call(r, frags)
    assert c["seconds"] == 0.5                      # not 0.7


def test_the_same_speaker_never_counts_as_the_second_one():
    r = overlap_row()
    c = call(r, [turn("SPEAKER_01", 101.0, 102.0)])
    assert c["evidence"] == "not_confirmed" and c["seconds"] == 0.0


def test_a_column_without_an_interval_is_unassessable():
    r = overlap_row(sources={}, time_method="bracketed", page_t=1.0)
    assert call(r, [turn("SPEAKER_02", 101.0, 102.0)])["evidence"] == "unassessable"


def test_a_column_that_was_never_overlap_is_left_alone():
    r = row(0, sources={"soniox": src(101.0, 102.0)})
    assert call(r, [turn("SPEAKER_02", 101.0, 102.0)])["evidence"] \
        == "not_applicable"


# -------------------------------------------------------- reference omission
def ins(i=0, **kw):
    r = row(i, ref={"word": None, "op": "insert", "ambiguous": False,
                    "ref_index": None})
    r.update(kw)
    return r


def test_two_systems_proposing_the_same_word_make_it_a_suspect_omission():
    assert derive.ref_omission_suspect(
        ins(scribe="θαλασσα", soniox="θαλασσα", whisper=None, w="θαλασσα"))


def test_one_system_alone_is_not_a_suspect_omission():
    assert not derive.ref_omission_suspect(
        ins(scribe="θαλασσα", soniox=None, whisper=None, w="θαλασσα"))


def test_the_test_is_occupancy_not_agreement_on_the_word():
    # Two systems heard SOMETHING here and the published text has nothing. That
    # they disagree about which word it was does not make the speech go away.
    assert derive.ref_omission_suspect(
        ins(scribe="βουνο", soniox=None, whisper="θαλασσα", w="θαλασσα"))
    assert derive.ref_omission_suspect(
        ins(scribe="βουνο", soniox="βουνο", whisper="θαλασσα", w="θαλασσα"))


def test_only_insertions_can_be_suspect_omissions():
    r = ins(scribe="θαλασσα", soniox="θαλασσα", whisper="θαλασσα", w="θαλασσα")
    r["ref"] = {"word": "θαλασσα", "op": "equal", "ambiguous": False,
                "ref_index": 0}
    assert not derive.ref_omission_suspect(r)
    r["ref"] = {"word": "βουνο", "op": "sub", "ambiguous": False, "ref_index": 0}
    assert not derive.ref_omission_suspect(r)


def test_an_empty_W_is_never_a_suspect_omission():
    assert not derive.ref_omission_suspect(
        ins(scribe=None, soniox=None, whisper=None, w=None))


# ------------------------------------------------------------------- census
def test_the_three_error_categories_reconcile_to_the_charged_edits():
    rows = [
        row(0, scribe="κυριος", soniox="κυριος", whisper="κυριος", w="κυριος",
            agree=True,
            ref={"word": "κ", "op": "sub", "ambiguous": False, "ref_index": 0}),
        row(1, scribe="θαλασσα", soniox="θαλασσα", whisper=None, w="θαλασσα",
            ref={"word": None, "op": "insert", "ambiguous": False,
                 "ref_index": None}),
        row(2, scribe="βουνο", soniox="βουνο", whisper="βουνα", w="βουνο",
            ref={"word": "βουνα", "op": "sub", "ambiguous": False,
                 "ref_index": 1}),
    ]
    data = {"rows": rows,
            "deletions": [{"system": "W", "word": "ν", "after_column": 1,
                           "method": "bracketed", "note": ""},
                          {"system": "scribe", "word": "αλλο",
                           "after_column": 1, "method": "bracketed", "note": ""}]}
    c = derive.error_census(data, rows)
    assert c["counts"] == {"convention": 2, "ref_omission": 1, "ours": 1,
                           "total": 4}


def test_a_published_word_some_system_produced_is_not_a_convention():
    vocab = {"κ", "θαλασσα"}
    assert not derive.is_publication_convention("κ", vocab)
    assert derive.is_publication_convention("κ", {"θαλασσα"})
    assert not derive.is_publication_convention("θαλασσες", {"θαλασσα"})


# -------------------------------------------------------------------- shape
def test_build_view_does_not_mutate_the_bundle():
    import copy
    data = {
        "manifest": {"page_start_abs": T0, "page_duration": 10.0},
        "diar": {"regular": [], "exclusive": [], "speakers": []},
        "rows": [row(0, sources={"soniox": src(101.0, 101.2)}, w="θαλασσα",
                     time_method="observed", agree=True)],
        "deletions": [],
        "per_system": {s: {"counts": {"equal": 1, "sub": 0, "delete": 0,
                                      "insert": 0, "ambiguous": 0},
                           "distance": 0, "n_hyp": 1, "wer": 0.0}
                       for s in ("scribe", "soniox", "whisper", "W")},
    }
    before = copy.deepcopy(data)
    view = derive.build_view(data)
    assert data == before
    assert view["summary"]["ref_tokens"] == 1
    assert len(view["rows"]) == 1


# ----------------------------------------------------- monotone interpolation
def anchored(i, start, **kw):
    return row(i, sources={"soniox": src(T0 + start, T0 + start + 0.2)},
               time_method="observed", time_conflict=False, **kw)


def place(rows, page_duration=None):
    return derive.positions(rows, derive.anchors(rows, T0), page_duration)


def test_a_conflicted_column_no_longer_appears_before_its_neighbour():
    # The bug: the middle column's own interval starts LATER than the next
    # column's, so ordering by it walked backwards.
    rows = [anchored(0, 1.0),
            row(1, sources={"soniox": src(T0 + 5.0, T0 + 5.2)},
                time_method="observed", time_conflict=True, conflict_gap=3.0),
            anchored(2, 2.0)]
    p = place(rows)
    assert [x["anchor"] for x in p] == [True, False, True]
    assert p[0]["t"] == 1.0 and p[2]["t"] == 2.0
    assert p[1]["t"] == 1.5                       # halfway, not 5.0
    assert p[1]["interpolated"] is True


def test_a_run_of_non_anchors_is_spread_evenly():
    rows = [anchored(0, 0.0),
            row(1, time_method="bracketed", page_t=9.0),
            row(2, time_method="bracketed", page_t=9.0),
            row(3, time_method="bracketed", page_t=9.0),
            anchored(4, 4.0)]
    t = [x["t"] for x in place(rows)]
    assert t == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_every_displayed_time_is_non_decreasing():
    rows = [anchored(0, 3.0),
            anchored(1, 1.0),                     # two anchors out of order
            row(2, time_method="observed", time_conflict=True,
                sources={"soniox": src(T0 + 0.5, T0 + 0.6)}),
            anchored(3, 5.0),
            row(4, time_method="unplaced")]
    p = place(rows, page_duration=10.0)
    t = [x["t"] for x in p]
    assert all(b >= a for a, b in zip(t, t[1:])), t
    assert p[1]["clamped"] is True and p[1]["t"] == 3.0


def test_columns_outside_the_anchors_are_spread_from_the_page_edges():
    rows = [row(0, time_method="unplaced"), anchored(1, 2.0),
            row(2, time_method="unplaced")]
    p = place(rows, page_duration=6.0)
    assert 0.0 <= p[0]["t"] < 2.0
    assert 2.0 < p[2]["t"] <= 6.0
    assert [x["interpolated"] for x in p] == [True, False, True]


def test_a_page_with_no_anchor_at_all_still_produces_times():
    rows = [row(0, time_method="unplaced"), row(1, time_method="unplaced")]
    p = place(rows, page_duration=4.0)
    assert [x["t"] for x in p] == [0.0, 4.0]
    assert all(x["interpolated"] for x in p)


def test_the_highlight_never_outlives_the_cap():
    rows = [anchored(0, 0.0), anchored(1, 100.0)]
    p = place(rows)
    assert p[0]["end"] == derive.MAX_WORD_SECONDS
    assert p[1]["end"] >= p[1]["t"] + derive.MIN_WORD_SECONDS


# --------------------------------------------------- overlap, at the threshold
def test_overlap_exactly_at_both_thresholds_is_confirmed():
    r = overlap_row(sources={"soniox": src(101.0, 102.0)})
    c = call(r, [turn("SPEAKER_02", 101.0, 101.3)])
    assert c["seconds"] == 0.3 and c["fraction"] == 0.3
    assert c["evidence"] == "confirmed"


# ------------------------------------------------------- per system breakdown
def test_the_per_system_table_reports_S_D_I_and_the_sd_rate():
    data = {"per_system": {
        "scribe": {"counts": {"equal": 8, "sub": 1, "delete": 1, "insert": 2,
                              "ambiguous": 0}, "n_hyp": 11},
        "soniox": {"counts": {"equal": 9, "sub": 1, "delete": 0, "insert": 0,
                              "ambiguous": 0}, "n_hyp": 10},
        "whisper": {"counts": {"equal": 7, "sub": 2, "delete": 1, "insert": 0,
                               "ambiguous": 0}, "n_hyp": 9},
        "W": {"counts": {"equal": 9, "sub": 1, "delete": 0, "insert": 1,
                         "ambiguous": 0}, "n_hyp": 11}}}
    rows = [ins(i=0, scribe="θαλασσα", soniox="θαλασσα", whisper=None,
                w="θαλασσα",
                sys_op={"scribe": {"op": "insert"}, "soniox": {"op": "insert"}})]
    t = {x["key"]: x for x in derive.per_system_table(data, rows)}
    assert t["scribe"]["S"] == 1 and t["scribe"]["D"] == 1
    assert abs(t["whisper"]["sd_rate"] - 3 / 10) < 1e-9
    assert abs(t["W"]["wer"] - 2 / 10) < 1e-9
    # the one insertion sits on a column two systems occupied
    assert t["W"]["suspect_insertions"] == 1
    assert abs(t["W"]["wer_excl_suspect"] - 1 / 10) < 1e-9
    assert t["scribe"]["suspect_insertions"] == 1        # clamped, not 2


def test_a_selection_loss_is_W_wrong_where_a_voter_was_right():
    r = row(0, scribe="βουνο", soniox="θαλασσα", whisper="θαλασσα",
            w="θαλασσα",
            ref={"word": "βουνο", "op": "sub", "ambiguous": False,
                 "ref_index": 0},
            sys_op={"scribe": {"op": "equal"}, "soniox": {"op": "sub"},
                    "whisper": {"op": "sub"}})
    e = derive.system_error_rows(r)
    assert e["wrong"] == ["soniox", "whisper"] and e["right"] == ["scribe"]
    assert e["w_wrong"] is True and e["selection_loss"] is True
    # nobody was right: W is wrong but nothing was lost in selection
    r["scribe"] = "λιμνη"
    r["sys_op"]["scribe"] = {"op": "sub"}
    assert derive.system_error_rows(r)["selection_loss"] is False


# --------------------------------------------------------------- warnings
def test_a_grazing_overlap_produces_no_warning_at_all():
    r = overlap_row()
    ov = call(r, [turn("SPEAKER_02", 101.9, 102.05)])
    w = derive.warnings_for(r, ov, {"interpolated": False})
    assert [x["k"] for x in w] == []


def test_a_confirmed_overlap_carries_its_seconds_and_share():
    r = overlap_row()
    ov = call(r, [turn("SPEAKER_02", 101.0, 101.6)])
    w = derive.warnings_for(r, ov, {"interpolated": False})
    assert w[0]["k"] == "overlap" and w[0]["seconds"] == 0.6
    assert w[0]["speaker"] == "SPEAKER_02"


def test_only_a_wide_interval_is_called_uncertain():
    quiet = derive.warnings_for(row(0, time_uncertainty=0.72),
                                {"evidence": "not_applicable"},
                                {"interpolated": False})
    assert [x["k"] for x in quiet] == []
    loud = derive.warnings_for(
        row(0, time_uncertainty=derive.WIDE_UNCERTAINTY_SECONDS),
        {"evidence": "not_applicable"}, {"interpolated": False})
    assert [x["k"] for x in loud] == ["wide"]
    assert loud[0]["seconds"] == derive.WIDE_UNCERTAINTY_SECONDS


def test_a_straddling_word_names_both_speakers():
    r = row(0, speaker_state="ambiguous", speaker="SPEAKER_01",
            speaker_runner_up="SPEAKER_02", overlap_fraction=0.6)
    w = derive.warnings_for(r, {"evidence": "not_applicable"},
                            {"interpolated": False})
    assert w[0]["k"] == "straddle" and w[0]["runner_up"] == "SPEAKER_02"
    assert w[0]["coverage"] == 0.6


# ------------------------------------------------------------------- cards
def test_a_card_follows_the_speaker_and_not_the_state():
    # One handover word the diarization could not resolve must not cut the
    # speech into three cards.
    rows = [{"i": 0, "window": "w", "speaker": "SPEAKER_01",
             "display_speaker_state": "named", "pos": {"t": 0.0, "end": 0.4},
             "anchor": {"page_start": 0.0, "page_end": 0.4}},
            {"i": 1, "window": "w", "speaker": None,
             "display_speaker_state": "unresolved",
             "pos": {"t": 0.5, "end": 0.9},
             "anchor": {"page_start": 0.5, "page_end": 0.9}},
            {"i": 2, "window": "w", "speaker": "SPEAKER_01",
             "display_speaker_state": "named", "pos": {"t": 1.0, "end": 1.4},
             "anchor": {"page_start": 1.0, "page_end": 1.4}}]
    t = derive.turns(rows)
    assert len(t) == 1
    assert t[0]["rows"] == [0, 1, 2]
    assert t[0]["speaker"] == "SPEAKER_01"


# ------------------------------------------------------------------ ledger
def led_bundle():
    """Three columns and a fourth reference word nobody wrote."""
    words = ["θαλασσα", "βουνο", "ποταμι"]
    rows = []
    for k, wrd in enumerate(words):
        said = "λιμνη" if k == 1 else wrd
        rows.append(row(k, scribe=wrd, soniox=said, whisper=said, w=said,
                        time_method="observed",
                        sources={"soniox": src(101.0 + k, 101.4 + k)},
                        sys_op={"scribe": {"op": "equal"},
                                "soniox": {"op": "sub" if k == 1 else "equal"},
                                "whisper": {"op": "sub" if k == 1 else "equal"}},
                        ref={"word": wrd,
                             "op": "sub" if k == 1 else "equal",
                             "ambiguous": False, "ref_index": k}))
    counts = {s: {"equal": 2, "sub": 1, "delete": 1, "insert": 0,
                  "ambiguous": 0}
              for s in ("soniox", "whisper", "W")}
    counts["scribe"] = {"equal": 3, "sub": 0, "delete": 1, "insert": 0,
                        "ambiguous": 0}
    return {
        "manifest": {"page_start_abs": T0, "page_duration": 10.0},
        "diar": {"regular": [], "exclusive": [], "speakers": []},
        "rows": rows,
        "deletions": [{"system": s, "ref_index": 3, "word": "δασος",
                       "after_column": 2, "method": "bracketed", "note": "",
                       "page_t": 3.6, "page_t_end": 3.6}
                      for s in ("scribe", "soniox", "whisper", "W")],
        "per_system": {s: {"counts": counts[s],
                           "distance": 1 if s == "scribe" else 2,
                           "n_hyp": 3, "wer": 0.5}
                       for s in ("scribe", "soniox", "whisper", "W")},
    }


def test_the_reference_is_rebuilt_from_the_columns_and_W_deletions():
    ref = derive.reference_by_window(led_bundle())
    assert ref["win_test"] == ["θαλασσα", "βουνο", "ποταμι", "δασος"]


def test_the_ledger_lists_every_edit_of_every_system_with_context():
    data = led_bundle()
    view = derive.build_view(data)
    led = view["ledger"]
    # scribe got the substitution right, the other two did not
    subs = [e for e in led if e["type"] == "S"]
    assert {e["system"] for e in subs} == {"soniox", "whisper", "W"}
    one = subs[0]
    assert one["ref_word"] == "βουνο" and one["hyp_word"] == "λιμνη"
    assert one["left"] == "θαλασσα" and one["right"] == "ποταμι δασος"
    assert one["right_systems"] == ["scribe"]
    assert one["t"] == 2.0 and one["time_borrowed"] is False
    # every system deleted the third word, and every such row says the time is
    # not its own
    dels = [e for e in led if e["type"] == "D"]
    assert len(dels) == 4
    assert all(e["time_borrowed"] and e["time_method"] == "bracketed"
               for e in dels)
    assert all(e["ref_word"] == "δασος" and e["right_systems"] == []
               for e in dels)


def test_the_ledger_reconciles_with_the_per_system_counts():
    data = led_bundle()
    view = derive.build_view(data)
    for r in view["per_system"]:
        got = {t: sum(1 for e in view["ledger"]
                      if e["system"] == r["key"] and e["type"] == t)
               for t in "SDI"}
        assert got == {"S": r["S"], "D": r["D"], "I": r["I"]}, r["key"]


# ------------------------------------------------------------------ clocks
def test_the_clock_report_counts_straddles_and_disagreements():
    rows = [row(0, sources={"soniox": src(101.0, 101.4),
                            "whisper": src(101.9, 102.3)},
                time_method="observed", w="α"),
            row(1, sources={"soniox": src(102.0, 103.0)},
                time_method="observed", w="β")]
    data = {
        "manifest": {"page_start_abs": T0, "page_duration": 10.0},
        "diar": {"regular": [],
                 "exclusive": [
                     {"speaker": "SPEAKER_01", "start": T0, "end": T0 + 2.5,
                      "page_start": 0.0, "page_end": 2.5},
                     {"speaker": "SPEAKER_02", "start": T0 + 2.5,
                      "end": T0 + 5.0, "page_start": 2.5, "page_end": 5.0}],
                 "speakers": ["SPEAKER_01", "SPEAKER_02"]},
        "rows": rows, "deletions": [],
        "per_system": {s: {"counts": {"equal": 2, "sub": 0, "delete": 0,
                                      "insert": 0, "ambiguous": 0},
                           "distance": 0, "n_hyp": 2, "wer": 0.0}
                       for s in ("scribe", "soniox", "whisper", "W")},
    }
    c = derive.build_view(data)["clocks"]
    sx = c["systems"]["soniox"]
    assert sx["n_words"] == 2
    assert sx["straddling"] == 1          # 2.0 ως 3.0 crosses the 2.5 edge
    assert c["systems"]["whisper"]["straddling"] == 0
    assert c["n_speaker_changes"] == 1
    assert c["both_timed"] == 1
    assert c["disagree_over_threshold"] == 1   # 101.0 vs 101.9
    assert abs(c["median_start_gap"] - 0.9) < 1e-6
