"""The rendered document. Invented Greek only, never council speech.

The charset test is not decoration: the page shipped once without one and every
Greek character arrived as mojibake over http.
"""
import json

from eval.tsfusion import page

T0 = 100.0


def bundle():
    def row(i, **kw):
        r = {"i": i, "window": "win_test", "scribe": None, "soniox": None,
             "whisper": None, "w": None, "w_reason": "unanimous", "ref": None,
             "agree": False, "occupancy": 0, "time_start": None,
             "time_end": None, "time_method": "observed",
             "time_uncertainty": 0.2, "time_conflict": False,
             "conflict_gap": None, "unresolved": False, "unresolved_reason": "",
             "sources": {}, "page_t": None, "speaker_state": "named",
             "speaker": "SPEAKER_01", "overlap_fraction": 1.0,
             "speaker_runner_up": None, "multiplicity": 1, "phase30": None,
             "phase30_meeting": None, "in_seam": False, "sys_op": {}}
        r.update(kw)
        return r

    def src(a, b):
        return {"start": a, "end": b, "provenance": "observed_word",
                "match": "stable", "envelope": [a, b], "conf": 0.9}

    words = ["θαλασσα", "βουνο", "ποταμι", "δασος"]
    rows = []
    for k, wrd in enumerate(words):
        rows.append(row(k, scribe=wrd, soniox=wrd, whisper=wrd, w=wrd,
                        agree=True, occupancy=3,
                        sources={"soniox": src(101.0 + k, 101.4 + k)},
                        time_start=101.0 + k, time_end=101.4 + k,
                        ref={"word": wrd, "op": "equal", "ambiguous": False,
                             "ref_index": k}))
    # one column carrying a token that would close the script tag if unescaped
    rows.append(row(4, scribe="</script>", soniox="</script>",
                    whisper="</script>", w="</script>", agree=True, occupancy=3,
                    sources={"soniox": src(105.0, 105.4)},
                    time_start=105.0, time_end=105.4,
                    ref={"word": None, "op": "insert", "ambiguous": False,
                         "ref_index": None}))
    return {
        "manifest": {
            "page_start_abs": T0, "page_end_abs": T0 + 10.0,
            "page_duration": 10.0, "city": "Δοκιμή", "meeting": "συνεδρίαση 1",
            "generated_at": "2026-08-18T00:00:00+00:00",
            "seam_abs": [T0 + 5.0, T0 + 5.1],
            "msa_sha256_16": "0000000000000000",
            "scoring_normalisation": "δοκιμή",
            "soniox_timestamp_model": "δοκιμή",
            "whisper_timestamp_decode": "δοκιμή",
            "thresholds": {"diarization_pad_s": 5.0},
            "audio_check": {"checks": [{"lag_ms": 0, "peak_corr": 0.99}]},
        },
        "diar": {
            "exclusive": [{"speaker": "SPEAKER_01", "start": T0, "end": T0 + 10,
                           "page_start": 0.0, "page_end": 10.0}],
            "regular": [{"speaker": "SPEAKER_01", "start": T0, "end": T0 + 10,
                         "page_start": 0.0, "page_end": 10.0}],
            "speakers": ["SPEAKER_01"],
        },
        "rows": rows,
        "deletions": [],
        "conditions": [],
        "per_system": {
            s: {"counts": {"equal": 4, "sub": 0, "delete": 0, "insert": 1,
                           "ambiguous": 0},
                "distance": 1, "n_hyp": 5, "wer": 0.25}
            for s in ("scribe", "soniox", "whisper", "W")},
        "seam_page": [5.0, 5.1],
    }


def html():
    return page.render(bundle())


def test_it_is_a_complete_document_with_a_charset():
    h = html()
    assert h.startswith("<!doctype html>")
    assert '<html lang="el">' in h
    assert '<meta charset="utf-8">' in h
    assert 'name="viewport"' in h
    assert "<title>" in h and "</title>" in h
    assert h.rstrip().endswith("</html>")
    assert h.index('<meta charset="utf-8">') < h.index("<title>")


def test_the_title_survives_a_utf8_round_trip():
    h = html()
    assert "Διαγνωστικό" in h.encode("utf-8").decode("utf-8")


def test_no_em_or_en_dashes_anywhere():
    h = html()
    assert "—" not in h
    assert "–" not in h


def test_a_token_cannot_close_the_payload_script_tag():
    h = html()
    payload = h.split('id="payload">')[1].split("</script>")[0]
    assert "</script>" not in payload
    assert json.loads(payload.encode().decode("unicode_escape")
                      if False else payload)


def test_the_words_are_rendered_as_flowing_text_in_a_turn_card():
    h = html()
    assert 'class="turn"' in h
    assert 'class="flow"' in h
    for w in ("θαλασσα", "βουνο", "ποταμι", "δασος"):
        assert f'>{w}</span>' in h


def test_the_filter_bar_opens_on_the_whole_text():
    # The page is read first and filtered second: starting on a filter hid most
    # of the speech before the reader had seen any of it.
    h = html()
    for f in ("all", "dis", "drift", "omit", "wrong", "loss",
              "scribe", "soniox", "whisper"):
        assert f'data-filter="{f}"' in h
    assert 'applyFilter("all")' in h


def test_there_is_a_follow_toggle_for_the_karaoke():
    h = html()
    assert 'id="follow"' in h
    assert "setFollow(false)" in h


def test_the_detail_panel_is_an_anchored_popover():
    h = html()
    assert 'pop.className = "pop"' in h
    assert "placePop" in h                       # flips above or below
    assert 'e.key === "Escape"' in h


def test_every_system_has_its_own_error_line():
    h = html()
    assert "Τι λάθη κάνει το κάθε σύστημα χωριστά" in h
    for name in ("Scribe v2", "Soniox", "Το μοντέλο μας", "W (σύνθεση)"):
        assert name in h
    assert "(Α+Ε)/Ν" in h


def test_a_deletion_after_the_final_column_is_still_shown():
    b = bundle()
    b["deletions"] = [{"system": "W", "word": "λιμνη", "after_column": 4,
                       "method": "bracketed", "note": "",
                       "page_t": None, "page_t_end": None}]
    h = page.render(b)
    assert "λιμνη" in h


def test_the_audio_stays_a_sibling_file():
    h = html()
    assert '<audio id="au" controls preload="metadata" src="page.mp3">' in h
    assert "base64" not in h
