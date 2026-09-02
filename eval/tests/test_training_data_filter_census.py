import json

from eval.training_data_filter_census import census, classify, diagnostics


def _row(**kw):
    row = {
        "id": "r1",
        "text": "Καλημέρα κύριε Παπαδόπουλε",
        "text_asr": "καλημέρα κύριε παπαδόπουλε",
        "dur": 1.5,
        "align": 0.98,
        "edge_clipped": False,
    }
    row.update(kw)
    return row


def test_levels_are_nested_and_l3_missing_is_not_failure():
    assert classify(_row(), None) == {"L0": True, "L1": True, "L2": True,
                                      "L3": None}
    assert classify(_row(), 0.04)["L3"] is True
    assert classify(_row(), 0.06)["L3"] is False
    got = classify(_row(align=0.80), 0.0)
    assert got == {"L0": True, "L1": True, "L2": False, "L3": False}


def test_diagnostics_use_frozen_proxies():
    got = diagnostics(_row(), {"παπαδοπουλε"})
    assert got["known_roster_name"]
    assert got["capitalized_noninitial"]
    assert not got["fast_speech"]
    assert not got["hard_example"]

    hard = diagnostics(_row(text="ένα δύο τρία τέσσερα πέντε έξι", dur=1.5,
                            align=0.80), set())
    assert hard["fast_speech"] and hard["hard_example"]


def test_census_reports_unmeasurable_separately(tmp_path):
    pack = tmp_path / "train.jsonl"
    rows = [_row(id="a"), _row(id="b", align=0.80)]
    pack.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    result = census({"toy": pack}, {}, {"παπαδοπουλε"})
    assert result["levels"]["L0"]["by_source"]["toy"]["rows"] == 2
    assert result["levels"]["L2"]["by_source"]["toy"]["rows"] == 1
    l3 = result["levels"]["L3"]["by_source"]["toy"]
    assert l3["rows"] == 0
    assert l3["unmeasurable_rows"] == 1
    assert l3["row_retention_vs_L0"] is None
    assert result["equal_hours_design"]["toy"]["target_seconds_per_arm"] > 0
    assert "Καλημέρα" not in json.dumps(result, ensure_ascii=False)
