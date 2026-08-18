from __future__ import annotations

import importlib.util
import json

import pytest

from eval.controlled_eval import conf_substrate as C
from notebooks import decode_ablation as DA


def test_resolved_config_is_frozen_control_plus_word_timestamps():
    expected = dict(DA.CONTROL, word_timestamps=True)
    assert C.resolved_config() == expected
    assert {key for key in C.resolved_config() if key not in DA.CONTROL} == set()
    assert C.resolved_config()["word_timestamps"] is True


def test_decode_refuses_to_extend_different_provenance():
    expected = {
        "model": "/model",
        "model_sha256_16": "digest",
        "config": {"beam_size": 5},
        "environment": {"python": "3.12"},
        "windows": {},
    }
    for key, changed in (
        ("model", "/other-model"),
        ("model_sha256_16", "other-digest"),
        ("config", {"beam_size": 2}),
        ("environment", {"python": "3.13"}),
    ):
        state = dict(expected)
        state[key] = changed
        with pytest.raises(SystemExit, match="different"):
            C.validate_resume(state, expected)


def test_two_window_smoke_persists_words_and_valid_probabilities():
    model = C.model_path()
    report = C.benchmark_report_path()
    if not model.is_dir() or not (model / "model.bin").is_file() or not report.is_file():
        pytest.skip(
            "two-window decoder smoke skipped: CT2 model directory or benchmark "
            "report cache is not present"
        )
    if importlib.util.find_spec("ctranslate2") is None or importlib.util.find_spec(
        "faster_whisper"
    ) is None:
        pytest.skip("two-window decoder smoke skipped: CT2 Python dependencies are absent")
    try:
        C.cache_dir().mkdir(parents=True, exist_ok=True)
    except OSError:
        pytest.skip(
            "two-window decoder smoke skipped: the required home cache directory "
            "is not writable in this environment"
        )

    path = C.decode(limit=2)
    state = json.loads(path.read_text())
    for item in C.substrate_items(limit=2):
        window = state["windows"][item["item_id"]]
        assert window["segments"]
        for segment in window["segments"]:
            assert isinstance(segment["words"], list)
            for word in segment["words"]:
                assert set(word) == {"w", "s", "e", "p"}
                assert 0 < word["p"] <= 1


def test_msa_alignment_cache_key_is_frozen():
    assert C._alignment_cache_guard().name == "align_65b1c4d64618a429.json"


def test_column_confidence_is_present_only_for_adapter_contributions():
    columns = [("a", "a", "b"), ("x", None, None), (None, "y", None), ("z", "z", "z")]
    confidence = C.column_confidences(columns, ["b", "z"], [0.25, 0.75])
    assert confidence == [0.25, None, None, 0.75]
