import hashlib

import pytest

from eval import exp_run2_local as run2
from eval.controlled_eval.exp_same_stack import sdi as frozen_sdi
from eval.controlled_eval.scoring import cluster_bootstrap as frozen_bootstrap


FROZEN_CONFIG = {
    "language": "el",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "word_timestamps": False,
    "vad_filter": False,
    "task": "transcribe",
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
}


def test_emitted_decode_config_matches_frozen_dict_exactly():
    assert run2.decode_config() == FROZEN_CONFIG


def test_scoring_delegates_to_the_frozen_scorer(monkeypatch):
    calls = []
    assert run2.sdi is frozen_sdi
    assert run2.cluster_bootstrap is frozen_bootstrap

    def fake_sdi(ref, hyp):
        calls.append((ref, hyp))
        return 1, 2, 3

    monkeypatch.setattr(run2, "sdi", fake_sdi)
    result = run2.score_counts({"w": ["ref"]}, {"w": ["hyp"]})

    assert calls == [(["ref"], ["hyp"])]
    assert result == {"w": (1, 2, 3, 1)}


def test_adapter_hash_check_rejects_a_mismatched_file(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(b"adapter")
    expected = hashlib.sha256(b"different adapter").hexdigest()

    with pytest.raises(ValueError, match="sha256 mismatch"):
        run2.verify_adapter_hash(path, expected)


def test_paired_comparison_refuses_different_decode_configs():
    left = {"config": FROZEN_CONFIG, "scores": {}}
    right = {"config": {**FROZEN_CONFIG, "beam_size": 1}, "scores": {}}

    with pytest.raises(ValueError, match="different decode configs"):
        run2.paired_comparison(left, right, [], [])
