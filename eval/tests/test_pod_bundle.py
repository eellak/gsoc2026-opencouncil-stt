import json
from pathlib import Path

import pytest

from scripts.pod_bundle import build, verify
from notebooks.train_runpod import (
    RUN_FINGERPRINT,
    data_dir_source_identity,
    feature_cache_signature,
    resolve_pack_audio_paths,
)


def make_fixture(root: Path) -> tuple[Path, Path]:
    packs = root / "packs"
    audio = packs / "audio"
    audio.mkdir(parents=True)
    (audio / "pk_1.flac").write_bytes(b"fake-flac-one")
    (audio / "pk_2.flac").write_bytes(b"fake-flac-two")
    rows = [
        {"pack_id": "pk_1", "audio": str(audio / "pk_1.flac"),
         "text_p": "p1", "text_pn": "pn1", "dur_sec": 20.0},
        {"pack_id": "pk_2", "audio": "audio/pk_2.flac",
         "text_p": "p2", "text_pn": "pn2", "dur_sec": 22.0},
    ]
    (packs / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    data = root / "data"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"train")
    (data / "validation.parquet").write_bytes(b"validation")
    return packs, data


def test_bundle_is_portable_and_verifies(tmp_path):
    packs, data = make_fixture(tmp_path)
    result = build(packs, data, tmp_path / "out", "test", archive=False)
    metadata = verify(result)
    rows = [json.loads(line) for line in (result / "packs/manifest.jsonl").read_text().splitlines()]
    assert metadata["pack_rows"] == 2
    assert [row["audio"] for row in rows] == ["audio/pk_1.flac", "audio/pk_2.flac"]
    assert all(not Path(row["audio"]).is_absolute() for row in rows)
    assert (result / "code/eval/controlled_eval/bench_data.py").is_file()


def test_bundle_materializes_frozen_validation_reference_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    packs, data = make_fixture(tmp_path)
    cache = tmp_path / ".cache/oc-public"
    windows = cache / "bench_windows"
    windows.mkdir(parents=True)
    (windows / "w1.wav").write_bytes(b"wav")
    (cache / "bench_test-run.json").write_text(json.dumps({
        "items": [{"itemId": "w1", "referenceText": "Γεια σου"}],
    }))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "source_run": "test-run",
        "eval_windows": [{"window_id": "w1", "ref_tokens": 2,
                          "audio_local": True}],
    }))

    result = build(packs, data, tmp_path / "out", "test", archive=False,
                   eval_manifest=freeze)
    validation = json.loads((result / "eval/validation/manifest.json").read_text())
    assert validation["eval_windows"][0]["ref_tokens"] == ["γεια", "σου"]


def test_verify_rejects_tampering(tmp_path):
    packs, data = make_fixture(tmp_path)
    result = build(packs, data, tmp_path / "out", "test", archive=False)
    (result / "packs/audio/pk_1.flac").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(result)


def test_trainer_resolves_bundle_relative_audio(tmp_path):
    manifest = tmp_path / "packs/manifest.jsonl"
    rows = resolve_pack_audio_paths(
        [{"audio": "audio/p.flac"}, {"audio": "/legacy/p.flac"}], manifest)
    assert rows[0]["audio"] == str(tmp_path / "packs/audio/p.flac")
    assert rows[1]["audio"] == "/legacy/p.flac"


def test_feature_cache_signature_tracks_audio_text_split_and_config(tmp_path):
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"first")
    rows = [{"audio": str(audio), "text": "κείμενο"}]
    assert feature_cache_signature(rows, "train") == feature_cache_signature(rows, "train")
    assert feature_cache_signature(rows, "train") != feature_cache_signature(rows, "valc")
    assert feature_cache_signature(rows, "train") != feature_cache_signature(
        [{"audio": str(audio), "text": "άλλο"}], "train")
    assert feature_cache_signature(rows, "train", {"config": 1}) != feature_cache_signature(
        rows, "train", {"config": 2})
    before = feature_cache_signature(rows, "train")
    audio.write_bytes(b"other")
    assert feature_cache_signature(rows, "train") != before


def test_data_dir_identity_uses_parquet_content_not_only_size(tmp_path):
    train = tmp_path / "train.parquet"
    train.write_bytes(b"aaaa")
    before = data_dir_source_identity(tmp_path)
    train.write_bytes(b"bbbb")
    after = data_dir_source_identity(tmp_path)
    assert before["train.parquet"]["bytes"] == after["train.parquet"]["bytes"]
    assert before["train.parquet"]["sha256"] != after["train.parquet"]["sha256"]


def test_resume_fingerprint_covers_trainer_state_compatibility():
    fingerprint = json.loads(RUN_FINGERPRINT)
    assert {"train_bs", "grad_acc", "max_steps", "lr"} <= fingerprint.keys()
