import json
from pathlib import Path

from eval.controlled_eval.build_training_listening_page import (
    build_items,
    dense_window_ids,
    render,
)


def test_dense_window_ids_aggregates_across_seeds():
    report = {"seeds": [
        {"all_windows": [{"window_id": "a", "insertion_delta": 2},
                         {"window_id": "b", "insertion_delta": 4}]},
        {"all_windows": [{"window_id": "a", "insertion_delta": 5},
                         {"window_id": "b", "insertion_delta": 1}]},
    ]}
    assert dense_window_ids(report) == ["a", "b"]


def test_page_keeps_hidden_metadata_outside_served_items(tmp_path: Path):
    residual, dense, out = tmp_path / "residual", tmp_path / "dense", tmp_path / "out"
    residual.mkdir()
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    (residual / "review.jsonl").write_text(json.dumps({
        "review_id": "r1", "audio": str(audio), "training_label": "λέξη"}) + "\n")
    (residual / "hidden-key.json").write_text(json.dumps([{
        "review_id": "r1", "stratum": "secret", "fixed_sdi": [1, 2, 3]}]))
    validation = dense / "corrected-input/eval/validation"
    validation.mkdir(parents=True)
    window_audio = validation / "window.wav"
    window_audio.write_bytes(b"audio")
    (validation / "manifest.json").write_text(json.dumps({"eval_windows": [{
        "window_id": "secret-window", "audio_local": "window.wav",
        "ref_tokens": ["δύο", "λέξεις"], "duration_sec": 4.0}]}))
    (dense / "dense-insertion-private.json").write_text(json.dumps({"seeds": [{
        "all_windows": [{"window_id": "secret-window", "insertion_delta": 9}]}]}))

    items, hidden = build_items(residual, dense, out)
    page = render(items)
    assert len(items) == 2
    assert "secret-window" not in page
    assert "secret" not in page
    assert "fixed_sdi" not in page
    assert "arm" not in page.lower()
    assert all((out / row["audio"]).is_file() for row in items)
    assert any(row.get("window_id") == "secret-window" for row in hidden)
