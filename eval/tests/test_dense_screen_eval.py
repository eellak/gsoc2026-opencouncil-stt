import json

from eval.controlled_eval.dense_screen_eval import load_rows, totals


def test_load_rows_resolves_relative_audio(tmp_path):
    audio = tmp_path / "audio/w.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"wav")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": [{"key": "k", "clip": "audio/w.wav"}]}))
    rows = load_rows(manifest, "rows", "clip")
    assert rows[0]["clip"] == str(audio)


def test_totals_uses_reference_token_denominator():
    result = totals({"a": (1, 2, 3, 10), "b": (0, 1, 0, 5)})
    assert result["wer"] == 7 / 15
    assert result["del_rate"] == 3 / 15
