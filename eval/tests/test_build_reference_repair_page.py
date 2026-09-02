import json
from pathlib import Path

import pytest

from eval.controlled_eval.build_reference_repair_page import (
    WINDOW_IDS,
    build_package,
    chunk_bounds,
    chunk_texts,
    render,
)


ALIGNED = {
    WINDOW_IDS[0]: [[1.0, 1.4], [21.0, 21.4], [41.0, 41.4]],
    WINDOW_IDS[1]: [[0.5, 0.9], [2.0, 2.4], [30.0, 30.4]],
}


def make_inputs(tmp_path: Path, completeness=("material_omission", "material_omission")):
    """A fake dense-recovery tree plus the prior audit key/answers for both windows."""
    validation = tmp_path / "dense/corrected-input/eval/validation"
    (validation / "audio").mkdir(parents=True)
    windows = []
    for i, window_id in enumerate(WINDOW_IDS):
        rel = f"audio/{window_id}.wav"
        (validation / rel).write_bytes(f"audio-{i}".encode())
        windows.append({
            "window_id": window_id,
            "city": "argos",
            "meeting_id": "m",
            "start_sec": 1.0,
            "duration_sec": 45.0,
            "ref_tokens": ["πρωτη", "λεξη", str(i)],
            "ref_tokens_raw": 3,
            "audio_local": rel,
        })
    (validation / "manifest.json").write_text(json.dumps({"eval_windows": windows}))

    audit = tmp_path / "audit"
    (audit / "served").mkdir(parents=True)
    key = [{"review_id": "hid0", "kind": "training_clip", "stratum": "secret"}]
    answers = {}
    for i, window_id in enumerate(WINDOW_IDS):
        rid = f"prior{i}"
        key.append({"review_id": rid, "kind": "validation_window", "window_id": window_id})
        answers[rid] = {"reference_completeness": completeness[i], "notes": "λειπουν ατακες"}
    (audit / "hidden-key.json").write_text(json.dumps(key, ensure_ascii=False))
    (audit / "served/answers.json").write_text(json.dumps(answers, ensure_ascii=False))
    return tmp_path / "dense", audit, tmp_path / "out"


def test_chunk_bounds_covers_the_whole_window():
    bounds = chunk_bounds(45.0, 20.0)
    assert bounds[0] == (0.0, 20.0)
    assert bounds[-1][1] == pytest.approx(45.0)
    assert len(bounds) == 3


def test_package_is_blind_to_window_identity_and_carries_both_windows(tmp_path: Path):
    dense, audit, out = make_inputs(tmp_path)
    items, sources = build_package(dense, audit, out, ALIGNED)
    page = render(items, sources["source_fingerprint"])

    assert len(items) == 2
    assert {row["review_id"] for row in items} == {row["review_id"] for row in sources["windows"]}
    for window_id in WINDOW_IDS:
        assert window_id not in page
    assert "insertion" not in page.lower()
    assert "material_omission" not in page
    assert all((out / "served" / row["audio"]).is_file() for row in items)
    assert (out / "sources.json").is_file()
    assert not (out / "served/sources.json").exists()


def test_chunk_texts_place_every_token_in_exactly_one_interval():
    tokens = ["a", "b", "c", "d"]
    times = [[0.0, 0.3], [19.9, 20.4], [20.1, 20.5], [44.0, 44.2]]
    texts = chunk_texts(tokens, times, chunk_bounds(45.0))
    assert texts == ["a b", "c", "d"]
    assert " ".join(t for t in texts if t).split() == tokens


def test_intervals_reproduce_the_published_reference(tmp_path: Path):
    dense, audit, out = make_inputs(tmp_path)
    items, _ = build_package(dense, audit, out, ALIGNED)
    for item in items:
        rejoined = " ".join(c["text"] for c in item["chunks"] if c["text"])
        assert rejoined == item["published_text"]


def test_prefill_is_exactly_the_published_reference(tmp_path: Path):
    dense, audit, out = make_inputs(tmp_path)
    items, sources = build_package(dense, audit, out, ALIGNED)
    by_id = {row["window_id"]: row for row in sources["windows"]}
    for item in items:
        published = by_id[[w["window_id"] for w in sources["windows"]
                           if w["review_id"] == item["review_id"]][0]]
        assert item["published_text"] == " ".join(published["published_tokens"])


def test_rejects_a_window_the_prior_audit_did_not_call_material(tmp_path: Path):
    dense, audit, out = make_inputs(tmp_path, ("material_omission", "complete"))
    with pytest.raises(RuntimeError, match="material_omission"):
        build_package(dense, audit, out, ALIGNED)


def test_rejects_a_manifest_missing_one_frozen_window(tmp_path: Path):
    dense, audit, out = make_inputs(tmp_path)
    manifest_path = dense / "corrected-input/eval/validation/manifest.json"
    rows = json.loads(manifest_path.read_text())["eval_windows"]
    manifest_path.write_text(json.dumps({"eval_windows": rows[:1]}))
    with pytest.raises(RuntimeError, match="frozen window"):
        build_package(dense, audit, out, ALIGNED)


def test_refuses_to_write_inside_the_git_repository(tmp_path: Path):
    dense, audit, _ = make_inputs(tmp_path)
    repo_out = Path(__file__).resolve().parents[2] / "eval/_repair_tmp"
    with pytest.raises(RuntimeError, match="repository"):
        build_package(dense, audit, repo_out, ALIGNED)
