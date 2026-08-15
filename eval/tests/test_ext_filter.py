"""Tests for eval/ext_filter.py — the generic external-source Soniox filter.

Everything here runs offline: synthetic parquet shards on disk, no HF, no ASR.
"""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.ext_filter import (
    SOURCES,
    canonical_rows,
    complete_sentence,
    cv_rows,
    edge_flags,
    fix_homoglyphs,
    sample_indices,
)


def _write_shard(path, rows, audio_col="audio", text_col="text", extra_cols=()):
    cols = {
        audio_col: [{"bytes": r["bytes"], "path": r.get("path")} for r in rows],
        text_col: [r["text"] for r in rows],
    }
    for c in extra_cols:
        cols[c] = [r[c] for r in rows]
    pq.write_table(pa.table(cols), path)


# ---------- adapter contract ----------

def test_canonical_rows_stoma_shape(tmp_path):
    shard = tmp_path / "train-00003-of-00015.parquet"
    _write_shard(
        shard,
        [{"bytes": b"AUD0", "path": "x.wav", "text": "Καλημέρα σας.",
          "speaker_id": "F1", "section": "B2", "session": 7}],
        extra_cols=("speaker_id", "section", "session"))
    pf = pq.ParquetFile(shard)
    rows = canonical_rows(pf, "data/train-00003-of-00015.parquet",
                          SOURCES["stoma"], base=0)
    assert len(rows) == 1
    r = rows[0]
    assert r["row_id"] == "train-00003-of-00015_000000"
    assert r["transcription"] == "Καλημέρα σας."
    assert r["audio_bytes"] == b"AUD0"
    assert r["extras"] == {"speaker_id": "F1", "section": "B2", "session": 7}


def test_canonical_rows_eurospeech_extras(tmp_path):
    shard = tmp_path / "train-00042-of-00527.parquet"
    _write_shard(
        shard,
        [{"bytes": b"A", "path": None, "text": "θα ήθελα να επισημάνω,",
          "wer": 0.24, "cer": 0.11, "video_id": "v1", "transcript_id": "t1",
          "duration_seconds": 13.0}],
        text_col="human_transcript",
        extra_cols=("wer", "cer", "video_id", "transcript_id", "duration_seconds"))
    pf = pq.ParquetFile(shard)
    rows = canonical_rows(pf, "greece/train-00042-of-00527.parquet",
                          SOURCES["eurospeech"], base=10)
    r = rows[0]
    assert r["row_id"] == "train-00042-of-00527_000010"
    assert r["extras"]["ds_wer"] == pytest.approx(0.24)
    assert r["extras"]["ds_cer"] == pytest.approx(0.11)
    assert r["extras"]["video_id"] == "v1"


# ---------- boundary completeness ----------

def test_edge_flags_clean():
    f = edge_flags("καλημέρα σας κύριε πρόεδρε", "καλημέρα σας κύριε πρόεδρε")
    assert f == {"first_ref_missing": False, "last_ref_missing": False}


def test_edge_flags_prefix_and_suffix():
    # first ref token absent from hyp -> clipped start
    f = edge_flags("καλημέρα σας κύριε πρόεδρε", "σας κύριε πρόεδρε")
    assert f["first_ref_missing"] and not f["last_ref_missing"]
    # last ref token substituted -> clipped/garbled end
    f = edge_flags("καλημέρα σας κύριε πρόεδρε", "καλημέρα σας κύριε πρόταση")
    assert f["last_ref_missing"] and not f["first_ref_missing"]


def test_edge_flags_empty_sides():
    assert edge_flags("", "να πω")["first_ref_missing"] is False
    f = edge_flags("να πω", "")
    assert f["first_ref_missing"] and f["last_ref_missing"]


# ---------- sentence completeness heuristic ----------

@pytest.mark.parametrize("text,want", [
    ("Καλημέρα σας.", True),
    ("Τι κάνετε;", True),
    ("θα ήθελα να επισημάνω,", False),      # lowercase start, comma end
    ("Η πρώτη αφορά τη σύσταση", False),    # no terminal mark
    ("«Ναι!»", True),
    ("", False),
])
def test_complete_sentence(text, want):
    assert complete_sentence(text) is want


# ---------- homoglyph repair ----------

def test_fix_homoglyphs_mixed_tokens():
    # Latin T/N/A lookalikes inside Greek words (observed in STOMA)
    assert fix_homoglyphs("Tο καλοκαίρι") == "Το καλοκαίρι"
    assert fix_homoglyphs("Nα συζητάτε") == "Να συζητάτε"
    assert fix_homoglyphs("Aρχικά") == "Αρχικά"


def test_fix_homoglyphs_leaves_real_latin_alone():
    assert fix_homoglyphs("τα fake news είναι πρόβλημα") == \
        "τα fake news είναι πρόβλημα"
    assert fix_homoglyphs("debate") == "debate"
    assert fix_homoglyphs("") == ""


# ---------- deterministic sampling ----------

def test_sample_indices_deterministic():
    a = sample_indices(1000, 50, seed=7)
    b = sample_indices(1000, 50, seed=7)
    assert a == b and len(a) == 50 and a == sorted(a)
    assert sample_indices(1000, 50, seed=8) != a
    assert sample_indices(30, 50, seed=7) == list(range(30))


# ---------- preselect ----------

def test_preselect_indices_gate_and_lane(tmp_path):
    from eval.ext_filter import preselect_indices
    shard = tmp_path / "t.parquet"
    wers = [0.02, 0.10, 0.14, 0.16, 0.25, 0.40, None, 0.30]
    _write_shard(shard,
                 [{"bytes": b"A", "path": None, "text": "x", "wer": w}
                  for w in wers],
                 text_col="human_transcript", extra_cols=("wer",))
    pf = pq.ParquetFile(shard)
    idx = preselect_indices(pf, SOURCES["eurospeech"], n=100, seed=1,
                            thr=0.15, explore_frac=0.5)
    # all <=0.15 rows in; None and 0.40 never; lane draws from (0.15, 0.35]
    assert set(idx) >= {0, 1, 2}
    assert 5 not in idx and 6 not in idx
    assert idx == sorted(idx)
    lane = set(idx) - {0, 1, 2}
    assert lane <= {3, 4, 7}


# ---------- Common Voice local adapter ----------

def _tsv(path, header, rows):
    path.write_text("\n".join(["\t".join(header)] +
                              ["\t".join(r) for r in rows]) + "\n")


def test_cv_rows_excludes_dev_test_and_invalid(tmp_path):
    root = tmp_path / "el"
    (root / "clips").mkdir(parents=True)
    hdr = ["client_id", "path", "sentence_id", "sentence", "up_votes", "down_votes"]
    _tsv(root / "validated.tsv", hdr, [
        ["c1", "a.mp3", "s1", "Καλημέρα σας", "2", "0"],
        ["c2", "b.mp3", "s2", "Τι κάνετε;", "3", "1"],
        ["c3", "c.mp3", "s3", "Στο dev split", "2", "0"],
        ["c4", "d.mp3", "s4", "Στο test split", "2", "0"],
    ])
    _tsv(root / "dev.tsv", hdr, [["c3", "c.mp3", "s3", "Στο dev split", "2", "0"]])
    _tsv(root / "test.tsv", hdr, [["c4", "d.mp3", "s4", "Στο test split", "2", "0"]])
    (root / "clip_durations.tsv").write_text(
        "clip\tduration[ms]\na.mp3\t3210\nb.mp3\t4500\nc.mp3\t100\nd.mp3\t100\n")
    for f in ("a", "b", "c", "d"):
        (root / "clips" / f"{f}.mp3").write_bytes(b"MP3")

    cfg = dict(SOURCES["cv"], root=root)
    rows = cv_rows(cfg)
    assert [r["row_id"] for r in rows] == ["a", "b"]
    r = rows[0]
    assert r["transcription"] == "Καλημέρα σας"
    assert r["dur"] == pytest.approx(3.21)
    assert r["mp3_src"] == root / "clips" / "a.mp3"
    assert r["extras"]["client_id"] == "c1"


# ---------- source registry sanity ----------

def test_sources_registry():
    for name, cfg in SOURCES.items():
        assert cfg["work"].name, name          # per-source cache dir
        if cfg.get("kind") == "local_cv":
            assert cfg["root"]
        else:
            assert cfg["repo"] and cfg["text_col"] and cfg["default_shards"], name
    assert json.dumps(sorted(SOURCES)) == '["cv", "eurospeech", "stoma"]'
