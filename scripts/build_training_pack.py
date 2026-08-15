"""Turn a filtered+punctuated external source into a training pack.

A *pack* is the unit this project adds to a fine-tune from an outside corpus. It is
deliberately the same shape whatever the source is, so a second and third source drop
in without touching the trainer. The contract lives in
`docs/reference/external-source-packs.md`.

    <packs>/<pack-id>/
      audio/            mono 16 kHz MP3, one clip per row
      train.jsonl       one row per clip: {id, audio, text, text_pn, dur, ...}
      meta.json         counts, hours, gate parameters, hashes, licence, caveats
      README.md         how this pack was built and what not to trust about it

`audio` in train.jsonl is an **absolute** path, because the trainer resolves it
directly (`notebooks/train_runpod.py` checks every path exists before it starts).
Re-run with `--relocate` after moving the pack to rewrite them.

Packs live under ~/.cache/oc-public/training-sets/ — audio and transcript text never
go in git.

Run:
    .venv-eval/bin/python scripts/build_training_pack.py \
        --source hparl2 --pack-id hparl2-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.scoring import greek_normalize  # noqa: E402

PACKS = Path.home() / ".cache/oc-public/training-sets"

# Per-source wiring. Adding a source = adding an entry here plus its own filter
# script; nothing downstream changes.
SOURCES = {
    "hparl2": {
        "title": "HParl (Elormiden/Hellenic-greek-parliamentary-speech)",
        "punctuated": Path.home() / ".cache/oc-public/hparl2/punctuated.jsonl",
        "licence": "DISPUTED: HF card YAML says cc-by-4.0, its own README body says "
                   "CC BY-NC 4.0. CLARIN 1602 is the authority and has NOT been checked "
                   "at source. Do not ship commercially on this pack until resolved.",
        "domain": "Hellenic Parliament plenary — OUT OF DOMAIN for municipal councils.",
        "report": "docs/reports/2026-08-14-hparl-audio-text-probe.md",
        "labels": "The source ships an accented but **lowercase, unpunctuated** "
                  "transcript. The label in `text` is the source's word sequence with "
                  "punctuation and casing restored from the Soniox output, arbitrated "
                  "by `gpt-5.6-luna` under a hard word guard: the model's normalized "
                  "token sequence must equal the source transcript, or the row falls "
                  "back to a deterministic punctuation transfer.",
        "caveats": [
            "The align>=0.95 admission gate is effectively an exact-match test on a "
            "median 9-token row: it selects easy audio and yields a Soniox-agreement "
            "sample, not a random one.",
            "Only ~26% of segments are complete sentences; the rest are fragments. "
            "Targets are punctuated accordingly (open-ended), but training on "
            "fragments risks teaching truncation - the deletion rate is the metric to "
            "watch.",
            "Punctuation and casing come from Soniox + gpt-5.6-luna, not from a human. "
            "The word guard proves no word was changed; it does not prove the "
            "punctuation is right.",
            "No word-level timestamps exist for this source, so the timestamped "
            "training arm (PACK_ARM=p) is not available - use pn.",
        ],
    },
    "stoma": {
        "title": "STOMA multi-speaker Greek read-speech corpus (aangelakis/STOMA)",
        "punctuated": Path.home() / ".cache/oc-public/stoma/filtered.jsonl",
        "licence": "CC-BY-4.0 (HF card of the authors' own repo; no upstream conflict "
                   "observed).",
        "domain": "Studio read speech (Harvard sentences + B2/C1/C2 exam texts) — "
                  "OUT OF DOMAIN for municipal councils.",
        "report": "docs/reports/2026-08-15-external-sources-probe.md",
        # same sentence read by another speaker is a legitimate extra sample here
        "dedupe_fields": ["text", "speaker_id"],
        "labels": "The label in `text` is the corpus's own accented, punctuated, "
                  "cased sentence, verbatim — no repair stage. Soniox verified the "
                  "word sequence against the audio (alignment gate + clean-edge "
                  "flags); punctuation and casing are the corpus authors'.",
        "caveats": [
            "Read speech from 6 speakers in studio conditions: no spontaneity, no "
            "overlap, no room acoustics. Stage-1 adaptation input only.",
            "Two main speakers cover the full sentence set; four secondary speakers "
            "repeat a subset, so raw sampling skews to the two mains. Balance at the "
            "sampler if this matters.",
            "Targets are the corpus's own accented, punctuated complete sentences; "
            "Soniox alignment verifies words, not punctuation.",
            "No word-level timestamps: PACK_ARM=p unavailable, use pn.",
        ],
    },
    "cv": {
        "title": "Common Voice Scripted Speech 26.0 Greek (Mozilla Data Collective)",
        "punctuated": Path.home() / ".cache/oc-public/cv-el/filtered.jsonl",
        "licence": "CC0-1.0 (Mozilla Data Collective). Terms: no re-hosting of the "
                   "dataset, no attempts to re-identify speakers.",
        "domain": "Community-recorded read prompts — OUT OF DOMAIN for municipal "
                  "councils.",
        "report": "docs/reports/2026-08-15-external-sources-probe.md",
        # the same prompt read by another contributor is a legitimate extra sample
        "dedupe_fields": ["text", "client_id"],
        "labels": "The label in `text` is the corpus's own validated prompt "
                  "(accented, cased), verbatim — no repair stage. Soniox verified "
                  "the word sequence against the audio. Original MP3 bytes are kept "
                  "as-is (no second lossy transcode).",
        "caveats": [
            "Read prompts from 454 community speakers on consumer microphones: no "
            "spontaneity or meeting acoustics. Stage-1 adaptation input only.",
            "Only the community-validated rows minus the official dev/test splits "
            "are eligible (dev/test held out as potential third-party benchmarks).",
            "~43% of prompts carry terminal punctuation; the rest end bare. Labels "
            "are kept verbatim — nothing was invented — so the pack may "
            "under-teach final periods.",
            "At a median of 6 reference tokens, align>=0.95 is exactly exact-match: "
            "the kept set is a Soniox-agreement sample, not a random one.",
            "No word-level timestamps: PACK_ARM=p unavailable, use pn.",
        ],
    },
    "eurospeech": {
        "title": "EuroSpeech Greek Parliament subset (disco-eth/EuroSpeech, greece)",
        "punctuated": Path.home() / ".cache/oc-public/eurospeech-el/filtered.jsonl",
        "licence": "OPEN, on a different basis than HParl: the HF card says 'other' and "
                   "cites, for Greece, ν.2121/1993 art. 2(5) (official State texts are "
                   "outside copyright) and art. 25(1)(b). That is a statutory exception on "
                   "the underlying proceedings, NOT a restrictive licence on a compilation - "
                   "so there is no NC term to clear, unlike the CLARIN HParl packaging. "
                   "Two things still to verify: that the citation covers the AUDIO and not "
                   "only the transcripts, and that the disco-eth compilation itself imposes "
                   "nothing (its card explicitly disclaims responsibility for accuracy).",
        "domain": "Hellenic Parliament plenary — OUT OF DOMAIN for municipal councils, "
                  "same domain as the hparl2 pack (dedupe across the two).",
        "report": "docs/reports/2026-08-15-external-sources-probe.md",
        "labels": "The label in `text` is the official parliamentary minutes span "
                  "(accented, punctuated, cased), verbatim — no repair stage. Soniox "
                  "verified the word sequence against the audio (alignment gate + "
                  "clean-edge flags); fragments keep their honest mid-sentence "
                  "punctuation.",
        "caveats": [
            "The corpus's own per-row wer/cer come from ITS pipeline (Whisper Turbo "
            "alignment search): using them to pre-select rows biases toward "
            "Whisper-easy speech. Admission is always our Soniox gate.",
            "Segments are alignment windows over official minutes: many start/end "
            "mid-sentence. Boundary flags (first/last ref token missing) gate "
            "admission; punctuation is the official text's own.",
            "Official minutes are lightly edited (non-verbatim) relative to speech; "
            "the Soniox gate limits but does not eliminate this.",
            "Same parliament as hparl2 — check overlap before ever using both packs.",
            "No word-level timestamps: PACK_ARM=p unavailable, use pn.",
        ],
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    except Exception:
        return "unknown"


def edge_flags(r: dict) -> tuple[bool, bool]:
    """(first_missing, last_missing): is the row's first/last reference word absent
    from the ASR hypothesis? Sources whose filter records this keep their own values;
    for the rest it is derived, because a clipped edge is the same defect either way."""
    if "first_ref_missing" in r or "last_ref_missing" in r:
        return bool(r.get("first_ref_missing")), bool(r.get("last_ref_missing"))
    tr = greek_normalize(r.get("ref") or "").split()
    th = set(greek_normalize(r.get("soniox_text") or "").split())
    if not tr:
        return False, False
    return tr[0] not in th, tr[-1] not in th


def build(source: str, pack_id: str, min_align: float, min_dur: float,
          max_dur: float, min_tokens: int, copy_audio: bool,
          edge_policy: str, edge_weight: float) -> None:
    cfg = SOURCES[source]
    rows = [json.loads(l) for l in
            cfg["punctuated"].read_text(encoding="utf-8").splitlines()]

    out = PACKS / pack_id
    (out / "audio").mkdir(parents=True, exist_ok=True)

    dedupe_fields = cfg.get("dedupe_fields", ["text"])
    kept, dropped = [], {"align": 0, "edge": 0, "dur": 0, "tokens": 0,
                         "no_audio": 0, "dup": 0}
    seen_text: set = set()
    for r in rows:
        text = r.get("final") or r.get("text_llm") or r.get("text_transfer")
        if r.get("align", 0) < min_align:
            dropped["align"] += 1
            continue
        fm, lm = edge_flags(r)
        if (fm or lm) and edge_policy == "drop":
            dropped["edge"] += 1        # clipped start/end teaches truncation
            continue
        if not (min_dur <= r.get("dur", 0) <= max_dur):
            dropped["dur"] += 1
            continue
        if len((text or "").split()) < min_tokens:
            dropped["tokens"] += 1
            continue
        src_mp3 = Path(r["mp3"]) if r.get("mp3") else None
        if not src_mp3 or not src_mp3.exists():
            dropped["no_audio"] += 1
            continue
        key = tuple(text if fld == "text" else r.get(fld) for fld in dedupe_fields)
        if key in seen_text:           # identical target twice = duplicated clip
            dropped["dup"] += 1
            continue
        seen_text.add(key)

        dst = out / "audio" / f"{r['row_id']}.mp3"
        if copy_audio and not dst.exists():
            shutil.copy2(src_mp3, dst)
        kept.append({
            "id": r["row_id"],
            "audio": str((dst if copy_audio else src_mp3).resolve()),
            "text": text,
            "text_pn": text,           # no timestamps exist for this source
            "dur": r["dur"],
            "source": source,
            "align": r["align"],
            "complete_sentence": r.get("complete"),
            "edge_clipped": bool(fm or lm),
            "weight": edge_weight if (fm or lm) else 1.0,
            "text_dataset": r.get("ref"),
            "text_asr": r.get("soniox_text"),
        })

    train = out / "train.jsonl"
    with train.open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    hours = sum(k["dur"] for k in kept) / 3600
    meta = {
        "pack_id": pack_id,
        "source": source,
        "title": cfg["title"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_by_git_rev": git_rev(),
        "n_rows": len(kept),
        "hours": round(hours, 3),
        "dropped": dropped,
        "gate": {"min_align": min_align, "min_dur": min_dur, "max_dur": max_dur,
                 "min_tokens": min_tokens, "edge_policy": edge_policy,
                 "edge_weight": edge_weight},
        "edge_clipped_frac": (
            round(sum(1 for k in kept if k["edge_clipped"]) / len(kept), 3)
            if kept else None),
        "complete_sentence_frac": (
            round(sum(1 for k in kept if k["complete_sentence"]) / len(kept), 3)
            if kept else None),
        "train_jsonl_sha256": sha256(train),
        "labels": cfg["labels"],
        "licence": cfg["licence"],
        "domain": cfg["domain"],
        "report": cfg["report"],
        "caveats": cfg["caveats"],
        "trainer": {
            "consumed_by": "notebooks/train_runpod.py",
            "env": {"PACK_MANIFEST": str(train.resolve()), "PACK_ARM": "pn"},
            "note": "PACK_ARM=p (Whisper timestamp tokens) is unavailable: this source "
                    "has no word-level timings.",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
    (out / "README.md").write_text(readme(meta), encoding="utf-8")

    print(f"[pack] {pack_id}: {len(kept)} rows, {hours:.2f} h -> {out}")
    print(f"[pack] dropped: {dropped}")
    print(f"[pack] sha256(train.jsonl) = {meta['train_jsonl_sha256'][:16]}…")


def readme(m: dict) -> str:
    caveats = "\n".join(f"- {c}" for c in m["caveats"])
    return f"""# Training pack `{m['pack_id']}`

{m['title']}

Built {m['built_at']} from repo rev `{m['built_by_git_rev']}`.
**{m['n_rows']} clips / {m['hours']} hours.** Complete sentences:
{m['complete_sentence_frac']}.

This is a **supplementary** pack. It is meant to be combined with the project's own
human corrections, not to replace them, and the corpus is out of domain:
{m['domain']}

## How to train with it

```bash
PACK_MANIFEST={m['trainer']['env']['PACK_MANIFEST']} \\
PACK_ARM=pn \\
python notebooks/train_runpod.py
```

{m['trainer']['note']}

## Admission gate

Every row here passed: alignment ≥ {m['gate']['min_align']} against an independent
ASR (Soniox), duration in [{m['gate']['min_dur']}, {m['gate']['max_dur']}] s, at least
{m['gate']['min_tokens']} tokens, non-duplicate target, audio present. Rows rejected
at this stage: {m['dropped']}.

## How the targets were made

{m['labels']} Full method and numbers: [`{m['report']}`](../../../{m['report']}).

## What not to trust

{caveats}

## Licence

{m['licence']}

## Files

- `train.jsonl` — one row per clip. `audio` is absolute; `text` is the training
  target. `text_dataset` and `text_asr` keep both inputs so any label can be
  re-derived or audited. sha256 `{m['train_jsonl_sha256']}`.
- `meta.json` — machine-readable version of everything above.
- `audio/` — mono 16 kHz MP3, ~4 kB/s.
"""


def relocate(pack_id: str) -> None:
    """Rewrite absolute audio paths after the pack directory has moved."""
    out = PACKS / pack_id
    train = out / "train.jsonl"
    rows = [json.loads(l) for l in train.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["audio"] = str((out / "audio" / f"{r['id']}.mp3").resolve())
    with train.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = json.loads((out / "meta.json").read_text())
    meta["train_jsonl_sha256"] = sha256(train)
    meta["trainer"]["env"]["PACK_MANIFEST"] = str(train.resolve())
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
    print(f"[pack] relocated {len(rows)} rows under {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hparl2", choices=sorted(SOURCES))
    ap.add_argument("--pack-id", default=None)
    ap.add_argument("--min-align", type=float, default=0.95)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--min-tokens", type=int, default=3)
    ap.add_argument("--edge-policy", choices=("drop", "flag"), default="drop",
                    help="clipped-edge rows: drop them, or admit them carrying "
                         "edge_clipped/weight for the sampler to down-weight")
    ap.add_argument("--edge-weight", type=float, default=0.5,
                    help="weight written on edge-clipped rows when --edge-policy=flag")
    ap.add_argument("--no-copy-audio", action="store_true",
                    help="point at the cache clips instead of copying them into the pack")
    ap.add_argument("--relocate", action="store_true")
    args = ap.parse_args()

    pack_id = args.pack_id or f"{args.source}-v1"
    if args.relocate:
        relocate(pack_id)
        return
    build(args.source, pack_id, args.min_align, args.min_dur, args.max_dur,
          args.min_tokens, not args.no_copy_audio, args.edge_policy,
          args.edge_weight)


if __name__ == "__main__":
    main()
