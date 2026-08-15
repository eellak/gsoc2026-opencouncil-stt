"""Adapt the frozen train-screen manifests to what train_runpod.py consumes.

`exp-2026-08-14-external-packs`, the two screening runs. The frozen inputs live
under `~/.cache/oc-public/train-screens-2026-08/{run1,run2}/` (manifest JSONL +
presentation list + val slice, hashes in each meta.json). The trainer consumes
none of those directly, so this script does the three mechanical conversions —
in the mold of `mix_arms.py`: an arm is a list over one shared clip build, and
every invariant is asserted rather than hoped for.

  to-parquet   run manifest + val slice -> DATA_DIR/{train,validation}.parquet
               (the columns build_from_parquet reads; spans are the export's
               adjusted spans already, so no start_adj/end_adj columns exist)
  emit         WORK/manifest.json (superset clip build, after BUILD_AND_EXIT)
               + presentations JSONL -> TRAIN_MANIFEST json {"train":[...]},
               the per-epoch presentation multiset mapped to clip paths
  emit-pack    stage-1 pack manifest + presentations -> PACK_MANIFEST jsonl
               (multiset; PACK_ARM=pn; `weight` intentionally absent — the
               recorded decision is edge-flagged rows at full weight)

Exposure attestation: `emit` fails if dropped-clip attrition shifts any bucket's
presentation share by more than 2 percentage points (the prereg tolerance).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

TRAIN_BS, GRAD_ACC = 2, 4    # train_runpod.py constants (per-device batch, accum)
SHARE_TOL = 0.02             # prereg exposure tolerance, absolute


def log(*a):
    print(*a, flush=True)


def optimizer_updates(n_examples: int) -> int:
    """Optimizer updates for ONE epoch, matching the trainer's real dataloader:
    HF Trainer defaults to drop_last=False, so micro-steps = ceil(N/TRAIN_BS),
    and a partial accumulation window at epoch end still flushes as an update:
    updates = ceil(micro_steps / GRAD_ACC). Not the same as ceil(N/8) in general."""
    return math.ceil(math.ceil(n_examples / TRAIN_BS) / GRAD_ACC)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def to_parquet(a) -> None:
    import pandas as pd
    run = Path(a.run_dir)
    man = run / ("stage2-manifest.jsonl" if a.stage2 else "manifest.jsonl")
    val = run / ("stage2-val-slice.jsonl" if a.stage2 else "val-slice.jsonl")
    cols = ["utterance_id", "city_id", "meeting_id", "audio_url",
            "start", "end", "text", "source"]

    def frame(rows):
        return pd.DataFrame([{**{c: r.get(c) for c in cols},
                              "utterance_id": r["id"]} for r in rows])[cols]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va = frame(read_jsonl(man)), frame(read_jsonl(val))
    assert tr.audio_url.notna().all() and va.audio_url.notna().all()
    assert not set(tr.utterance_id) & set(va.utterance_id)
    tr.to_parquet(out / "train.parquet", index=False)
    va.to_parquet(out / "validation.parquet", index=False)
    log(f"-> {out}/train.parquet ({len(tr)}) + validation.parquet ({len(va)}; "
        f"valc={int((va.source == 'correction').sum())} "
        f"valr={int((va.source == 'no_edit').sum())})")


def emit(a) -> None:
    man = json.loads((Path(a.work) / "manifest.json").read_text())
    by_id = {}
    for c in man["train"]:
        uid = Path(c["audio"]).stem
        assert uid not in by_id, f"duplicate clip stem {uid}"
        by_id[uid] = c
    pres = read_jsonl(Path(a.presentations))
    want = collections.Counter(p["bucket"] for p in pres)
    arm, dropped = [], collections.Counter()
    for p in pres:
        c = by_id.get(p["id"])
        if c is None:
            dropped[p["bucket"]] += 1       # clip lost to ok_span/min-dur at build
            continue
        arm.append({"audio": c["audio"], "text": c["text"]})
    got = collections.Counter()
    for p in pres:
        if p["id"] in by_id:
            got[p["bucket"]] += 1
    log(f"presentations {len(pres)} -> {len(arm)} (dropped {dict(dropped)})")
    for b in want:
        drift = abs(got[b] / len(arm) - want[b] / len(pres))
        if drift > SHARE_TOL:
            raise SystemExit(f"bucket {b} share drifted {drift:.3f} > {SHARE_TOL} "
                             f"after clip attrition — rebuild, do not train")
    Path(a.out).write_text(json.dumps({"train": arm}, ensure_ascii=False))
    steps1 = optimizer_updates(len(arm))
    att = {
        "presentations_in": len(pres), "emitted": len(arm),
        "dropped_by_bucket": dict(dropped),
        "realized_presentations_by_bucket": dict(got),
        "realized_shares": {b: round(got[b] / len(arm), 4) for b in got},
        "target_shares": {b: round(want[b] / len(pres), 4) for b in want},
        "share_tolerance": SHARE_TOL,
        "max_steps": {"1_epoch": steps1, "2_epochs": 2 * steps1,
                      "formula": "ceil(ceil(N/2)/4) per epoch (drop_last=False, "
                                 "partial accumulation flushes)"},
    }
    Path(str(a.out) + ".attestation.json").write_text(
        json.dumps(att, ensure_ascii=False, indent=2))
    log(f"realized shares: {att['realized_shares']}")
    log(f"-> {a.out} (+ .attestation.json); "
        f"MAX_STEPS: 1 epoch = {steps1}, 2 epochs = {2 * steps1}")


def emit_pack(a) -> None:
    rows = {r["id"]: r for r in read_jsonl(Path(a.manifest))}
    pres = read_jsonl(Path(a.presentations))
    out = Path(a.out)
    n = 0
    per_source = collections.Counter()
    with out.open("w") as f:
        for p in pres:
            r = rows[p["id"]]
            audio = r["audio"]
            if a.relocate:
                old, new = a.relocate
                assert audio.startswith(old), audio
                audio = new + audio[len(old):]
            f.write(json.dumps({
                "pack_id": r["id"], "audio": audio,
                "text_p": "", "text_pn": r["text_pn"] or r["text"],
                "dur_sec": float(r.get("dur") or 0.0), "n_utt": 1,
            }, ensure_ascii=False) + "\n")
            per_source[p["bucket"]] += 1
            n += 1
    steps1 = optimizer_updates(n)
    att = {"presentations": n,
           "realized_presentations_by_source": dict(per_source),
           "max_steps": {"one_balanced_pass": steps1,
                         "formula": "ceil(ceil(N/2)/4) (drop_last=False, "
                                    "partial accumulation flushes)"}}
    Path(str(out) + ".attestation.json").write_text(
        json.dumps(att, ensure_ascii=False, indent=2))
    log(f"per-source presentations: {dict(per_source)}")
    log(f"-> {out} (+ .attestation.json) ({n} presentations); PACK_ARM=pn; "
        f"MAX_STEPS (one balanced pass) = {steps1}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("to-parquet")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage2", action="store_true",
                   help="use the stage2-* file names (run2 directory)")
    p = sub.add_parser("emit")
    p.add_argument("--work", required=True, help="WORK_DIR of the superset build")
    p.add_argument("--presentations", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("emit-pack")
    p.add_argument("--manifest", required=True, help="stage1-manifest.jsonl")
    p.add_argument("--presentations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--relocate", nargs=2, metavar=("OLD", "NEW"),
                   help="rewrite audio path prefix for the pod")
    a = ap.parse_args()
    {"to-parquet": to_parquet, "emit": emit, "emit-pack": emit_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
