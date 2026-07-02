"""Next-batch verification — audio-faithfulness of a SAMPLE of the final list.

Answers "how reliable is our auto-selection?" without transcribing all 7,364.
Draws a stratified sample of the selected edits, re-transcribes each clip with the
Soniox realtime path (no API key — same path as the 320-clip calibration), scores
cer_soniox against the human label, and reports the faithful fraction + bucket mix.

Reuses extract_clip/transcribe from step2 and cer/assign_bucket from step3-4.

Outputs under data/next-batch/verify/:
  sample.parquet            the sampled rows
  clips/<id>.wav, asr/<id>.json   cached extraction + transcript (resumable)
  verify_manifest.csv       one row per item with cer_soniox + bucket
  verify_summary.md         faithful %, bucket mix, cer distribution
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from eval.next_batch_step2_transcribe import extract_clip, transcribe
from eval.next_batch_step3_4 import assign_bucket
from eval.scoring import cer, greek_normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "next-batch"
VERIFY = OUT / "verify"
CLIPS = VERIFY / "clips"
ASR = VERIFY / "asr"
SEED = 13
_INPUT = OUT / "selected_edits.jsonl"
_ONLY_NEW: set[str] = set()


def _short_special(gold: str) -> bool:
    n = greek_normalize(gold)
    return len(n) < 20 or len(n.split()) < 5


def draw_sample(n: int) -> pd.DataFrame:
    rows = [json.loads(l) for l in _INPUT.read_text().splitlines() if l.strip()]
    if _ONLY_NEW:
        rows = [r for r in rows if r["utterance_id"] in _ONLY_NEW]
    df = pd.DataFrame(rows)
    df["dur_bucket"] = pd.cut(df["duration"], [0, 1.5, 3, 5, 10, 30],
                              labels=["<1.5", "1.5-3", "3-5", "5-10", "10-30"]).astype(str)
    # proportional stratified over (category, dur_bucket), seeded
    frac = min(1.0, n / len(df))
    parts = []
    for _, g in df.groupby(["category", "dur_bucket"], observed=True):
        k = max(1, round(len(g) * frac))
        parts.append(g.sample(n=min(k, len(g)), random_state=SEED))
    s = pd.concat(parts)
    if len(s) > n:
        s = s.sample(n=n, random_state=SEED)
    return s.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default="verify", help="subdir under data/next-batch/")
    ap.add_argument("--only-new-vs", default="", help="path to json id list; sample only ids NOT in it")
    args = ap.parse_args()

    global VERIFY, CLIPS, ASR, _ONLY_NEW
    VERIFY = OUT / args.outdir
    CLIPS = VERIFY / "clips"
    ASR = VERIFY / "asr"
    if args.only_new_vs:
        seen = set(json.loads(Path(args.only_new_vs).read_text()))
        allids = {json.loads(l)["utterance_id"] for l in _INPUT.read_text().splitlines() if l.strip()}
        _ONLY_NEW = allids - seen

    CLIPS.mkdir(parents=True, exist_ok=True)
    ASR.mkdir(parents=True, exist_ok=True)

    sp = VERIFY / "sample.parquet"
    if sp.exists():
        sample = pd.read_parquet(sp)
    else:
        sample = draw_sample(args.n)
        sample.to_parquet(sp, index=False)
    print(f"[verify] sample = {len(sample)} of 7,364 selected", flush=True)

    todo = []
    for _, r in sample.iterrows():
        jp = ASR / f"{r['utterance_id']}.json"
        if jp.exists():
            try:
                if json.loads(jp.read_text()).get("asr_ok"):
                    continue
            except Exception:
                pass
        todo.append(r)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[verify] {len(todo)} to transcribe", flush=True)

    t0 = time.time()
    for i, r in enumerate(todo, 1):
        uid = r["utterance_id"]
        wav = CLIPS / f"{uid}.wav"
        rec = {"utterance_id": uid, "city_id": r["city_id"], "meeting_id": r["meeting_id"],
               "category": r["category"], "duration": float(r["duration"]),
               "input_raw": r["input_raw"], "gold_final": r["gold_final"]}
        ext = extract_clip(r["audio_url"], float(r["utterance_start"]), float(r["utterance_end"]), wav)
        rec.update({"extract_ok": ext["extract_ok"], "error": ext["error"]})
        if ext["extract_ok"]:
            rec.update(transcribe(wav, ext["expected_dur"]))
        else:
            rec.update({"asr_ok": False, "soniox_text": ""})
        (ASR / f"{uid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        el = time.time() - t0
        st = "ok" if rec.get("asr_ok") else "FAIL"
        print(f"[{i}/{len(todo)}] {uid} {st}  {el/60:.1f}m  ETA {(len(todo)-i)/(i/el)/60:.0f}m", flush=True)

    build_report()


def build_report() -> None:
    rows = []
    for jp in sorted(ASR.glob("*.json")):
        try:
            rows.append(json.loads(jp.read_text()))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    ok = df[df["asr_ok"] == True].copy()  # noqa: E712
    ok["cer_before"] = [cer(i, g) for i, g in zip(ok["input_raw"], ok["gold_final"])]
    ok["cer_soniox"] = [cer(s, g) for s, g in zip(ok["soniox_text"].fillna(""), ok["gold_final"])]
    ok["short"] = [_short_special(g) for g in ok["gold_final"]]
    ok["bucket"] = [assign_bucket(cb, cs, sh) for cb, cs, sh in
                    zip(ok["cer_before"], ok["cer_soniox"], ok["short"])]

    faithful = (ok["cer_soniox"] <= 0.15).mean() * 100
    suspect = (ok["cer_soniox"] >= 0.18).mean() * 100
    df.to_csv(VERIFY / "verify_manifest.csv", index=False)
    ok.to_parquet(VERIFY / "verify_scored.parquet", index=False)

    L = ["# Verify sample — audio-faithfulness of the selected edits\n"]
    L.append(f"- transcribed: {len(ok)} / {len(df)} (failed {len(df)-len(ok)})")
    L.append(f"- **faithful (cer_soniox ≤ 0.15): {faithful:.1f}%**")
    L.append(f"- **suspect (cer_soniox ≥ 0.18): {suspect:.1f}%**")
    L.append(f"- in-between: {100-faithful-suspect:.1f}%\n")
    L.append("## Bucket mix\n")
    for k, v in ok["bucket"].value_counts().items():
        L.append(f"- {k}: {v} ({v/len(ok)*100:.1f}%)")
    L.append("\n## cer_soniox distribution\n```\n" + ok["cer_soniox"].describe().to_string() + "\n```")
    L.append("\n## faithful % by category\n")
    for cat, g in ok.groupby("category"):
        L.append(f"- {cat}: {(g['cer_soniox']<=0.15).mean()*100:.0f}% faithful (n={len(g)})")
    (VERIFY / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
