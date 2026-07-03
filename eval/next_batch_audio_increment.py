"""Next-batch INCREMENT — add ~N more audio-verified edits to the nb2audio queue.

Pulls fresh candidates from the LLM-kept pool that are NOT yet audio-verified
(and not val cities), city-capped for balance, audio-verifies them (Soniox
realtime, reusing every prior transcript), keeps the faithful, and MERGES them
into the existing nb2audio set (dedup + reshuffle).

  python -m eval.next_batch_audio_increment --cap 200
then copy final_audio/nb2audio_ids.json -> ui/.../nb2audio-ids.json and deploy.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import pandas as pd

from eval.next_batch_audio_final import local_cer, load_cached, FAITHFUL_MAX, VAL_CITIES
from eval.next_batch_step2_transcribe import extract_clip, transcribe

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "next-batch"
FINAL = OUT / "final_audio"
INC = OUT / "final_audio_inc"
CLIPS = INC / "clips"
ASR = INC / "asr"
SEED = 17  # different from the first pass so we draw NEW rows


def build_increment(cap: int) -> pd.DataFrame:
    ranked = pd.read_parquet(OUT / "ranked.parquet")
    judged = {json.loads(l)["utterance_id"]: json.loads(l) for l in (OUT / "judged.jsonl").read_text().splitlines() if l.strip()}
    keep_ids = {u for u, r in judged.items() if r["verdict"] == "keep"}
    # already transcribed (verified or attempted) in the first pass — exclude
    verified = set(pd.read_parquet(FINAL / "scored.parquet")["utterance_id"])
    cand = ranked[ranked["city_id"].isin(  # non-val, LLM-kept, not yet verified
        set(ranked["city_id"].unique()) - VAL_CITIES)].copy()
    cand = cand[cand["utterance_id"].isin(keep_ids) & ~cand["utterance_id"].isin(verified)]
    parts = [g.sample(n=min(cap, len(g)), random_state=SEED) for _, g in cand.groupby("city_id")]
    return pd.concat(parts).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=200, help="max new candidates per city")
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    CLIPS.mkdir(parents=True, exist_ok=True)
    ASR.mkdir(parents=True, exist_ok=True)
    bp = INC / "candidates.parquet"
    inc = pd.read_parquet(bp) if bp.exists() else build_increment(args.cap)
    inc.to_parquet(bp, index=False)
    print(f"[inc] increment candidates: {len(inc)} across {inc.city_id.nunique()} cities", flush=True)
    print(inc["city_id"].value_counts().to_string(), flush=True)

    cache = load_cached()

    def _done(uid):
        if uid in cache:
            return True
        p = ASR / f"{uid}.json"
        if p.exists():
            try:
                return bool(json.loads(p.read_text()).get("asr_ok"))
            except Exception:
                return False
        return False

    if not args.finalize_only:
        todo = [r for _, r in inc.iterrows() if not _done(r["utterance_id"])]
        print(f"[inc] {len(todo)} to transcribe", flush=True)
        t0, n_fail = time.time(), 0
        for i, r in enumerate(todo, 1):
            uid = r["utterance_id"]
            wav = CLIPS / f"{uid}.wav"
            rec = {"utterance_id": uid, "gold_final": r["gold_final"]}
            tr = None
            for attempt in range(3):
                ext = extract_clip(r["audio_url"], float(r["utterance_start"]), float(r["utterance_end"]), wav)
                tr = transcribe(wav, ext["expected_dur"]) if ext["extract_ok"] else {"asr_ok": False, "soniox_text": "", "error": ext["error"]}
                if tr.get("asr_ok"):
                    break
                time.sleep(5)
            rec.update(tr)
            if not tr.get("asr_ok"):
                n_fail += 1
            (ASR / f"{uid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {el/60:.0f}m ETA {(len(todo)-i)/(i/el)/60:.0f}m fails={n_fail}", flush=True)

    finalize(inc)


def finalize(inc: pd.DataFrame) -> None:
    cache = load_cached()
    inc = inc.copy()
    inc["soniox_text"] = inc["utterance_id"].map(cache)
    inc = inc[inc["soniox_text"].notna()].copy()
    inc["cer_local"] = [local_cer(s, g) for s, g in zip(inc["soniox_text"], inc["gold_final"])]
    new_faithful = inc[inc["cer_local"] <= FAITHFUL_MAX]["utterance_id"].tolist()

    existing = json.loads((FINAL / "faithful_ids.json").read_text())
    combined = list(dict.fromkeys(existing + new_faithful))  # dedup, preserve
    random.Random(SEED).shuffle(combined)
    (FINAL / "nb2audio_ids.json").write_text(json.dumps(combined), encoding="utf-8")

    print(f"\n[inc] new faithful: {len(new_faithful)} (of {len(inc)} scored)", flush=True)
    print(f"[inc] combined nb2audio: {len(existing)} + {len(new_faithful)} new = {len(combined)}", flush=True)
    print(f"[inc] by-city of new faithful:", flush=True)
    nf = inc[inc["cer_local"] <= FAITHFUL_MAX]
    print(nf["city_id"].value_counts().to_string(), flush=True)
    print(f"[inc] wrote {FINAL/'nb2audio_ids.json'} — copy to ui and deploy.", flush=True)


if __name__ == "__main__":
    main()
