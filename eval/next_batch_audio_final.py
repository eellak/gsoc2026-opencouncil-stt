"""Next-batch FINAL — city-balanced, audio-verified, randomized selection.

Rebuilt per owner feedback (2026-07-02):
  - representative across TRAINING cities (cap per city; no Athens/Chania domination)
  - val cities orestiada + argos excluded (whole-city holdout; per-speaker holdout
    needs meeting-JSON speaker metadata the CSV lacks — tracked as a gap)
  - EVERY kept edit is audio-verified: a fresh ASR (Soniox realtime) must actually
    say the gold text (local-alignment faithfulness), not inferred from text/LLM
  - queue order randomized (seeded), not score-ranked

Reuses cached Soniox transcripts from earlier passes; transcribes the rest
(realtime, resumable). Then keeps only faithful and writes the UI id list.

Outputs under data/next-batch/final_audio/:
  balanced.parquet, asr/<id>.json (new transcripts), scored.parquet
  faithful_ids.json (-> copied to ui/src/lib/server/state/nb2-ids.json on finalize)
  final_summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from eval.next_batch_step2_transcribe import extract_clip, transcribe
from eval.scoring import greek_normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "next-batch"
FINAL = OUT / "final_audio"
CLIPS = FINAL / "clips"
ASR = FINAL / "asr"
VAL_CITIES = {"orestiada", "argos"}
CAP = 600
SEED = 13
FAITHFUL_MAX = 0.20


def local_cer(soniox: str, gold: str) -> float:
    g, s = greek_normalize(gold), greek_normalize(soniox)
    if not g:
        return 0.0
    if not s:
        return 1.0
    return 1 - fuzz.partial_ratio(g, s) / 100.0


def build_balanced() -> pd.DataFrame:
    sel = pd.DataFrame([json.loads(l) for l in (OUT / "selected_edits.jsonl").read_text().splitlines() if l.strip()])
    sel = sel[~sel["city_id"].isin(VAL_CITIES)].copy()
    parts = [g.sample(n=min(CAP, len(g)), random_state=SEED) for _, g in sel.groupby("city_id")]
    return pd.concat(parts).reset_index(drop=True)


def load_cached() -> dict[str, str]:
    """id -> soniox_text from every prior asr pass (calib/verify/verify_tier2/final_audio)."""
    cache: dict[str, str] = {}
    for f in glob.glob(str(OUT / "*" / "asr" / "*.json")):
        try:
            r = json.loads(Path(f).read_text())
            if r.get("asr_ok"):
                cache[r["utterance_id"]] = r.get("soniox_text", "")
        except Exception:
            pass
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    CLIPS.mkdir(parents=True, exist_ok=True)
    ASR.mkdir(parents=True, exist_ok=True)

    bp = FINAL / "balanced.parquet"
    bal = pd.read_parquet(bp) if bp.exists() else build_balanced()
    bal.to_parquet(bp, index=False)
    print(f"[final] balanced set: {len(bal)} across {bal.city_id.nunique()} cities", flush=True)

    cache = load_cached()

    def _done(uid: str) -> bool:
        # done only if we have a GOOD transcript (cached ok, or a prior asr_ok json).
        # failed clips (asr_ok False / empty) are retried — a temp-key hiccup must not
        # permanently drop an edit.
        if uid in cache:
            return True
        p = ASR / f"{uid}.json"
        if p.exists():
            try:
                return bool(json.loads(p.read_text()).get("asr_ok"))
            except Exception:
                return False
        return False

    def _try_once(r) -> dict:
        wav = CLIPS / f"{r['utterance_id']}.wav"
        ext = extract_clip(r["audio_url"], float(r["utterance_start"]), float(r["utterance_end"]), wav)
        if not ext["extract_ok"]:
            return {"asr_ok": False, "soniox_text": "", "error": ext["error"]}
        return transcribe(wav, ext["expected_dur"])

    if not args.finalize_only:
        todo = [r for _, r in bal.iterrows() if not _done(r["utterance_id"])]
        print(f"[final] {len(cache)} cached; {len(todo)} to transcribe", flush=True)
        t0, n_fail = time.time(), 0
        for i, r in enumerate(todo, 1):
            uid = r["utterance_id"]
            rec = {"utterance_id": uid, "gold_final": r["gold_final"]}
            tr = _try_once(r)
            # up to 2 retries on failure — a failed mint/renew usually recovers on
            # the next attempt (get_key re-mints within 90s of expiry).
            for _ in range(2):
                if tr.get("asr_ok"):
                    break
                time.sleep(5)
                tr = _try_once(r)
            rec.update(tr)
            if not tr.get("asr_ok"):
                n_fail += 1
            (ASR / f"{uid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {el/60:.0f}m ETA {(len(todo)-i)/(i/el)/60:.0f}m "
                      f"fails={n_fail}", flush=True)
        if n_fail:
            print(f"[final] {n_fail} still failing after retries — re-run to recover "
                  f"(likely stale Chrome cf_clearance; refresh perplexity.ai).", flush=True)

    finalize(bal)


def finalize(bal: pd.DataFrame) -> None:
    cache = load_cached()  # now includes final_audio/asr
    bal = bal.copy()
    bal["soniox_text"] = bal["utterance_id"].map(cache)
    scored = bal[bal["soniox_text"].notna()].copy()
    n_missing = len(bal) - len(scored)
    scored["cer_local"] = [local_cer(s, g) for s, g in zip(scored["soniox_text"], scored["gold_final"])]
    scored["faithful"] = scored["cer_local"] <= FAITHFUL_MAX
    scored.to_parquet(FINAL / "scored.parquet", index=False)

    keep = scored[scored["faithful"]].copy()
    # randomized queue order (seeded) — representative, not score-ranked
    ids = keep["utterance_id"].tolist()
    random.Random(SEED).shuffle(ids)
    (FINAL / "faithful_ids.json").write_text(json.dumps(ids), encoding="utf-8")

    L = ["# Final audio-verified, city-balanced, randomized selection\n",
         f"- balanced candidates: {len(bal)} (cap {CAP}/city, excl val {sorted(VAL_CITIES)})",
         f"- transcribed/available: {len(scored)} (missing {n_missing})",
         f"- **AUDIO-VERIFIED FAITHFUL kept (cer_local ≤ {FAITHFUL_MAX}): {len(keep)}** ({len(keep)/max(1,len(scored))*100:.1f}%)",
         f"- queue order: seeded-random\n", "## Kept by city\n"]
    for c, n in keep["city_id"].value_counts().items():
        L.append(f"- {c}: {n}")
    L.append("\n## Kept by category\n")
    for c, n in keep["category"].value_counts().items():
        L.append(f"- {c}: {n}")
    (OUT / "final_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    print(f"\n[final] wrote faithful_ids.json ({len(keep)} ids). Finalize UI with --finalize-only then copy.", flush=True)


if __name__ == "__main__":
    main()
