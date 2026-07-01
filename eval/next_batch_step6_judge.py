"""Next-batch Step 6 — Sonnet text-plausibility triage of the ranked shortlist.

Runbook Step 6. The free score found *interesting* edits; this LLM pass reads each
edit and judges whether it is a GENUINE, clean transcription correction worth
training Whisper on — vs a semantic rewrite, a formatting/grammar-only change, or
a label that looks wrong. NO audio here (that is Soniox's job); this is a cheap
text guard that removes non-acoustic / implausible edits before any paid ASR.

Sequential by design: `claude -p` is process-wide serialized (OAuth throttle).
Processes best-ranked candidates first, checkpoints per batch, resumable.

Output: data/next-batch/judged.jsonl   (one row per judged utterance)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from eval.backends import generate

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "next-batch"
OUT = OUT_DIR / "judged.jsonl"

SYSTEM = (
    "You are a data curator for Greek automatic speech recognition (ASR) "
    "fine-tuning. You are given human corrections of raw Greek STT output: a "
    "'before' (what the ASR produced) and an 'after' (the human-corrected text). "
    "Your job: decide whether each pair is a GENUINE, clean transcription fix that "
    "would teach a speech model to hear Greek better.\n\n"
    "KEEP  = 'after' is a plausible correction of a MIS-HEARING in 'before' "
    "(a misheard word, name, ending, or word-boundary; something the audio "
    "actually said and the ASR got wrong).\n"
    "REJECT = 'after' is a semantic REWRITE or adds meaning not explainable by a "
    "mishearing; OR it is only punctuation / capitalization / number-formatting / "
    "acronym-expansion / pure grammar-style; OR 'after' itself looks wrong, "
    "truncated, or garbled.\n"
    "UNSURE = genuinely ambiguous.\n\n"
    "Also mark 'acoustic': true if the error is the kind a speech model must learn "
    "(phonetic mishear / name / ending / boundary), false if it is text-only "
    "(punctuation, number, acronym, pure grammar).\n\n"
    "You cannot hear the audio — judge plausibility from the text only.\n"
    "Return ONLY a JSON array, one object per item, in order:\n"
    '[{"i":0,"verdict":"keep|reject|unsure","acoustic":true|false,"why":"<=8 words"}]'
)


def build_prompt(batch: list[dict]) -> str:
    lines = ["Judge these edits. Return the JSON array only.\n"]
    for k, r in enumerate(batch):
        lines.append(f"[{k}] category={r['category']}")
        lines.append(f"    before: {r['input_raw']}")
        lines.append(f"    after : {r['gold_final']}")
    return "\n".join(lines)


def parse_verdicts(text: str, n: int) -> list[dict] | None:
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(arr, list):
        return None
    by_i = {}
    for o in arr:
        if isinstance(o, dict) and "i" in o:
            by_i[int(o["i"])] = o
    out = []
    for k in range(n):
        o = by_i.get(k, {})
        out.append({
            "verdict": str(o.get("verdict", "unsure")).lower(),
            "acoustic": bool(o.get("acoustic", False)),
            "why": str(o.get("why", ""))[:60],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="max utterances to judge (0=all shortlist)")
    ap.add_argument("--target-keep", type=int, default=0, help="stop once this many KEEP reached (0=off)")
    args = ap.parse_args()

    short = pd.read_parquet(OUT_DIR / "shortlist.parquet").sort_values("select_rank")
    if args.limit:
        short = short.head(args.limit)

    done = {}
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip():
                try:
                    r = json.loads(l); done[r["utterance_id"]] = r
                except Exception:
                    pass
    n_keep = sum(1 for r in done.values() if r.get("verdict") == "keep")

    rows = [r for _, r in short.iterrows() if r["utterance_id"] not in done]
    print(f"[step6] shortlist {len(short):,}, already judged {len(done):,}, "
          f"todo {len(rows):,}, keeps so far {n_keep}", flush=True)

    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as f:
        for bi in range(0, len(rows), args.batch):
            batch = [dict(r) for r in rows[bi:bi + args.batch]]
            try:
                raw = generate(SYSTEM, build_prompt(batch), backend="claude",
                               model="sonnet", timeout=180)
                verdicts = parse_verdicts(raw, len(batch))
            except Exception as e:
                verdicts = None
                print(f"  batch {bi} error: {str(e)[:120]}", flush=True)
            if verdicts is None:
                verdicts = [{"verdict": "unsure", "acoustic": False, "why": "parse_fail"}] * len(batch)
            for r, v in zip(batch, verdicts):
                rec = {"utterance_id": r["utterance_id"], "select_rank": int(r["select_rank"]),
                       "category": r["category"], "base_score": round(float(r["base_score"]), 4),
                       **v}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if v["verdict"] == "keep":
                    n_keep += 1
            f.flush()
            n_proc = bi + len(batch)
            rate = n_proc / (time.time() - t0)
            print(f"  {n_proc}/{len(rows)}  keeps={n_keep}  {rate*60:.0f}/min  "
                  f"ETA {(len(rows)-n_proc)/rate/60:.1f}m", flush=True)
            if args.target_keep and n_keep >= args.target_keep:
                print(f"[step6] reached target-keep {args.target_keep}, stopping.", flush=True)
                break

    print(f"[step6] done. judged so far -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
