"""Segment-level validation pass (runbook Step 5, second pass).

The corrections CSV has no speaker column and no cached meeting JSON, so — per
the runbook's documented fallback — we approximate a speaker segment by a window
of consecutive same-meeting utterances ordered by `utterance_start`. We re-run
baseline vs glossary on the context-sensitive categories with this context and
score only the TARGET line, to measure the per-utterance-vs-segment gap.

Usage:
  python -m eval.segment [--per-cat N] [--window K] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from eval.backends import generate
from eval.fix_call import parse_numbered
from eval.glossary import prepare_retrieval_pool, render_terms_block, select_glossary_terms
from eval.prompts import SYSTEM_PROMPT, build_user_prompt
from eval.scoring import score_pair

_BACKEND = "claude"
_MODEL: str | None = None

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
REPORTS = ROOT / "data" / "reports" / "fix-task-eval"
CSV = ROOT / "data-1779206108158.csv"
LOG = ROOT / "eval" / "run.log"

CONTEXT_CATS = {"named_entity", "acronym_abbreviation", "word_boundary", "number_date"}

_lock = threading.Lock()
_pools: dict[str, dict] = {}


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [seg] {msg}"
    print(line, flush=True)
    with _lock:
        LOG.open("a", encoding="utf-8").write(line + "\n")


def build_meeting_index() -> dict:
    """meeting_id -> ordered list of {utterance_id, input_raw} by utterance_start."""
    df = pd.read_csv(CSV, usecols=["utterance_id", "meeting_id", "utterance_start",
                                   "before_text", "edit_timestamp"], dtype=str)
    df["before_text"] = df["before_text"].fillna("")
    df["edit_timestamp"] = pd.to_datetime(df["edit_timestamp"], errors="coerce")
    df["ustart"] = pd.to_numeric(df["utterance_start"], errors="coerce")
    df = df.sort_values(["utterance_id", "edit_timestamp"], kind="stable")
    first = df.groupby("utterance_id", sort=False).first().reset_index()
    idx: dict[str, list] = {}
    for meeting, grp in first.sort_values("ustart", kind="stable").groupby("meeting_id"):
        idx[meeting] = grp[["utterance_id", "before_text"]].to_dict("records")
    # position lookup
    pos = {}
    for meeting, lst in idx.items():
        for i, r in enumerate(lst):
            pos[r["utterance_id"]] = (meeting, i)
    return {"by_meeting": idx, "pos": pos}


def _get_pool(gloss, city):
    with _lock:
        if city not in _pools:
            _pools[city] = prepare_retrieval_pool(gloss, city)
        return _pools[city]


def process_row(row, gloss, mindex, window):
    meeting, i = mindex["pos"][row["utterance_id"]]
    lst = mindex["by_meeting"][meeting]
    lo = max(0, i - window)
    hi = min(len(lst), i + window + 1)
    seg = lst[lo:hi]
    target_idx = i - lo  # 0-based position of target in the window
    lines = [s["before_text"] for s in seg]
    n = len(lines)

    base_up = build_user_prompt(row["city_id"], lines)
    terms = select_glossary_terms(_get_pool(gloss, row["city_id"]), row["input_raw"])
    block = render_terms_block(terms)

    def target_line(raw):
        parsed = parse_numbered(raw, n)
        if parsed is not None:
            return parsed[target_idx], True
        return "", False

    base_raw = generate(SYSTEM_PROMPT, base_up, backend=_BACKEND, model=_MODEL)
    bt, bok = target_line(base_raw)
    if terms:
        gloss_up = build_user_prompt(row["city_id"], lines, glossary_block=block)
        gloss_raw = generate(SYSTEM_PROMPT, gloss_up, backend=_BACKEND, model=_MODEL)
        gt, gok = target_line(gloss_raw)
    else:
        gt, gok = bt, bok

    bs = score_pair(row["input_raw"], bt, row["gold_final"])
    gs = score_pair(row["input_raw"], gt, row["gold_final"])
    return {
        "utterance_id": row["utterance_id"],
        "category": row["category"],
        "ebclass": row["ebclass"],
        "window": n,
        "input_raw": row["input_raw"],
        "gold_final": row["gold_final"],
        "n_glossary_terms": len(terms),
        "baseline": {"output": bt, "parse_ok": bok, **bs},
        "glossary": {"output": gt, "parse_ok": gok, **gs},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=60)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(REPORTS / "ab_results_segment.jsonl"))
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "gemini"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    global _BACKEND, _MODEL
    _BACKEND, _MODEL = args.backend, args.model

    gloss = json.loads((ROOT / "data" / "glossary" / "glossary.json").read_text())
    sample = [json.loads(l) for l in (EVAL / "sample.jsonl").read_text().splitlines() if l.strip()]
    sample = [r for r in sample if r["category"] in CONTEXT_CATS]

    # cap per category
    by_cat: dict[str, list] = {}
    for r in sample:
        by_cat.setdefault(r["category"], []).append(r)
    rows = []
    for cat, rs in by_cat.items():
        rows.extend(rs[: args.per_cat])

    _log(f"building meeting index ...")
    mindex = build_meeting_index()
    rows = [r for r in rows if r["utterance_id"] in mindex["pos"]]

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                try:
                    rec = json.loads(l)
                    if "error" not in rec:
                        done.add(rec["utterance_id"])
                except Exception:
                    pass
    todo = [r for r in rows if r["utterance_id"] not in done]
    _log(f"segment pass: {len(rows)} rows, {len(todo)} todo, window=±{args.window}")

    t0 = time.time()
    n_done = 0
    with out_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_row, r, gloss, mindex, args.window): r for r in todo}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"utterance_id": r["utterance_id"], "category": r["category"],
                           "error": str(e)[:300]}
                    _log(f"ERROR {r['utterance_id']}: {str(e)[:120]}")
                with _lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                n_done += 1
                if n_done % 25 == 0:
                    rate = n_done / (time.time() - t0)
                    _log(f"{n_done}/{len(todo)} done  {rate:.2f} rows/s  "
                         f"ETA {(len(todo)-n_done)/rate/60:.1f}m")
    _log(f"segment DONE {n_done} in {(time.time()-t0)/60:.1f}m -> {out_path}")


if __name__ == "__main__":
    main()
