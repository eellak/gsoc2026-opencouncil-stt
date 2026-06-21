"""A/B run: baseline vs glossary-augmented prompt over the held-out sample.

Runbook Step 5. Per-utterance ("cheap") pass. Checkpoints append-per-row to a
JSONL so a dropped connection resumes instead of restarting. Row work uses a
thread pool, while `claude -p` calls are serialized to avoid OAuth retry storms.

Usage:
  python -m eval.run_ab [--workers N] [--limit N] [--out PATH] [--sample PATH]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.backends import generate
from eval.fix_call import parse_numbered
from eval.glossary import prepare_retrieval_pool, render_terms_block, select_glossary_terms
from eval.prompts import SYSTEM_PROMPT, build_user_prompt
from eval.scoring import score_pair

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
REPORTS = ROOT / "data" / "reports" / "fix-task-eval"
LOG = ROOT / "eval" / "run.log"

_write_lock = threading.Lock()
_pool_lock = threading.Lock()
_pools: dict[str, dict] = {}

# inference target (set in main); kept as module globals so the thread-pool
# workers see one consistent backend/model per run.
_BACKEND = "claude"
_MODEL: str | None = None


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with _write_lock:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _get_pool(gloss: dict, city_id: str) -> dict:
    with _pool_lock:
        if city_id not in _pools:
            _pools[city_id] = prepare_retrieval_pool(gloss, city_id)
        return _pools[city_id]


def _candidate(raw: str) -> tuple[str, bool]:
    """Extract the single corrected line; (text, parse_ok)."""
    parsed = parse_numbered(raw, 1)
    if parsed is not None:
        return parsed[0], True
    # fallback: first numbered line if any, else whole text
    for line in raw.splitlines():
        s = line.strip()
        if s and s[0].isdigit() and "." in s[:4]:
            return s.split(".", 1)[1].strip(), False
    return raw.strip(), False


def process_row(row: dict, gloss: dict) -> dict:
    city = row["city_id"]
    inp = row["input_raw"]
    gold = row["gold_final"]

    base_up = build_user_prompt(city, [inp])
    terms = select_glossary_terms(_get_pool(gloss, city), inp)
    block = render_terms_block(terms)

    base_raw = generate(SYSTEM_PROMPT, base_up, backend=_BACKEND, model=_MODEL)
    base_out, base_ok = _candidate(base_raw)

    if terms:
        gloss_up = build_user_prompt(city, [inp], glossary_block=block)
        gloss_raw = generate(SYSTEM_PROMPT, gloss_up, backend=_BACKEND, model=_MODEL)
        gloss_out, gloss_ok = _candidate(gloss_raw)
    else:
        # empty glossary block => arms are identical; skip the redundant call
        gloss_out, gloss_ok = base_out, base_ok

    base_score = score_pair(inp, base_out, gold)
    gloss_score = score_pair(inp, gloss_out, gold)

    return {
        "utterance_id": row["utterance_id"],
        "city_id": city,
        "meeting_id": row["meeting_id"],
        "category": row["category"],
        "ebclass": row["ebclass"],
        "input_raw": inp,
        "gold_final": gold,
        "n_glossary_terms": len(terms),
        "glossary_terms": terms,
        "backend": _BACKEND,
        "model": _MODEL or _BACKEND,
        "baseline": {"output": base_out, "parse_ok": base_ok, **base_score},
        "glossary": {"output": gloss_out, "parse_ok": gloss_ok, **gloss_score},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", default=str(EVAL / "sample.jsonl"))
    ap.add_argument("--out", default=str(REPORTS / "ab_results.jsonl"))
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "gemini"])
    ap.add_argument("--model", default=None, help="backend model override (e.g. haiku)")
    args = ap.parse_args()

    global _BACKEND, _MODEL
    _BACKEND, _MODEL = args.backend, args.model

    REPORTS.mkdir(parents=True, exist_ok=True)
    gloss = json.loads((ROOT / "data" / "glossary" / "glossary.json").read_text())

    rows = [json.loads(l) for l in Path(args.sample).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                try:
                    rec = json.loads(l)
                    if "error" not in rec:  # retry errored rows on resume
                        done.add(rec["utterance_id"])
                except Exception:
                    pass
    todo = [r for r in rows if r["utterance_id"] not in done]
    _log(f"start: {len(rows)} rows, {len(done)} already done, {len(todo)} todo, "
         f"workers={args.workers}, backend={_BACKEND}, model={_MODEL or 'default'}")

    t0 = time.time()
    n_done = 0
    with out_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_row, r, gloss): r for r in todo}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # never lose the run on one bad row
                    rec = {"utterance_id": r["utterance_id"], "category": r["category"],
                           "error": str(e)[:300]}
                    _log(f"ERROR {r['utterance_id']}: {str(e)[:120]}")
                with _write_lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                n_done += 1
                if n_done % 25 == 0:
                    rate = n_done / (time.time() - t0)
                    eta = (len(todo) - n_done) / rate if rate else 0
                    _log(f"{n_done}/{len(todo)} done  {rate:.2f} rows/s  ETA {eta/60:.1f}m")
    _log(f"DONE {n_done} rows in {(time.time()-t0)/60:.1f}m -> {out_path}")


if __name__ == "__main__":
    main()
