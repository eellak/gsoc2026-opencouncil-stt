#!/usr/bin/env python3
"""DS-WER v2: the same metric as `scripts/ds_wer.py`, scored per preregistered cut.

The v2 term lists (`research/ds_wer/terms/{city}.v2.json`) carry a `klass` and a
`cut` on every term. A cut is scored by building a `TermList` from **only** that
cut's terms and aligning independently: adding or removing atomic multi-token terms
changes the alignment, so a class score obtained by filtering the operations of a
single full-list alignment would not be well defined. Class scores therefore need
not sum to the `all` cut.

    SC=~/.cache/oc-public .venv-eval/bin/python scripts/ds_wer_v2.py \
        --decode ~/.cache/oc-public/decode-ablation/eval-A.json
    ... scripts/ds_wer_v2.py --provider soniox
    ... scripts/ds_wer_v2.py --audit          # count-only reference audit, no system
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ds_wer import TermList, aggregate, ds_wer  # noqa: E402

MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"


def load_terms(version: str) -> tuple[dict[str, dict[str, TermList]], dict]:
    """{city: {cut: TermList}} plus the raw records."""
    raw, out = {}, {}
    for city in ("argos", "orestiada"):
        p = TERMS_DIR / (f"{city}.json" if version == "v1" else f"{city}.v2.json")
        raw[city] = json.loads(p.read_text())
    if version == "v1":
        return {c: {"v1": TermList(raw[c]["terms"])} for c in raw}, raw

    # Every city must expose the same cut keys, even when a class happens to be empty
    # there: the scorer iterates one city's keys and would otherwise KeyError.
    klasses = sorted({t["klass"] for rec in raw.values() for t in rec["terms"]})
    for city, rec in raw.items():
        out[city] = {c: TermList([t for t in rec["terms"] if t["klass"] in ks])
                     for c, ks in rec["cuts"].items()}
        for k in klasses:
            out[city][f"klass:{k}"] = TermList(
                [t for t in rec["terms"] if t["klass"] == k])
    return out, raw


def hypotheses(decode: str | None, provider: str | None):
    man = json.loads(MANIFEST.read_text())
    from eval.controlled_eval import bench_data as B
    rep = B.load_report(man["source_run"])
    items = {it["itemId"]: it for it in rep["items"]}
    if provider:
        hyps = {w: (items[w]["perProvider"].get(provider) or {}).get("hypothesisText")
                for w in items}
        label = f"provider {provider}"
    elif decode:
        state = json.loads(Path(decode).read_text())
        hyps = {w: v.get("text") for w, v in state["windows"].items()}
        label = decode
    else:
        hyps, label = {}, "reference-only"
    return man, items, hyps, label


def score(version: str, decode: str | None, provider: str | None) -> dict:
    terms, _ = load_terms(version)
    man, items, hyps, label = hypotheses(decode, provider)
    cuts = sorted(next(iter(terms.values())).keys())

    rows: dict[str, list[dict]] = {c: [] for c in cuts}
    missing = []
    for r in man["eval_windows"]:
        wid = r["window_id"]
        hyp = (hyps.get(wid) or "").strip()
        if not hyp:
            missing.append(wid)
            continue
        for c in cuts:
            rows[c].append(ds_wer(items[wid]["referenceText"], hyp,
                                  terms[r["city"]][c]))
    if missing:
        raise SystemExit(f"{len(missing)}/39 windows missing a hypothesis "
                         f"({missing[:5]}...) - refusing to score a subset")
    out = {"input": label, "version": version, "windows": len(rows[cuts[0]])}
    for c in cuts:
        a = aggregate(rows[c])
        out[c] = {k: a[k] for k in ("S", "D", "I", "N", "errors")}
        out[c]["ds_wer"] = round(a["ds_wer"], 4) if a["ds_wer"] is not None else None
    return out


# --------------------------------------------------------------- count-only audit

def audit(version: str) -> dict:
    """Reference-side concentration diagnostics. Reads NO system output."""
    terms, _ = load_terms(version)
    man, items, _, _ = hypotheses(None, None)
    cuts = sorted(next(iter(terms.values())).keys())

    out: dict = {"version": version, "cuts": {}}
    for c in cuts:
        per_window, per_meeting, per_term = {}, collections.Counter(), collections.Counter()
        for r in man["eval_windows"]:
            wid, city = r["window_id"], r["city"]
            res = ds_wer(items[wid]["referenceText"], "", terms[city][c])
            per_window[wid] = res["N"]
            per_meeting[r["meeting_id"]] += res["N"]
            for tid, v in res["per_term"].items():
                per_term[f"{city}/{tid}"] += v["N"]
        N = sum(per_window.values())
        n_terms = sum(len(terms[city][c]) for city in terms)
        ws = sorted(per_window.values(), reverse=True)
        ms = sorted(per_meeting.values(), reverse=True)
        ts = [v for v in per_term.values() if v]

        def share(x, tot):
            return round(x / tot, 4) if tot else None

        def hhi(vals, tot):
            return round(1 / sum((v / tot) ** 2 for v in vals), 2) if tot and vals else None

        out["cuts"][c] = {
            "terms_in_cut": n_terms,
            "terms_with_at_least_one_occurrence": len(ts),
            "zero_hit_terms": n_terms - len(ts),
            "N": N,
            "N_per_1000_ref_tokens": round(1000 * N / sum(w["ref_tokens"] for w in man["eval_windows"]), 2)
            if N else 0.0,
            "windows_with_N_gt_0": sum(1 for v in per_window.values() if v),
            "meetings_with_N_gt_0": sum(1 for v in per_meeting.values() if v),
            "meetings": len(per_meeting),
            "median_N_per_window": ws[len(ws) // 2] if ws else 0,
            "p90_N_per_window": ws[max(0, int(math.ceil(0.1 * len(ws))) - 1)] if ws else 0,
            "max_N_per_window": ws[0] if ws else 0,
            "largest_term_share": share(max(ts, default=0), N),
            "top5_term_share": share(sum(sorted(ts, reverse=True)[:5]), N),
            "largest_meeting_share": share(ms[0] if ms else 0, N),
            "top4_window_share": share(sum(ws[:4]), N),
            "effective_terms_hhi": hhi(ts, N),
            "effective_meetings_hhi": hhi([v for v in ms if v], N),
        }
        d = out["cuts"][c]
        d["warnings"] = [w for w, hit in (
            ("one term > 10% of N", (d["largest_term_share"] or 0) > 0.10),
            ("top-5 terms > 40% of N", (d["top5_term_share"] or 0) > 0.40),
            ("one meeting > 20% of N", (d["largest_meeting_share"] or 0) > 0.20),
            ("top-4 windows > 50% of N", (d["top4_window_share"] or 0) > 0.50),
            ("fewer than half the meetings carry an occurrence",
             d["meetings_with_N_gt_0"] * 2 < d["meetings"]),
        ) if hit]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="v2", choices=("v1", "v2"))
    ap.add_argument("--decode")
    ap.add_argument("--provider")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    res = audit(a.version) if a.audit else score(a.version, a.decode, a.provider)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
