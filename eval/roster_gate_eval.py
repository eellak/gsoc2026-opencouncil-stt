"""Test the core hypothesis: NER-gate + SMALL per-meeting candidate set (the
roster), instead of the dense 5894-name global glossary, makes deterministic
name correction safe.

We don't yet ingest OpenCouncil's per-meeting `partiesWithPeople`, so we use a
LEAKAGE-SAFE proxy: for an utterance u in meeting m, the candidate set is the
name-like tokens harvested from the gold_final of every OTHER utterance in m
(correct spellings, u itself excluded). This approximates "the names spoken in
this meeting" (~tens), exactly the small set the production roster would give.

Compares against the global-glossary gated numbers (eval/ner_gate_eval.py).
GreekBERT gate (the better one). Pure local — no LLM quota.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from eval.fuzzy_correct import fuzzy_correct
from eval.ner_gate_eval import _make_gate
from eval.scoring import greek_normalize, score_pair

ROOT = Path(__file__).resolve().parent.parent
LOOP = ROOT / "data" / "improve_loop"
EVAL = ROOT / "data" / "eval"
_TOK = re.compile(r"\w+", flags=re.UNICODE)
CFG = dict(name_like_only=False, phrases=True)   # candidates are already names; keep it permissive


def name_phrases(text: str) -> set:
    """Capitalized, length>=4 word runs -> name phrases + singles (proxy roster)."""
    toks = [m.group(0) for m in _TOK.finditer(str(text))]
    out, cur = set(), []
    for w in toks:
        if w[:1].isupper() and len(w) >= 4:
            cur.append(w)
        else:
            if cur:
                out.add(" ".join(cur)); cur = []
    if cur:
        out.add(" ".join(cur))
    out |= {w for w in toks if w[:1].isupper() and len(w) >= 4}
    return out


def _wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def main():
    cdf = pd.read_parquet(EVAL / "chains.parquet")
    split = json.loads((EVAL / "split.json").read_text())
    ev = cdf[cdf.meeting_id.isin(split["eval_meeting_ids"])]

    # per-meeting: name -> set(uids that mention it). A real roster name recurs;
    # require >=2 distinct utterances so sentence-initial common words drop out.
    meeting_name_uids: dict = defaultdict(lambda: defaultdict(set))
    for _, r in ev.iterrows():
        for nm in name_phrases(r["gold_final"]):
            meeting_name_uids[r["meeting_id"]][nm].add(r["utterance_id"])

    def roster_for(meeting_id, exclude_uid, min_uttr=2) -> list:
        return [nm for nm, uids in meeting_name_uids.get(meeting_id, {}).items()
                if len(uids - {exclude_uid}) >= min_uttr]

    sizes = [len(roster_for(m, None)) for m in list(meeting_name_uids)[:200]]
    print(f"meetings: {len(meeting_name_uids)}  | median proxy-roster size ~"
          f"{sorted(sizes)[len(sizes)//2] if sizes else 0}\n")

    spans_fn = _make_gate("greekbert")
    test = [json.loads(l) for l in (LOOP / "test.jsonl").read_text().splitlines() if l.strip()]
    test_spans = spans_fn([r["input_raw"] for r in test])

    # ---- precision / recall / collateral on the held-out test ----
    for gated, slist in (("ungated", [None] * len(test)), ("gated", test_spans)):
        ne_n = ne_fix = inter = correct = nonent_n = nonent_worse = 0
        for r, sp in zip(test, slist):
            inp, gold, cat = r["input_raw"], r["gold_final"], r["category"]
            terms = roster_for(r["meeting_id"], r["utterance_id"])
            out = fuzzy_correct(inp, terms, allowed_spans=sp, **CFG)
            changed = greek_normalize(out) != greek_normalize(inp)
            s_in, s_out = score_pair(inp, inp, gold), score_pair(inp, out, gold)
            if cat == "named_entity":
                ne_n += 1; ne_fix += int(s_out["normalized_exact"])
            else:
                nonent_n += 1
                if changed and s_out["wer"] > s_in["wer"]:
                    nonent_worse += 1
            if changed:
                inter += 1
                correct += int(s_out["wer"] < s_in["wer"])
        prec = round(correct / inter, 4) if inter else None
        print(f"  {gated.upper():7} precision {prec}  ne_recall {round(ne_fix/ne_n,4) if ne_n else None}  "
              f"interventions {inter}  collateral/100 {round(100*nonent_worse/nonent_n,2) if nonent_n else None}")

    # ---- clean control: gold_final_retention with per-meeting roster ----
    seen, clean = set(), []
    for _, r in ev.iterrows():
        g = str(r["gold_final"]).strip()
        k = (r["city_id"], greek_normalize(g))
        if g and k not in seen:
            seen.add(k)
            clean.append({"meeting_id": r["meeting_id"], "utterance_id": r["utterance_id"], "gold_final": g})
        if len(clean) >= 600:
            break
    clean_spans = spans_fn([c["gold_final"] for c in clean])
    print(f"\nclean gold_final sample: {len(clean)}")
    for gated, slist in (("ungated", [None] * len(clean)), ("gated", clean_spans)):
        kept = 0
        for c, sp in zip(clean, slist):
            terms = roster_for(c["meeting_id"], c["utterance_id"])
            out = fuzzy_correct(c["gold_final"], terms, allowed_spans=sp, **CFG)
            kept += int(greek_normalize(out) == greek_normalize(c["gold_final"]))
        ret = kept / len(clean)
        lo, hi = _wilson(kept, len(clean))
        print(f"  {gated.upper():7} gold_final_retention {ret:.4f}  (95% CI {lo}-{hi})")


if __name__ == "__main__":
    main()
