"""NER-gated fuzzy correction: does gating the deterministic corrector on GLiNER
entity spans recover the named-entity win WITHOUT corrupting clean text?

Compares ungated vs gated fuzzy on (a) the held-out test set (precision,
named_entity recall, collateral) and (b) a clean gold_final sample
(gold_final_retention). Pure local — GLiNER on CPU, no LLM quota.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from eval.fuzzy_correct import fuzzy_correct
from eval.scoring import greek_normalize, score_pair

GREEKBERT_MODEL = "amichailidis/bert-base-greek-uncased-v1-finetuned-ner"
GLINER_MODEL = "urchade/gliner_multi-v2.1"

ROOT = Path(__file__).resolve().parent.parent
LOOP = ROOT / "data" / "improve_loop"
EVAL = ROOT / "data" / "eval"
GLOSS = ROOT / "data" / "glossary" / "glossary.json"
LABELS = ["person", "location", "organization"]
THRESH = 0.45
CONFIGS = {
    "permissive": dict(name_like_only=True, phrases=True),
    "strict": dict(name_like_only=True, phrases=True, max_dist_cap=1, min_len=6, margin=2),
}


def _terms(gloss, city):
    return list(gloss.get("global", [])) + gloss.get("per_city", {}).get(str(city), [])


def _make_gate(name: str):
    """Return a fn texts->list[list[(start,end)]] of entity char-spans."""
    if name == "gliner":
        from gliner import GLiNER
        model = GLiNER.from_pretrained(GLINER_MODEL)

        def spans(texts):
            out = model.batch_predict_entities(texts, LABELS, threshold=THRESH)
            return [[(e["start"], e["end"]) for e in ents] for ents in out]
        return spans
    if name == "greekbert":
        from transformers import pipeline
        ner = pipeline("token-classification", model=GREEKBERT_MODEL,
                       aggregation_strategy="simple")

        def spans(texts):
            out = ner(list(texts), batch_size=16)
            if texts and isinstance(out[0], dict):  # single-input edge case
                out = [out]
            return [[(e["start"], e["end"]) for e in ents] for ents in out]
        return spans
    raise ValueError(f"unknown gate {name}")


def _wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def _arm(rows, gloss, span_list, cfg):
    ne_n = ne_fix = inter = correct = 0
    nonent_worse = nonent_n = 0
    for r, sp in zip(rows, span_list):
        inp, gold, cat = r["input_raw"], r["gold_final"], r["category"]
        out = fuzzy_correct(inp, _terms(gloss, r["city_id"]), allowed_spans=sp, **cfg)
        changed = greek_normalize(out) != greek_normalize(inp)
        s_in, s_out = score_pair(inp, inp, gold), score_pair(inp, out, gold)
        if cat == "named_entity":
            ne_n += 1
            ne_fix += int(s_out["normalized_exact"])
        else:
            nonent_n += 1
            if changed and s_out["wer"] > s_in["wer"]:
                nonent_worse += 1
        if changed:
            inter += 1
            if s_out["wer"] < s_in["wer"]:
                correct += 1
    return {
        "interventions": inter,
        "precision": round(correct / inter, 4) if inter else None,
        "named_entity_recall": round(ne_fix / ne_n, 4) if ne_n else None,
        "collateral_worse_per100_nonentity": round(100 * nonent_worse / nonent_n, 2) if nonent_n else None,
    }


def main(gate="gliner"):
    gloss = json.loads(GLOSS.read_text())
    spans_fn = _make_gate(gate)
    test = [json.loads(l) for l in (LOOP / "test.jsonl").read_text().splitlines() if l.strip()]

    # NER spans computed ONCE, reused across fuzzy configs
    test_spans = spans_fn([r["input_raw"] for r in test])
    none_test = [None] * len(test)

    print(f"test rows: {len(test)}  (gate={gate}, thr={THRESH})\n")
    res = {"test_n": len(test), "gate": gate, "threshold": THRESH, "configs": {}}
    for cname, cfg in CONFIGS.items():
        ung = _arm(test, gloss, none_test, cfg)
        gat = _arm(test, gloss, test_spans, cfg)
        res["configs"][cname] = {"ungated": ung, "gated": gat}
        print(f"[{cname}]")
        for tag, a in (("UNGATED", ung), ("GATED  ", gat)):
            print(f"  {tag}  precision {a['precision']}  ne_recall {a['named_entity_recall']}  "
                  f"interventions {a['interventions']}  collateral/100 {a['collateral_worse_per100_nonentity']}")

    # clean control: gold_final_retention, ungated vs gated, both configs
    cdf = pd.read_parquet(EVAL / "chains.parquet")
    split = json.loads((EVAL / "split.json").read_text())
    ev = cdf[cdf.meeting_id.isin(split["eval_meeting_ids"])]
    seen, clean = set(), []
    for _, r in ev.iterrows():
        g = str(r["gold_final"]).strip()
        k = (r["city_id"], greek_normalize(g))
        if g and k not in seen:
            seen.add(k)
            clean.append({"city_id": r["city_id"], "gold_final": g})
        if len(clean) >= 600:
            break
    clean_spans = spans_fn([c["gold_final"] for c in clean])
    none_clean = [None] * len(clean)
    print(f"\nclean gold_final sample: {len(clean)}")
    res["clean_n"] = len(clean)
    for cname, cfg in CONFIGS.items():
        res["configs"][cname]["clean"] = {}
        print(f"[{cname}]")
        for tag, spans in (("UNGATED", none_clean), ("GATED  ", clean_spans)):
            kept = sum(int(greek_normalize(
                fuzzy_correct(c["gold_final"], _terms(gloss, c["city_id"]), allowed_spans=s, **cfg)
            ) == greek_normalize(c["gold_final"])) for c, s in zip(clean, spans))
            ret = kept / len(clean)
            lo, hi = _wilson(kept, len(clean))
            res["configs"][cname]["clean"][tag.strip().lower()] = {"retention": round(ret, 4), "ci95": [lo, hi]}
            print(f"  {tag}  gold_final_retention {ret:.4f}  (95% CI {lo}-{hi})")

    out_path = LOOP / f"ner_gate_eval_{gate}.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["gliner", "greekbert"], default="gliner")
    main(ap.parse_args().gate)
