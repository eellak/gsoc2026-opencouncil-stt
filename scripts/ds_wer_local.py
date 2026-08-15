#!/usr/bin/env python3
"""DS-WER over a LOCAL decode of the 39 frozen windows.

`eval/controlled_eval/score_ds_wer.py` scores the public benchmark providers from
the cached report; this is the same metric (scripts/ds_wer.py + the frozen term
lists in research/ds_wer/terms/) pointed at a decode-ablation-style JSON —
{"windows": {window_id: {"text": ...}}} — which is what the frozen local decode
path (notebooks/decode_ablation.py / correction_only_score.py) produces for a
new adapter. References come from the cached benchmark report, same as every
other consumer of the freeze.

    SC=~/.cache/oc-public .venv-eval/bin/python scripts/ds_wer_local.py \
        ~/.cache/oc-public/decode-ablation/eval-A.json
    # validation mode: score a benchmark provider's hyps instead of a file
    ... scripts/ds_wer_local.py --provider oc-runpod-fixed-2026-08-10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ds_wer import TermList, aggregate, ds_wer  # noqa: E402

MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("decode", nargs="?", help="decode JSON with windows->text")
    ap.add_argument("--provider", help="benchmark provider instanceId to score "
                                       "from the cached report (validation mode)")
    a = ap.parse_args()
    if bool(a.decode) == bool(a.provider):
        ap.error("give exactly one of: a decode JSON path, or --provider")

    man = json.loads(MANIFEST.read_text())
    from eval.controlled_eval import bench_data as B
    rep = B.load_report(man["source_run"])
    items = {it["itemId"]: it for it in rep["items"]}
    if a.provider:
        hyps = {w: (items[w]["perProvider"].get(a.provider) or {}).get("hypothesisText")
                for w in items}
        label = f"provider {a.provider}"
    else:
        state = json.loads(Path(a.decode).read_text())
        hyps = {w: v.get("text") for w, v in state["windows"].items()}
        label = a.decode

    terms = {p.stem: TermList.load(p) for p in sorted(TERMS_DIR.glob("*.json"))}
    per_window, missing = [], []
    for r in man["eval_windows"]:
        wid = r["window_id"]
        hyp = (hyps.get(wid) or "").strip()
        if not hyp:
            missing.append(wid)
            continue
        per_window.append(ds_wer(items[wid]["referenceText"], hyp, terms[r["city"]]))
    if missing:
        raise SystemExit(f"{len(missing)}/39 windows missing a hypothesis "
                         f"({missing[:5]}...) — refusing to score a subset")
    agg = aggregate(per_window)
    print(json.dumps({"input": label, "windows": len(per_window),
                      **{k: agg[k] for k in ("S", "D", "I", "N", "errors")},
                      "ds_wer": round(agg["ds_wer"], 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
