#!/usr/bin/env python3
"""Does Scribe v2's per-word logprob rank the words a human listener marked wrong?

`exp-2026-09-09-scribe-confidence-fidelity`. The third and last of the three
per-word confidence probes: Soniox scored 0.8167, our own adapter 0.8151, and
Scribe -- the system the fusion falls back to, and the one R2 hands every
three-way disagreement -- has never been measured at all.

SUBSTRATE: the frozen gold set. FIDELITY-TO-AUDIO -- a human listened and typed
what was said. This is the metric that decides. It is NOT the 391-window
agreement-with-OpenCouncil substrate and the two are never merged.

EVERY STATISTIC IS IMPORTED from eval/soniox_confidence_probe rather than
rewritten, because the only claim worth making here is a comparison, and a
reimplemented AUROC is a different measurement wearing the same name.

WHAT AN ERROR IS, declared before the number (Astra a5b91215): one deterministic
Levenshtein alignment of the emitted Scribe words against the human sequence,
operation priority ("S","D","I"), normalization by `wtoks`. An emitted word is an
error iff its operation is S or I. Deletions have no emitted word to attach a
confidence to and are structurally invisible; their share of edit operations is
reported as the ceiling, not buried.

A MEETING WITH NO ERRORS has an undefined AUROC. Such meetings are excluded from
the equal-weight mean and `n_meetings_informative` records how many contributed.

WHAT THIS COMPARISON IS NOT: identical methodology does not make the three
AUROCs strictly like-for-like. The three systems emit different words, so each
faces a different error population -- a system that abstains on hard audio is
graded on an easier set.

    .venv-eval/bin/python eval/scribe_confidence_fidelity.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gold_set_score import (                                    # noqa: E402
    GOLD, REGIONS, gold_view, in_intervals, load, masked_intervals, region_span,
)
from eval.soniox_confidence_probe import (                           # noqa: E402
    EQUAL_WIDTH_EDGES, GO_THRESHOLD, N_PERM, _align, bucket_table, meeting_cluster_ci,
    per_meeting_auroc, permutation_null, quantile_edges, subset_metrics, word_units,
)

SCRDIR = GOLD / "hyp" / "scribe"
OUT = ROOT / "eval" / "results_scribe_confidence_fidelity.json"
PRIMARY_REGION = "core_envelope"

# Two arms of the same vendor, and they are NOT interchangeable. `hyp/scribe` is
# the 2026-09-09 decode, which is VERBATIM: the decode script retyped its form
# fields and omitted no_verbatim, so the published 0.7974 is a statement about the
# verbatim arm. `hyp/scribe-nv` is the clean read decoded on 2026-09-16 -- the arm
# that wins on the published reference, and whose logprobs have never been scored.
# Separate output files, because merging them would silently average two systems.
ARMS = {
    "verbatim": (GOLD / "hyp" / "scribe",
                 ROOT / "eval" / "results_scribe_confidence_fidelity.json"),
    "clean": (GOLD / "hyp" / "scribe-nv",
              ROOT / "eval" / "results_scribe_confidence_fidelity_clean.json"),
}


def scribe_words(cid: str) -> tuple[list[dict], int, int]:
    """Raw response -> words with a confidence. Mirror of logprobToConfidence in
    src/lib/ScribeTranscribe.ts, which is what production actually shows."""
    body = json.loads((SCRDIR / f"{cid}.json").read_text(encoding="utf-8"))
    words, no_lp = [], 0
    for w in body.get("words", []):
        if w.get("type") != "word":
            continue
        lp = w.get("logprob")
        if lp is None:
            no_lp += 1
            continue
        words.append({"raw": w.get("text", ""), "start": w.get("start"),
                      "end": w.get("end"),
                      "conf": max(0.0, min(1.0, math.exp(lp))), "logprob": lp})
    return words, no_lp, len(body.get("words", []))


def build_units(scored, ans, sel, region: str):
    units, ceiling = [], Counter()
    drop_punct = split_words = missing_lp = 0
    for cid in scored:
        if not (SCRDIR / f"{cid}.json").exists():
            continue
        seq, _by, _ov, blocks, _kept = gold_view(ans[cid], region)
        span = region_span(blocks, region)
        if span is None or not seq:
            continue
        outside = [b for b in ans[cid]["b"]["blocks"]
                   if b["id"] not in {x["id"] for x in blocks}]
        ivs = masked_intervals(blocks, span, outside)
        ov_blocks = [b for b in blocks if b.get("ov_with") and not b.get("text_unc")]

        words, no_lp, _tot = scribe_words(cid)
        missing_lp += no_lp
        words = [w for w in words
                 if w["start"] is not None and in_intervals(w["start"], ivs)]
        us, dp, sw = word_units(words)
        drop_punct += dp
        split_words += sw

        ops = _align(seq, [u["tok"] for u in us], ("S", "D", "I"), False)
        lab = {}
        for op, _i, j in ops:
            if op in ("M", "S", "I"):
                lab[j] = op
            ceiling[op] += 1
        for j, u in enumerate(us):
            op = lab.get(j)
            if op is None:
                continue
            u = dict(u)
            u["op"] = op
            u["err"] = int(op != "M")
            u["cell"] = cid
            u["meeting"] = sel[cid]["meeting_id"]
            u["overlap"] = any(b["s"] <= u["start"] < b["e"] for b in ov_blocks)
            units.append(u)
    c = dict(ceiling)
    tot = sum(c.get(k, 0) for k in ("S", "D", "I"))
    c["deletion_share_of_edits"] = (c.get("D", 0) / tot) if tot else None
    c["edits_visible_to_confidence"] = ((c.get("S", 0) + c.get("I", 0)) / tot) if tot else None
    c["punctuation_only_words_dropped"] = drop_punct
    c["words_splitting_into_multiple_tokens"] = split_words
    c["emitted_words_without_logprob"] = missing_lp
    return units, c


def main() -> None:
    global SCRDIR, OUT
    arm = sys.argv[1] if len(sys.argv) > 1 else "verbatim"
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {sorted(ARMS)}")
    SCRDIR, OUT = ARMS[arm]

    man, ans, _cells, sel = load()
    scored = man["scored_cell_ids"]
    have = [c for c in scored if (SCRDIR / f"{c}.json").exists()]
    res = {
        "protocol": "scribe-confidence-fidelity-2026-09-09",
        "experiment": "exp-2026-09-09-scribe-confidence-fidelity",
        "arm": arm,
        "model": (f"ElevenLabs scribe_v2, {arm} arm "
                  f"(no_verbatim={'true' if arm == 'clean' else 'ABSENT from the request'})"),
        "answers_sha256": man["answers_sha256"],
        "substrate": "gold set, FIDELITY-TO-AUDIO; never merged with the 391-window "
                     "agreement-with-OpenCouncil substrate",
        "frozen_before_looking": {
            "primary_statistic": "mean of within-meeting AUROC of (1 - exp(logprob)) "
                                 "for predicting that an EMITTED word is an error",
            "go_threshold": GO_THRESHOLD,
            "primary_region": PRIMARY_REGION,
            "auroc_null": 0.5,
            "error_definition": "one Levenshtein alignment, priority (S,D,I), wtoks "
                                "normalization; err = op in {S, I}",
            "meeting_with_no_errors": "AUROC undefined -> excluded from the mean; "
                                      "n_meetings_informative reports the count",
            "words_without_logprob": "excluded and counted, never scored as confident",
            "comparability": "identical methodology does NOT make the three systems' "
                             "AUROCs strictly like-for-like: they emit different words "
                             "and therefore face different error populations",
        },
        "published_comparators": {"soniox_stt_rt_v4": 0.8167, "our_adapter": 0.8151},
        "n_cells_scored": len(have),
        "n_cells_expected": len(scored),
        "n_meetings": len({sel[c]["meeting_id"] for c in have}),
    }
    if len(have) != len(scored):
        res["MISSING_CELLS"] = [c for c in scored if c not in have]

    res["regions"] = {}
    for region in REGIONS:
        units, ceiling = build_units(have, ans, sel, region)
        r = {"edit_op_ceiling": ceiling, "conf": subset_metrics(units, "conf")}
        if region == PRIMARY_REGION and units:
            r["meeting_cluster_ci"] = meeting_cluster_ci(units, "conf")
            r["permutation_null"] = permutation_null(units, "conf", n=N_PERM)
            r["calibration_equal_width"] = bucket_table(units, "conf", EQUAL_WIDTH_EDGES)
            r["calibration_deciles"] = bucket_table(
                units, "conf", quantile_edges([u["conf"] for u in units]))
            for name, grp in (("overlap", [u for u in units if u["overlap"]]),
                              ("non_overlap", [u for u in units if not u["overlap"]])):
                r[f"split_{name}"] = subset_metrics(grp, "conf") if grp else None
            ins = [u for u in units if u["op"] in ("I", "M")]
            subs = [u for u in units if u["op"] in ("S", "M")]
            for u in ins:
                u["err"] = int(u["op"] == "I")
            r["insertions_vs_match"] = subset_metrics(ins, "conf")
            for u in ins:
                u["err"] = int(u["op"] != "M")
            for u in subs:
                u["err"] = int(u["op"] == "S")
            r["substitutions_vs_match"] = subset_metrics(subs, "conf")
            for u in subs:
                u["err"] = int(u["op"] != "M")
            ordered = sorted(units, key=lambda u: u["conf"])
            k = max(1, len(ordered) // 10)
            bot = ordered[:k]
            prev = sum(u["err"] for u in units) / len(units)
            br = sum(u["err"] for u in bot) / len(bot)
            r["bottom_decile"] = {"n": len(bot), "err_rate": br, "prevalence": prev,
                                  "lift": (br / prev) if prev else None}
        res["regions"][region] = r

    p = res["regions"][PRIMARY_REGION]["conf"]
    res["VERDICT"] = {
        "mean_within_meeting_auroc": p["mean_within_meeting_auroc"],
        "n_meetings_informative": p["n_meetings_informative"],
        "go_threshold": GO_THRESHOLD,
        "passes_go": (p["mean_within_meeting_auroc"] is not None
                      and p["mean_within_meeting_auroc"] >= GO_THRESHOLD),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["VERDICT"], ensure_ascii=False, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
