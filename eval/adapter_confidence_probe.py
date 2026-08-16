#!/usr/bin/env python3
"""Per-word confidence from `artifact-adapter-fixed`: gate, calibration,
head-to-head with Soniox.  (`exp-2026-08-16-adapter-confidence`)

Everything this module measures was fixed in
[`docs/specs/2026-08-16-adapter-confidence-prereg.md`](../docs/specs/2026-08-16-adapter-confidence-prereg.md)
**before the module existed** and before any outcome statistic was computed. Read
that spec, not this docstring, for the frozen definitions; the code must not add a
choice the spec does not already make.

Definitions are inherited from `eval/soniox_confidence_probe.py` wherever one exists,
so the two systems are measured the same way. Its `auroc`, `average_precision`,
`bucket_table`, `quantile_edges`, `per_meeting_auroc` and the alignment envelope are
imported rather than reimplemented.

Two substrates, never merged (CLAUDE.md):

  GOLD SET      fidelity-to-audio, 27 cores / 6 meetings, human-verified. The adapter
                hypotheses here were already produced with word_timestamps=True by the
                local server, so the probabilities attach to exactly the text that was
                scored - no re-decode, no gate risk. beam_size=2, CPU int8.
  247 WINDOWS   agreement-with-OpenCouncil, descriptive only. Needs the re-decode of
                `eval/adapter_confidence_decode.py`.

Soniox per-word confidence exists ONLY on the gold set, so the head-to-head is a
27-cell measurement.

Writes aggregates only. Transcript text never leaves the cache.

    SC=~/.cache/oc-public .venv-eval/bin/python eval/adapter_confidence_probe.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.scoring import wtoks                       # noqa: E402
from eval.gold_set_score import (                                    # noqa: E402
    GOLD, REGIONS, gold_view, in_intervals, load, masked_intervals, region_span,
)
from eval.soniox_confidence_probe import (                           # noqa: E402
    _align, auroc, average_precision, bucket_table, per_meeting_auroc,
    quantile_edges, word_units,
)  # noqa: E402
from eval.soniox_confidence_probe import group_words as snx_group_words  # noqa: E402

ADPDIR = GOLD / "hyp" / "adapter"
SNXDIR = GOLD / "hyp" / "soniox-tokens"
OUT = ROOT / "eval" / "results_adapter_confidence.json"

PRIMARY_REGION = "core_envelope"
INFORMATIVE_THRESHOLD = 0.60      # inherited from the Soniox probe; NOT a promotion gate
EQUAL_WIDTH_EDGES = [i / 10 for i in range(11)]
N_PERM = int(os.environ.get("N_PERM", "2000"))
N_BOOT = int(os.environ.get("N_BOOT", "2000"))
SEED = 21


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


# ------------------------------------------------------------------ adapter words
def adapter_words(cid: str) -> tuple[list[dict], dict]:
    """Emitted words of the cached adapter hypothesis, with their probabilities.

    The server returns `confidence` per word - faster-whisper's real per-word
    probability - and a per-utterance `confidence` = exp(avg_logprob), which is the
    SEGMENT PROXY. The proxy is attached to every word of its segment and is constant
    within it by construction (asserted in the tests).
    """
    d = json.loads((ADPDIR / f"{cid}.json").read_text())
    words, stats = [], Counter()
    for si, u in enumerate(d["transcription"]["utterances"]):
        seg_conf = u.get("confidence")
        if seg_conf is None:
            # a missing segment confidence must be REPORTED, not scored as a
            # confident-at-0.0 segment
            stats["utterances_without_segment_confidence"] += 1
        for wi, w in enumerate(u.get("words") or []):
            p = w.get("confidence")
            if p is None:
                stats["words_without_probability"] += 1
                continue
            if w.get("start") is None:
                stats["words_without_timestamp"] += 1
                continue
            words.append({"raw": w["word"],
                          "start": float(w["start"]),
                          "end": float(w["end"]) if w.get("end") is not None
                                 else float(w["start"]),
                          "p_word": float(p),
                          "seg_conf": float(seg_conf) if seg_conf is not None else 0.0,
                          "seg_idx": si,
                          "word_id": (cid, si, wi)})
    words.sort(key=lambda w: (w["start"], w["seg_idx"]))
    return words, dict(stats)


# ------------------------------------------------------------------- unit building
def word_units_ftoks(words):
    """`word_units`, but with the FILLER-STRIPPED tokenizer.

    The 247-window benchmark reference is tokenized with `eval_freeze.ftoks`, which
    drops hesitation fillers. Tokenizing the hypothesis with plain `wtoks` there would
    make every emitted «εεε» a spurious insertion. The gold set uses `wtoks` on both
    sides and is unaffected.
    """
    from eval.controlled_eval.eval_freeze import ftoks
    units, dropped, split = [], 0, 0
    for w in words:
        ts = ftoks(w["raw"])
        if not ts:
            dropped += 1
            continue
        if len(ts) > 1:
            split += 1
        for t in ts:
            u = dict(w)
            u["tok"] = t
            units.append(u)
    return units, dropped, split


def _units_from_words(words, seq, op_priority, reverse, cid, mid, ov_blocks,
                      tokenize=None):
    """Label every normalized hypothesis token M / S / I against the reference."""
    us, dropped, split = (tokenize or word_units)(words)
    hyp = [u["tok"] for u in us]
    ops = _align(seq, hyp, op_priority, reverse)
    lab = {}
    counts = Counter()
    for op, i, j in ops:
        counts[op] += 1
        if op in ("M", "S", "I"):
            lab[j] = (op, i)
    out = []
    for j, u in enumerate(us):
        got = lab.get(j)
        if got is None:
            continue
        op, ref_i = got
        u = dict(u)
        u["op"] = op
        u["err"] = int(op != "M")
        u["ref_i"] = ref_i if op in ("M", "S") else None
        u["cell"] = cid
        u["meeting"] = mid
        u["overlap"] = any(b["s"] <= u["start"] < b["e"] for b in ov_blocks)
        out.append(u)
    return out, counts, dropped, split


def build_units(scored, ans, sel, region: str, op_priority=("S", "D", "I"),
                reverse=False, system: str = "adapter"):
    """Emitted-word units for one system inside the scored intervals of one region."""
    units, ceiling = [], Counter()
    drop_punct = split_words = 0
    for cid in scored:
        mid = sel[cid]["meeting_id"]
        seq, _by, _ov, blocks, _kept = gold_view(ans[cid], region)
        span = region_span(blocks, region)
        if span is None or not seq:
            continue
        outside = [b for b in ans[cid]["b"]["blocks"]
                   if b["id"] not in {x["id"] for x in blocks}]
        ivs = masked_intervals(blocks, span, outside)
        ov_blocks = [b for b in blocks if b.get("ov_with") and not b.get("text_unc")]

        if system == "adapter":
            if not (ADPDIR / f"{cid}.json").exists():
                continue
            words, wstats = adapter_words(cid)
        else:
            p = SNXDIR / f"{cid}.json"
            if not p.exists():
                continue
            words, wstats = snx_group_words(json.loads(p.read_text())["tokens"])
            for i, w in enumerate(words):
                w["word_id"] = (cid, -1, i)
                w["seg_idx"] = -1
        for k, v in wstats.items():
            ceiling[k] += v
        words = [w for w in words if in_intervals(w["start"], ivs)]

        us, counts, dp, sw = _units_from_words(words, seq, op_priority, reverse,
                                               cid, mid, ov_blocks)
        ceiling.update(counts)
        drop_punct += dp
        split_words += sw
        units += us

    ceiling = dict(ceiling)
    tot = sum(ceiling.get(k, 0) for k in ("S", "D", "I"))
    ceiling["deletions"] = ceiling.get("D", 0)
    ceiling["deletion_share_of_edits"] = (ceiling.get("D", 0) / tot) if tot else None
    # (S + I) / (S + I + D): the share of edit OPERATIONS a per-word confidence can
    # even see. Not a ceiling on AUROC.
    ceiling["edit_operation_coverage"] = (
        (ceiling.get("S", 0) + ceiling.get("I", 0)) / tot) if tot else None
    ceiling["punctuation_only_words_dropped"] = drop_punct
    ceiling["words_splitting_into_multiple_tokens"] = split_words
    return units, ceiling


# ------------------------------------------------------------------ block statistics
def _blocks(units, score_key):
    """(meeting -> [(block_id, score, [row indices])]). The block is the emitted word:
    normalizer expansions share one probability, so permuting rows independently would
    violate exchangeability. The segment proxy blocks on the segment."""
    key = "seg_idx" if score_key == "seg_conf" else "word_id"
    by = defaultdict(lambda: defaultdict(list))
    for idx, u in enumerate(units):
        bid = (u["cell"], u[key]) if key == "seg_idx" else u["word_id"]
        by[u["meeting"]][bid].append(idx)
    return {m: [(b, units[ix[0]][score_key], ix) for b, ix in sorted(
        d.items(), key=lambda kv: str(kv[0]))] for m, d in by.items()}


def permutation_null(units, score_key, n=N_PERM, seed=SEED):
    """Permute the score at the BLOCK level, within meeting. One-sided AUROC > 0.5,
    p = (b + 1) / (B + 1)."""
    import numpy as np
    obs = per_meeting_auroc(units, score_key)["mean_within_meeting_auroc"]
    if obs is None:
        return {"observed": None}
    blocks = _blocks(units, score_key)
    rng = np.random.default_rng(seed)
    labels = {m: None for m in blocks}
    for m in blocks:
        labels[m] = [units[i]["err"] for _b, _s, ix in blocks[m] for i in ix]
    vals = []
    for _ in range(n):
        ms = []
        for m, bl in blocks.items():
            scores = np.array([s for _b, s, _ix in bl], dtype=float)
            rng.shuffle(scores)
            row_scores = [1.0 - scores[k] for k, (_b, _s, ix) in enumerate(bl)
                          for _ in ix]
            a = auroc(row_scores, labels[m])
            if a is not None:
                ms.append(a)
        if ms:
            vals.append(sum(ms) / len(ms))
    v = np.array(vals)
    return {"observed": obs, "null_mean": float(v.mean()),
            "null_p2.5": float(np.percentile(v, 2.5)),
            "null_p97.5": float(np.percentile(v, 97.5)),
            "p_value_one_sided": float((v >= obs).sum() + 1) / (len(v) + 1),
            "n_perm": len(vals), "permutation_unit": "emitted word"
            if score_key != "seg_conf" else "segment"}


def meeting_cluster_ci(units, score_key, n_boot=N_BOOT, seed=SEED):
    """Resample the meetings. Descriptive; 6 clusters is not significance."""
    import numpy as np
    by = defaultdict(list)
    for u in units:
        by[u["meeting"]].append(u)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    vals, dropped = [], 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        ms = [a for k in pick
              for a in [auroc([1.0 - u[score_key] for u in by[keys[k]]],
                              [u["err"] for u in by[keys[k]]])]
              if a is not None]
        if ms:
            vals.append(sum(ms) / len(ms))
        else:
            dropped += 1
    v = np.array(vals)
    return {"ci95_meeting_cluster": [float(np.percentile(v, 2.5)),
                                     float(np.percentile(v, 97.5))],
            "n_clusters": len(keys), "n_boot": len(vals),
            "replicates_dropped_no_informative_meeting": dropped}


def subset_metrics(units, key):
    s = [1.0 - u[key] for u in units]
    y = [u["err"] for u in units]
    prev = (sum(y) / len(y)) if y else None
    m = {"n": len(units), "n_err": sum(y), "prevalence": prev,
         "pooled_auroc": auroc(s, y), "average_precision": average_precision(s, y),
         "ap_null_prevalence": prev}
    pm = per_meeting_auroc(units, key)
    vals = [r["auroc"] for r in pm["per_meeting"].values() if r["auroc"] is not None]
    pm["n_meetings_undefined_auroc"] = len(pm["per_meeting"]) - len(vals)
    if vals:
        loo = [x for x in pm["leave_one_meeting_out_mean"].values() if x is not None]
        pm["loo_range"] = [min(loo), max(loo)] if loo else None
        pm["loo_all_above_0.5"] = bool(loo) and all(x > 0.5 for x in loo)
        pm["loo_all_above_0.60"] = bool(loo) and all(x > INFORMATIVE_THRESHOLD
                                                     for x in loo)
    m.update(pm)
    return m


# ------------------------------------------------------------------- rank machinery
def midranks(vals: list[float]) -> list[float]:
    """Average ranks for ties, scaled to [0, 1]. All-tied or singleton -> 0.5."""
    n = len(vals)
    if n <= 1 or len(set(vals)) == 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return [x / (n - 1) for x in r]


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rho on midranks."""
    if len(a) < 3 or len(set(a)) == 1 or len(set(b)) == 1:
        return None
    ra, rb = midranks(a), midranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return (num / (da * db)) if da and db else None


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a bucket's error rate. Codex asked for bucket
    uncertainty before the calibration curve is described as monotone."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def annotate_wilson(rows):
    for r in rows:
        r["wilson95"] = wilson(r["n_err"], r["n"])
    return rows


def bottom_decile_by_rank(rows, rank_key):
    """Flag the bottom decile BY RANK, keeping every observation tied at the cutoff.
    The realized flag rate is reported; ties are never broken to force exactly 10%."""
    if not rows:
        return None
    vals = sorted(r[rank_key] for r in rows)
    cut = vals[max(0, int(len(vals) * 0.10) - 1)]
    flagged = [r for r in rows if r[rank_key] <= cut]
    return {"cutoff": cut, "n_flagged": len(flagged),
            "realized_flag_rate": len(flagged) / len(rows)}


# ----------------------------------------------------------------- head-to-head
def head_to_head(scored, ans, sel, region=PRIMARY_REGION):
    """Shared columns: reference indices where BOTH systems emitted an aligned word.

    A conditioned subset - every deletion by either system and every insertion is
    excluded - so it is a fair same-row comparison and NOT a comparison of overall
    confidence quality.
    """
    adp, _ = build_units(scored, ans, sel, region, system="adapter")
    snx, _ = build_units(scored, ans, sel, region, system="soniox")

    def by_ref(units):
        out = {}
        for u in units:
            if u["ref_i"] is None:
                continue
            k = (u["cell"], u["ref_i"])
            assert k not in out, f"duplicate reference index {k}"
            out[k] = u
        return out

    A, S = by_ref(adp), by_ref(snx)
    shared = sorted(set(A) & set(S))
    rows = [{"cell": k[0], "ref_i": k[1], "meeting": A[k]["meeting"],
             "p_adp": A[k]["p_word"], "p_snx": S[k]["conf_min"],
             "err_adp": A[k]["err"], "err_snx": S[k]["err"]} for k in shared]
    if not rows:
        return {"n_shared": 0}

    # within-meeting percentile ranks among shared eligible columns
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["meeting"]].append(r)
    for m, rs in by_m.items():
        for key, dst in (("p_adp", "rank_adp"), ("p_snx", "rank_snx")):
            for r, v in zip(rs, midranks([x[key] for x in rs])):
                r[dst] = v
    for r in rows:
        r["rank_comb"] = (r["rank_adp"] + r["rank_snx"]) / 2.0

    def macro(rows, score, label):
        by = defaultdict(list)
        for r in rows:
            by[r["meeting"]].append(r)
        per = {}
        for m, rs in sorted(by.items()):
            per[m] = {"auroc": auroc([1.0 - r[score] for r in rs],
                                     [r[label] for r in rs]),
                      "n": len(rs), "n_err": sum(r[label] for r in rs)}
        vals = [v["auroc"] for v in per.values() if v["auroc"] is not None]
        return {"per_meeting": per, "n_meetings_informative": len(vals),
                "mean_within_meeting_auroc": (sum(vals) / len(vals)) if vals else None,
                "pooled_auroc": auroc([1.0 - r[score] for r in rows],
                                      [r[label] for r in rows])}

    co = Counter()
    for r in rows:
        co[("both" if r["err_adp"] and r["err_snx"] else
            "adapter_only" if r["err_adp"] else
            "soniox_only" if r["err_snx"] else "neither")] += 1

    res = {
        "n_shared": len(rows),
        "n_meetings": len(by_m),
        "note": ("conditioned subset: excludes every deletion by either system and "
                 "every insertion; not a comparison of overall confidence quality"),
        "error_cooccurrence": dict(co),
        "n_err_adapter": sum(r["err_adp"] for r in rows),
        "n_err_soniox": sum(r["err_snx"] for r in rows),
        "spearman_per_meeting": {m: spearman([r["p_adp"] for r in rs],
                                             [r["p_snx"] for r in rs])
                                 for m, rs in sorted(by_m.items())},
        "spearman_pooled_secondary": spearman([r["p_adp"] for r in rows],
                                              [r["p_snx"] for r in rows]),
        "ties": {
            "adapter": {"n_unique": len({r["p_adp"] for r in rows}),
                        "tied_share": 1 - len({r["p_adp"] for r in rows}) / len(rows)},
            "soniox": {"n_unique": len({r["p_snx"] for r in rows}),
                       "tied_share": 1 - len({r["p_snx"] for r in rows}) / len(rows)},
        },
        "predict_adapter_error": {
            "adapter_alone": macro(rows, "rank_adp", "err_adp"),
            "soniox_alone": macro(rows, "rank_snx", "err_adp"),
            "combination_mean_rank": macro(rows, "rank_comb", "err_adp"),
        },
        "predict_soniox_error": {
            "soniox_alone": macro(rows, "rank_snx", "err_snx"),
            "adapter_alone": macro(rows, "rank_adp", "err_snx"),
            "combination_mean_rank": macro(rows, "rank_comb", "err_snx"),
        },
    }
    # who is uncertain: 2x2 of the two bottom-decile flags
    fa = bottom_decile_by_rank(rows, "rank_adp")
    fs = bottom_decile_by_rank(rows, "rank_snx")
    tab = Counter()
    for r in rows:
        tab[(r["rank_adp"] <= fa["cutoff"], r["rank_snx"] <= fs["cutoff"])] += 1
    # flags are chosen WITHIN meeting, so the independence expectation must be
    # stratified by meeting too (Codex, findings review 6dd13192).
    strat_flag = sum(
        sum(1 for r in rs if r["rank_adp"] <= fa["cutoff"])
        * sum(1 for r in rs if r["rank_snx"] <= fs["cutoff"]) / len(rs)
        for rs in by_m.values())
    res["uncertainty_flag_agreement"] = {
        "adapter_flag": fa, "soniox_flag": fs,
        "both": tab[(True, True)], "adapter_only": tab[(True, False)],
        "soniox_only": tab[(False, True)], "neither": tab[(False, False)],
        "expected_both_pooled_independence": (
            (fa["n_flagged"] * fs["n_flagged"]) / len(rows)),
        "expected_both_meeting_stratified": strat_flag,
    }
    # error co-occurrence: the same stratification question
    n11, n10, n01 = co["both"], co["adapter_only"], co["soniox_only"]
    n00 = co["neither"]
    res["error_cooccurrence_detail"] = {
        "expected_both_pooled_independence": (
            res["n_err_adapter"] * res["n_err_soniox"] / len(rows)),
        "expected_both_meeting_stratified": sum(
            sum(r["err_adp"] for r in rs) * sum(r["err_snx"] for r in rs) / len(rs)
            for rs in by_m.values()),
        "odds_ratio": ((n11 * n00) / (n10 * n01)) if n10 and n01 else None,
        "share_of_adapter_errors_shared": (n11 / res["n_err_adapter"])
                                          if res["n_err_adapter"] else None,
        "share_of_soniox_errors_shared": (n11 / res["n_err_soniox"])
                                         if res["n_err_soniox"] else None,
    }

    # paired per-meeting AUROC differences for the combination (Codex): the point
    # estimate on its own is not allowed to carry the claim.
    def paired(label, base, comb):
        import numpy as np
        ms = sorted(by_m)
        diffs = {}
        for m in ms:
            rs = by_m[m]
            a = auroc([1.0 - r[base] for r in rs], [r[label] for r in rs])
            b = auroc([1.0 - r[comb] for r in rs], [r[label] for r in rs])
            diffs[m] = (None if a is None or b is None else b - a)
        pairs = [(m, v) for m, v in diffs.items() if v is not None]
        vals = [v for _m, v in pairs]
        if not vals:
            return {"per_meeting": diffs}
        # iterate the INFORMATIVE meetings only; zipping all meetings against the
        # filtered values would misalign as soon as one meeting is undefined.
        loo = [sum(v for k, v in pairs if k != m) / (len(vals) - 1)
               for m, _v in pairs] if len(vals) > 1 else []
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(vals), len(vals))
            boot.append(float(np.mean([vals[k] for k in pick])))
        return {"per_meeting": diffs, "mean": sum(vals) / len(vals),
                "range": [min(vals), max(vals)], "n_positive": sum(v > 0 for v in vals),
                "n_meetings": len(vals),
                "leave_one_meeting_out_mean_range": ([min(loo), max(loo)] if loo
                                                     else None),
                "ci95_meeting_cluster_descriptive": [
                    float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}

    res["combination_paired_gain"] = {
        "predict_adapter_error_vs_adapter_alone":
            paired("err_adp", "rank_adp", "rank_comb"),
        "predict_soniox_error_vs_soniox_alone":
            paired("err_snx", "rank_snx", "rank_comb"),
    }
    return res


# ------------------------------------------------------------------- 247 windows
def gate_and_247():
    """The gate contrasts, and the descriptive agreement-metric arm."""
    from eval.adapter_confidence_decode import RUN_ID, substrate
    from eval.controlled_eval.eval_freeze import ftoks
    from eval.controlled_eval.exp_same_stack import sdi
    from eval.controlled_eval.scoring import edist

    d = sc() / "adapter-confidence"
    paths = {"wt_true": d / "wt-true-247.json", "wt_false": d / "wt-false-247.json"}
    have = {k: json.loads(p.read_text()) for k, p in paths.items() if p.exists()}
    items = {it["item_id"]: it for it in substrate()}
    armA = sc() / "decode-ablation" / "eval-A.json"
    cached_local = json.loads(armA.read_text())["windows"] if armA.exists() else {}

    out = {"n_windows": len(items),
           "decoded": {k: len(v["windows"]) for k, v in have.items()},
           "config": have.get("wt_true", have.get("wt_false", {})).get("config")}

    def text_of(src, wid):
        if src == "cached_gpu":
            return items[wid]["hyp"]["oc-runpod-fixed-2026-08-10"]
        if src == "local_cpu_wt_false_armA":
            return cached_local.get(wid, {}).get("text")
        return have[src]["windows"].get(wid, {}).get("text")

    def toks_of(src, wid, segment_aware):
        """Token sequence for one window.

        The stored `text` is `"".join(segment.text)`, and faster-whisper does NOT always
        put a leading space on a segment: 505 of 1,677 segment boundaries here do not
        have one, so the joined string FUSES the last word of a segment with the first
        word of the next. `word_timestamps` moves segment boundaries, so scoring the
        joined string charges the gate for fusions that are an artifact of the join.
        Where segments are available the tokens are built per segment instead. The two
        cached sources (the GPU benchmark text and the arm-A local control) store only
        the joined string, so contrasts against them stay on it, and say so.
        """
        if segment_aware and src in have:
            w = have[src]["windows"].get(wid)
            return None if w is None else [t for s in w["segments"]
                                           for t in wtoks(s["text"])]
        t = text_of(src, wid)
        return None if t is None else wtoks(t)

    def contrast(a, b, ids, segment_aware=True):
        rows = []
        for wid in ids:
            na, nb = toks_of(a, wid, segment_aware), toks_of(b, wid, segment_aware)
            if na is None or nb is None:
                continue
            ta, tb = text_of(a, wid), text_of(b, wid)
            rows.append({"wid": wid, "raw_exact": ta == tb, "norm_exact": na == nb,
                         "n_a": len(na), "n_b": len(nb), "e": edist(na, nb)})
        if not rows:
            return None
        ea = sum(r["e"] for r in rows)
        sa = sum(r["n_a"] for r in rows)
        sb = sum(r["n_b"] for r in rows)
        per = sorted(r["e"] / r["n_a"] for r in rows if r["n_a"])
        return {"n": len(rows),
                "raw_identical_rate": sum(r["raw_exact"] for r in rows) / len(rows),
                "norm_identical_rate": sum(r["norm_exact"] for r in rows) / len(rows),
                "pooled_wer_denominator_a": ea / sa if sa else None,
                "pooled_wer_denominator_b": ea / sb if sb else None,
                "symmetric_edit_rate": (2 * ea / (sa + sb)) if (sa + sb) else None,
                "per_window_wer_a": {"min": per[0], "p50": per[len(per) // 2],
                                     "max": per[-1]} if per else None,
                "n_windows_changed": sum(1 for r in rows if not r["norm_exact"])}

    # The ANALYSIS SET is the windows completed in BOTH local passes (prereg
    # amendment, 2026-08-16 20:05: randomized order, wall-clock stop). The two passes
    # walk the identical seeded order, so the slower one's set is nested in the
    # faster one's.
    ids = sorted(set(have.get("wt_true", {}).get("windows", {}))
                 & set(have.get("wt_false", {}).get("windows", {})))
    out["analysis_set"] = {
        "n": len(ids), "n_substrate": len(items),
        "cities": dict(Counter(items[w]["city_id"] for w in ids)),
        "n_meetings": len({items[w]["meeting_id"] for w in ids}),
        "construction": ("substrate-order head decoded before the amendment "
                         "(city-clustered) + seeded random sample of the remainder"),
    }
    # The 23:45 stop was not enforced (the watchdog died with the session) and the
    # realized set overran it. What CANNOT have been affected: the 9 windows already
    # decoded in both passes before the 20:05 amendment. They are in the deadline set
    # by construction, whatever the deadline was. If they already fail the conjunction,
    # the overrun cannot have created the gate failure.
    pre = [it["item_id"] for it in substrate()][:9]
    if all(w in ids for w in pre):
        pre_same = [w for w in pre
                    if toks_of("wt_false", w, True) == toks_of("wt_true", w, True)]
        out["analysis_set"]["pre_amendment_head"] = {
            "n": len(pre), "cities": sorted({items[w]["city_id"] for w in pre}),
            "n_normalized_identical": len(pre_same),
            "note": ("decoded in substrate order before the 20:05 amendment, so inside "
                     "the 23:45 deadline set regardless of when decoding stopped"),
        }
    overlap = [w for w in ids if w in cached_local]
    # Same config, different run: how much of the gate difference could be run-to-run
    # nondeterminism rather than `word_timestamps`? Only comparable on the SAME windows.
    out["rerun_noise_vs_gate_same_windows"] = {
        "n": len(overlap),
        "rerun_same_config": contrast("local_cpu_wt_false_armA", "wt_false", overlap,
                                      segment_aware=False),
        "gate_wtF_vs_wtT": contrast("wt_false", "wt_true", overlap,
                                    segment_aware=False),
        "note": ("the cached arm-A local control and this pass share every decode "
                 "argument and the per-window seed; any difference is run-to-run "
                 "nondeterminism (thread count was not recorded for arm A). arm A "
                 "stores only the joined string, so BOTH rows here are scored on the "
                 "joined string - matched, and comparable only to each other."),
    }
    # The two cached-GPU contrasts are TOKENIZED FROM DIFFERENT SOURCES: the local side
    # per segment (correct), the benchmark side from its stored string (the only form
    # it exists in, and the form every other experiment scores it in). Both variants are
    # reported rather than one being chosen, because neither is unambiguously right.
    out["contrasts"] = {
        "stack__cachedGPU_wtF_vs_localCPU_wtF": contrast("cached_gpu", "wt_false", ids),
        "stack__both_sides_on_the_stored_string": contrast("cached_gpu", "wt_false",
                                                           ids, segment_aware=False),
        "gate__localCPU_wtF_vs_localCPU_wtT": contrast("wt_false", "wt_true", ids),
        "endtoend__cachedGPU_wtF_vs_localCPU_wtT": contrast("cached_gpu", "wt_true",
                                                            ids),
        "endtoend__both_sides_on_the_stored_string": contrast("cached_gpu", "wt_true",
                                                              ids, segment_aware=False),
        "gate_on_the_joined_string__superseded_by_the_segment_aware_row": contrast(
            "wt_false", "wt_true", ids, segment_aware=False),
        "reproducibility__localCPU_wtF_vs_armA": (
            contrast("local_cpu_wt_false_armA", "wt_false", overlap,
                     segment_aware=False)),
        "n_overlap_windows_with_cached_local_control": len(overlap),
    }
    g = out["contrasts"]["gate__localCPU_wtF_vs_localCPU_wtT"]
    out["gate"] = {
        "criterion": ("normalized token sequence identical in EVERY window of the "
                      "analysis set; a conjunction, so a smaller sample can fail it "
                      "but cannot pass it as convincingly"),
        "n_analysis_set": len(ids),
        "passed": bool(g and g["norm_identical_rate"] == 1.0),
        "note": ("if this is false, the confidences belong to THIS re-run and cannot "
                 "be attached to the frozen fusion input W"),
    }

    if ids and "wt_true" in have and "wt_false" in have:
        out["gate_anatomy"] = gate_anatomy(have, items, ids)

    # descriptive arm: agreement-with-OpenCouncil, never merged with the gold set.
    # Runs on the same analysis set as the gate, so the two are comparable.
    if ids:
        units = []
        ceiling = Counter()
        for wid in ids:
            it = items[wid]
            w = have["wt_true"]["windows"].get(wid)
            if not w:
                continue
            words = [{"raw": x["w"], "start": x["s"], "end": x["e"], "p_word": x["p"],
                      "seg_conf": (math.exp(s["avg_logprob"])
                                   if s["avg_logprob"] is not None else 0.0),
                      "seg_idx": si, "word_id": (wid, si, k)}
                     for si, s in enumerate(w["segments"])
                     for k, x in enumerate(s["words"])]
            us, counts, _dp, _sw = _units_from_words(
                words, ftoks(it["ref"]), ("S", "D", "I"), False, wid,
                it["meeting_id"], [], tokenize=word_units_ftoks)
            ceiling.update(counts)
            units += us
        tot = sum(ceiling.get(k, 0) for k in ("S", "D", "I"))
        out["descriptive_247"] = {
            "metric": "agreement-with-OpenCouncil, NOT fidelity-to-audio",
            "n_windows": len(ids),
            "n_substrate": len(items),
            "n_meetings": len({items[w]["meeting_id"] for w in ids}),
            "edit_operation_coverage": ((ceiling.get("S", 0) + ceiling.get("I", 0))
                                        / tot) if tot else None,
            "deletion_share_of_edits": (ceiling.get("D", 0) / tot) if tot else None,
            "p_word": subset_metrics(units, "p_word"),
            "seg_conf": subset_metrics(units, "seg_conf"),
            "calibration_equal_width_p_word": bucket_table(units, "p_word",
                                                           EQUAL_WIDTH_EDGES),
            "calibration_deciles_p_word": bucket_table(
                units, "p_word", quantile_edges([u["p_word"] for u in units])),
            "insertions_vs_match": _op_subset(units, "I"),
            "substitutions_vs_match": _op_subset(units, "S"),
        }
        # ftoks (filler-stripped) is the benchmark's own normalizer; note the deviation
        out["descriptive_247"]["normalizer"] = "eval_freeze.ftoks (filler-stripped)"
    return out


def gate_anatomy(have, items, ids):
    """What the `word_timestamps` difference IS, once the gate has already failed.

    Everything here is fixed BEFORE it was run, on 2026-08-17, and none of it is a
    threshold that anything is promoted on:

      * tokenizer          `eval_freeze.ftoks`, the benchmark's own normalizer, on both
                           sides; the `wtoks` identity rates in `contrasts` stay as they
                           are and are reported next to these.
      * diff typing        `exp_same_stack.sdi(ref=wt_false, hyp=wt_true)`. D = a word
                           the wt=F pass emitted and the wt=T pass did not; I = a word
                           only wt=T emitted; S = a word that changed.
      * reference profile  `sdi(ftoks(ref), ftoks(text))` per pass per window, pooled
                           over the analysis set, denominator = reference tokens.
      * paired CI          2,000 bootstrap resamples of the MEETINGS in the analysis
                           set, seed 21. Descriptive.
      * per-window bins    [0], (0, .02], (.02, .05], (.05, .10], (.10, .20], (.20, 1].
      * concentration      share of the total F-vs-T edit distance contributed by the
                           10% most-changed windows.
    """
    import numpy as np

    from eval.controlled_eval.eval_freeze import ftoks
    from eval.controlled_eval.exp_same_stack import sdi
    from eval.controlled_eval.scoring import edist

    BIN_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 1.01]

    def segtoks(w):
        """Per SEGMENT, never off the joined string - see `toks_of`. `word_timestamps`
        moves segment boundaries, and 505 of 1,677 boundaries have no leading space, so
        the joined string fuses words across exactly the boundaries under test."""
        return [t for s in w["segments"] for t in ftoks(s["text"])]

    rows = []
    for wid in ids:
        it = items[wid]
        wf = have["wt_false"]["windows"][wid]
        wt = have["wt_true"]["windows"][wid]
        ref = ftoks(it["ref"])
        a, b = segtoks(wf), segtoks(wt)
        s_ft, d_ft, i_ft = sdi(a, b)
        sf, df, if_ = sdi(ref, a)
        st, dt, it_ = sdi(ref, b)
        rows.append({
            "wid": wid, "meeting": it["meeting_id"], "city": it["city_id"],
            "n_ref": len(ref), "n_a": len(a), "n_b": len(b),
            "ft_S": s_ft, "ft_D": d_ft, "ft_I": i_ft, "ft_e": edist(a, b),
            "raw_exact": wf["text"] == wt["text"],
            "ftok_exact": a == b,
            "segs_false": wf["n_segments"], "segs_true": wt["n_segments"],
            "F": {"S": sf, "D": df, "I": if_}, "T": {"S": st, "D": dt, "I": it_},
        })

    n_ref = sum(r["n_ref"] for r in rows)
    n_a = sum(r["n_a"] for r in rows)

    def prof(k):
        S = sum(r[k]["S"] for r in rows)
        D = sum(r[k]["D"] for r in rows)
        I = sum(r[k]["I"] for r in rows)
        return {"S": S, "D": D, "I": I, "n_ref_tokens": n_ref,
                "n_hyp_tokens": n_ref - D + I,
                "wer": (S + D + I) / n_ref, "deletion_rate": D / n_ref,
                "insertion_rate": I / n_ref, "substitution_rate": S / n_ref}

    def boot_paired(fn):
        """Paired per-window quantity, resampled by meeting. Ratio-of-sums estimator."""
        by = defaultdict(list)
        for r in rows:
            by[r["meeting"]].append(r)
        keys = sorted(by)
        rng = np.random.default_rng(SEED)
        obs = fn(rows)
        vals = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(keys), len(keys))
            samp = [r for k in pick for r in by[keys[k]]]
            vals.append(fn(samp))
        v = np.array(vals, dtype=float)
        return {"estimate": obs,
                "ci95_meeting_cluster_descriptive": [float(np.percentile(v, 2.5)),
                                                     float(np.percentile(v, 97.5))],
                "n_clusters": len(keys), "n_boot": len(vals)}

    def _rate(rs, arm, op):
        den = sum(r["n_ref"] for r in rs)
        return sum(r[arm][op] for r in rs) / den if den else float("nan")

    delta = {op: boot_paired(lambda rs, o=op: _rate(rs, "T", o) - _rate(rs, "F", o))
             for op in ("S", "D", "I")}
    delta["WER"] = boot_paired(
        lambda rs: sum(sum(r["T"].values()) for r in rs) / sum(r["n_ref"] for r in rs)
        - sum(sum(r["F"].values()) for r in rs) / sum(r["n_ref"] for r in rs))

    per_window = sorted(r["ft_e"] / r["n_a"] for r in rows if r["n_a"])
    # the exact-zero bin is its own bin; the rest are half-open (lo, hi]
    hist = [{"label": "exactly 0", "lo": 0.0, "hi": 0.0,
             "n": sum(1 for x in per_window if x == 0)}]
    # the top bin is unbounded: e / n_a can exceed 1 when the wt=True pass emits far
    # more tokens. Nothing here does, so this changes no count - it stops the partition
    # from silently dropping a window if one ever did.
    edges = BIN_EDGES[:-1] + [float("inf")]
    for lo, hi in zip(edges, edges[1:]):
        hist.append({"label": f"({lo}, {hi}]", "lo": lo, "hi": hi,
                     "n": sum(1 for x in per_window if lo < x <= hi)})
    assert sum(h["n"] for h in hist) == len(per_window)

    tot_e = sum(r["ft_e"] for r in rows)
    top = sorted(rows, key=lambda r: -r["ft_e"])[:max(1, len(rows) // 10)]
    seg_delta = [r["segs_true"] - r["segs_false"] for r in rows]

    return {
        "frozen_before_running": ("tokenizer ftoks; diff typed by sdi(wt_false, "
                                 "wt_true); bins [0],(0,.02],(.02,.05],(.05,.10],"
                                 "(.10,.20],(.20,1]; 2000 meeting-cluster resamples, "
                                 "seed 21; fixed 2026-08-17 before any of it was run"),
        "n_windows": len(rows), "n_meetings": len({r["meeting"] for r in rows}),
        "identity": {
            "raw_identical": sum(r["raw_exact"] for r in rows),
            "ftok_identical": sum(r["ftok_exact"] for r in rows),
            "differ_only_in_punctuation_or_spacing": sum(
                1 for r in rows if not r["raw_exact"] and r["ftok_exact"]),
        },
        "wt_false_vs_wt_true_ops": {
            "S": sum(r["ft_S"] for r in rows), "D": sum(r["ft_D"] for r in rows),
            "I": sum(r["ft_I"] for r in rows),
            "denominator_wt_false_tokens": n_a,
            "pooled_wer_denominator_wt_false": tot_e / n_a if n_a else None,
            "note": ("D = a token the wt=False pass emitted and the wt=True pass did "
                     "not; I = a token only wt=True emitted"),
        },
        "reference_anchored": {
            "metric": "agreement-with-OpenCouncil (ftoks on both sides)",
            "wt_false": prof("F"), "wt_true": prof("T"),
            "paired_delta_true_minus_false": delta,
        },
        "reconciliation_with_the_descriptive_arm": {
            "note": ("the descriptive 247-window arm builds its rows from the per-word "
                     "output; this block aligns whole segment texts. They must agree "
                     "token for token, and error for error, or one of them is wrong."),
            "wt_true_hyp_tokens_here": n_ref - sum(r["T"]["D"] for r in rows)
                                       + sum(r["T"]["I"] for r in rows),
            "wt_true_S_plus_I_here": sum(r["T"]["S"] + r["T"]["I"] for r in rows),
        },
        "per_window_deletion_delta": {
            "note": ("per-window (D_true - D_false). A near-zero pooled delta can hide "
                     "large per-window swings that cancel."),
            "sum": sum(r["T"]["D"] - r["F"]["D"] for r in rows),
            "n_windows_nonzero": sum(1 for r in rows if r["T"]["D"] != r["F"]["D"]),
            "n_windows_abs_ge_10": sum(
                1 for r in rows if abs(r["T"]["D"] - r["F"]["D"]) >= 10),
            "sum_abs": sum(abs(r["T"]["D"] - r["F"]["D"]) for r in rows),
            "min": min(r["T"]["D"] - r["F"]["D"] for r in rows),
            "max": max(r["T"]["D"] - r["F"]["D"] for r in rows),
        },
        "per_window_change": {
            "bins": hist,
            "min": per_window[0], "p25": per_window[len(per_window) // 4],
            "p50": per_window[len(per_window) // 2],
            "p75": per_window[3 * len(per_window) // 4], "max": per_window[-1],
            "top_decile_share_of_total_edits": (
                sum(r["ft_e"] for r in top) / tot_e) if tot_e else None,
            "n_windows_in_top_decile": len(top),
        },
        "segmentation": {
            "n_windows_with_different_segment_count": sum(1 for x in seg_delta if x),
            "mean_segment_delta_true_minus_false": sum(seg_delta) / len(seg_delta),
            "range": [min(seg_delta), max(seg_delta)],
            "n_more_segments_under_wt_true": sum(1 for x in seg_delta if x > 0),
            "n_fewer_segments_under_wt_true": sum(1 for x in seg_delta if x < 0),
        },
    }


def _op_subset(units, op):
    sub = [dict(u, err=int(u["op"] == op)) for u in units if u["op"] in (op, "M")]
    return subset_metrics(sub, "p_word") if sub else None


# -------------------------------------------------------------------------- main
def main():
    man, ans, _cells, sel = load()
    scored = man["scored_cell_ids"]
    res = {
        "protocol": "adapter-confidence-2026-08-16",
        "prereg": "docs/specs/2026-08-16-adapter-confidence-prereg.md",
        "system": "artifact-adapter-fixed via faster-whisper CTranslate2 int8",
        "answers_sha256": man["answers_sha256"],
        "frozen_before_looking": {
            "unit": "normalized hypothesis-token instance (0-token words excluded, "
                    ">1-token words duplicated with the same probability)",
            "primary_statistic": "mean of within-meeting AUROC of 1 - p_word, ties 0.5",
            "informative_threshold": INFORMATIVE_THRESHOLD,
            "threshold_status": "inherited from the Soniox probe for comparability; "
                                "NOT a promotion gate",
            "auroc_null": 0.5,
            "primary_region": PRIMARY_REGION,
            "permutation_unit": "emitted word (segment, for the segment proxy)",
            "segment_proxy": "exp(avg_logprob) - a RANKING score; no calibrated word "
                             "probability is claimed",
            "combination": "mean of within-meeting percentile ranks; no fitted weight",
            "gold_decode": "local server, beam_size=2, CPU int8 - NOT the 247-window "
                           "benchmark config (beam 5, RunPod GPU)",
        },
        "n_cells_scored": len(scored),
        "n_meetings": len({sel[c]["meeting_id"] for c in scored}),
    }

    res["regions"] = {}
    for region in REGIONS:
        units, ceiling = build_units(scored, ans, sel, region)
        r = {"edit_op_ceiling": ceiling,
             "p_word": subset_metrics(units, "p_word"),
             "seg_conf_segment_proxy": subset_metrics(units, "seg_conf")}
        if region == PRIMARY_REGION and units:
            r["calibration_equal_width_p_word"] = annotate_wilson(
                bucket_table(units, "p_word", EQUAL_WIDTH_EDGES))
            r["calibration_deciles_p_word"] = annotate_wilson(bucket_table(
                units, "p_word", quantile_edges([u["p_word"] for u in units])))
            r["calibration_equal_width_seg_conf"] = bucket_table(units, "seg_conf",
                                                                 EQUAL_WIDTH_EDGES)
            for name, grp in (("overlap", [u for u in units if u["overlap"]]),
                              ("non_overlap", [u for u in units if not u["overlap"]])):
                r[f"split_{name}"] = subset_metrics(grp, "p_word") if grp else None
            r["insertions_vs_match"] = _op_subset(units, "I")
            r["substitutions_vs_match"] = _op_subset(units, "S")
            ordered = sorted(units, key=lambda u: u["p_word"])
            k = max(1, len(ordered) // 10)
            cut = ordered[k - 1]["p_word"]
            bot = [u for u in units if u["p_word"] <= cut]
            prev = sum(u["err"] for u in units) / len(units)
            br = sum(u["err"] for u in bot) / len(bot)
            r["bottom_decile"] = {"n": len(bot),
                                  "realized_rate": len(bot) / len(units),
                                  "err_rate": br, "prevalence": prev,
                                  "lift": (br / prev) if prev else None}
            r["permutation_null_p_word"] = permutation_null(units, "p_word")
            r["permutation_null_seg_conf"] = permutation_null(units, "seg_conf")
            r["meeting_cluster_ci_p_word"] = meeting_cluster_ci(units, "p_word")
            # paired per-meeting AUROC differences: word signal vs segment proxy
            pw = per_meeting_auroc(units, "p_word")["per_meeting"]
            sg = per_meeting_auroc(units, "seg_conf")["per_meeting"]
            r["proxy_vs_word_paired_per_meeting"] = {
                m: {"p_word": pw[m]["auroc"], "seg_conf": sg[m]["auroc"],
                    "diff": (None if pw[m]["auroc"] is None or sg[m]["auroc"] is None
                             else pw[m]["auroc"] - sg[m]["auroc"])}
                for m in sorted(pw)}
            env = []
            for pri in permutations(("S", "D", "I")):
                for rev in (False, True):
                    us2, _c = build_units(scored, ans, sel, region, pri, rev)
                    if not us2:
                        continue
                    pm = per_meeting_auroc(us2, "p_word")
                    env.append({"priority": "".join(pri), "reversed": rev,
                                "mean_within_meeting_auroc":
                                    pm["mean_within_meeting_auroc"],
                                "pooled_auroc": auroc([1 - u["p_word"] for u in us2],
                                                      [u["err"] for u in us2]),
                                "n_S": sum(1 for u in us2 if u["op"] == "S"),
                                "n_I": sum(1 for u in us2 if u["op"] == "I"),
                                "n": len(us2)})
            r["alignment_sensitivity"] = env
        res["regions"][region] = r

    res["head_to_head_soniox"] = head_to_head(scored, ans, sel)
    try:
        res["gate_and_247"] = gate_and_247()
    except Exception as e:                                          # noqa: BLE001
        res["gate_and_247"] = {"error": repr(e)}

    p = res["regions"][PRIMARY_REGION]["p_word"]["mean_within_meeting_auroc"]
    res["headline"] = {
        "mean_within_meeting_auroc_p_word": p,
        "informative_threshold": INFORMATIVE_THRESHOLD,
        "informative": bool(p is not None and p >= INFORMATIVE_THRESHOLD),
        "note": "27 cells / 6 meetings. Not a population claim, not a promotion gate.",
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")
    print(json.dumps(res["headline"], ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
