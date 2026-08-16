#!/usr/bin/env python3
"""Score the frozen gold set (`exp-2026-08-16-gold-set`) — ONE confirmatory pass.

The asset is `~/.cache/oc-public/gold-set-2026-08/`, frozen by sha256 on
2026-08-16: 27 scored 15 s cores in 6 meetings / 6 cities, human-verified word by
word AND by speaker, with 39 overlap-participating blocks.

ABSOLUTE RULE, from the issue and the prereg: nothing here may be tuned on this
asset. Every region, every threshold and every rule below was written down before
any number was computed, and all variants are reported together — none is selected
after the fact.

WHAT IS NOT MEASURED HERE
  W, the per-column composition of `exp-2026-08-16-composition-over-selection`,
  cannot be run on this audio: it is defined over ElevenLabs Scribe v2 + Soniox +
  our adapter, and no ElevenLabs credential exists in this environment (GPU spend
  is forbidden, so Scribe cannot be re-derived either). What is measured instead is
  a finite-corpus audit of the lexical evidence available OUTSIDE the adapter. It
  is not W's added-word precision and must never be quoted as such.

SYSTEMS
  ADP    artifact-ct2-fixed over the local CPU endpoint. Word timestamps, NO speakers.
  PUB    the published OpenCouncil transcript — the production pipeline output
         (pyannote /identify -> Scribe v2 `no_verbatim` -> applyDiarization ->
         claude fixTranscript). Utterance timestamps + speaker segments. This is
         also the reference every "agreement-with-OpenCouncil" WER in this project
         was computed against.
  TURBO  the PREFILL (pyannote precision-2 + faster-whisper-large-v3-turbo). The
         human corrected this text, so it is ANCHORED to the gold by construction
         and its fidelity numbers are biased downward. Reported, never ranked.
  SNX    Soniox, whole-clip text, NO timestamps. Used only as a position-free
         lexical witness with multiset capacity, never for WER.

REGIONS — all three always reported, no selection among them
  R1 core-strict     gold blocks wholly inside [10, 25) of the clip
  R2 core-envelope   gold blocks intersecting [10, 25), taken whole   (primary)
  R3 clip            every gold block

Writes aggregates only. Transcript text never leaves the cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.scoring import wtoks  # noqa: E402

GOLD = Path.home() / ".cache/oc-public/gold-set-2026-08"
HYP = GOLD / "hyp"
MEET = Path.home() / ".cache/oc-public/meetings"
CORE = (10.0, 25.0)          # clip-relative core, identical for all 27 cells
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
OUT = ROOT / "eval" / "results_gold_set.json"

EXPECT_ANSWERS_SHA = "fa7a5dec12a9d2787e4151d2388388b480e19de8a35c8fb7e3585da3a5d1f063"
EXPECT_CELLS_SHA = "7f734db8c0de2a4cc07892544f806607a29b190a5819da8c33ed238ac2b1b46a"


# --------------------------------------------------------------------- alignment
def align_ops(ref: list[str], hyp: list[str]) -> list[tuple[str, int, int]]:
    """Levenshtein backtrace as ops: (op, ref_idx, hyp_idx), op in M/S/D/I.

    Deterministic tie-break: substitution, then deletion, then insertion, and the
    backtrace walks the same order. Two callers must never get different alignments
    for the same input.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], op[i][0] = i, "D"
    for j in range(1, m + 1):
        d[0][j], op[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j], op[i][j] = d[i - 1][j - 1], "M"
                continue
            sub, dele, ins = d[i - 1][j - 1] + 1, d[i - 1][j] + 1, d[i][j - 1] + 1
            best = min(sub, dele, ins)
            d[i][j] = best
            op[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o in ("M", "S"):
            out.append((o, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif o == "D":
            out.append(("D", i - 1, -1))
            i -= 1
        else:
            out.append(("I", -1, j - 1))
            j -= 1
    out.reverse()
    return out


def sdi_counts(ref: list[str], hyp: list[str]) -> dict:
    ops = align_ops(ref, hyp)
    c = Counter(o for o, _, _ in ops)
    return {"S": c["S"], "D": c["D"], "I": c["I"], "M": c["M"],
            "N": len(ref), "err": c["S"] + c["D"] + c["I"]}


def cp_wer(ref_by_spk: dict[str, list[str]], hyp_by_spk: dict[str, list[str]]) -> dict:
    """cpWER: concatenate per speaker, take the best one-to-one speaker assignment.

    Missing/extra speakers are padded with empty streams on either side, so the
    permutation is always over a square problem. Capped at 7 speakers a side.
    """
    rk = sorted(ref_by_spk)
    hk = sorted(hyp_by_spk)
    k = max(len(rk), len(hk))
    if k > 7:
        raise ValueError("too many speakers for exhaustive cpWER")
    rs = [ref_by_spk[x] for x in rk] + [[]] * (k - len(rk))
    hs = [hyp_by_spk[x] for x in hk] + [[]] * (k - len(hk))
    best, best_perm = None, None
    for perm in permutations(range(k)):
        tot = {"S": 0, "D": 0, "I": 0, "M": 0, "N": 0, "err": 0}
        for a in range(k):
            c = sdi_counts(rs[a], hs[perm[a]])
            for key in tot:
                tot[key] += c[key]
        if best is None or tot["err"] < best["err"]:
            best, best_perm = tot, perm
    mapping = {}
    for a in range(k):
        if a < len(rk) and best_perm[a] < len(hk):
            mapping[rk[a]] = hk[best_perm[a]]
    best["mapping"] = mapping
    return best


# --------------------------------------------------------------------- bootstrap
def cluster_ci(per_cell: list[tuple[str, int, int]], n_boot: int = N_BOOT, seed: int = 21):
    """CI on a ratio sum(num)/sum(den), resampling MEETINGS.

    per_cell: (meeting_id, numerator, denominator). Six clusters — this interval is
    descriptive, not a significance test.
    """
    import numpy as np
    groups = defaultdict(list)
    for i, (m, _, _) in enumerate(per_cell):
        groups[m].append(i)
    keys = sorted(groups)
    num = np.array([x[1] for x in per_cell], dtype=float)
    den = np.array([x[2] for x in per_cell], dtype=float)
    point = num.sum() / den.sum() if den.sum() else float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        dd = den[idx].sum()
        vals[i] = num[idx].sum() / dd if dd else np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    per_meeting = {}
    loo = {}
    for k in keys:
        idx = groups[k]
        dd = sum(den[i] for i in idx)
        per_meeting[k] = {"num": int(sum(num[i] for i in idx)), "den": int(dd),
                          "value": (sum(num[i] for i in idx) / dd) if dd else None}
        rest = [i for kk in keys if kk != k for i in groups[kk]]
        dd2 = sum(den[i] for i in rest)
        loo[k] = (sum(num[i] for i in rest) / dd2) if dd2 else None
    return {"point": point, "ci95_meeting_cluster": [float(lo), float(hi)],
            "n_clusters": len(keys), "per_meeting": per_meeting,
            "leave_one_meeting_out": loo,
            "num": int(num.sum()), "den": int(den.sum())}


def paired_delta(a: list[tuple[str, int, int]], b: list[tuple[str, int, int]],
                 n_boot: int = N_BOOT, seed: int = 21):
    """CI on WER(a) - WER(b) with the SAME reference, resampling meetings.

    Six clusters: descriptive, not a test. Reported because a paired contrast on
    identical audio carries information that two overlapping marginal CIs do not.
    """
    import numpy as np
    assert len(a) == len(b) and all(x[0] == y[0] and x[2] == y[2] for x, y in zip(a, b)), \
        "the pairing is broken - different cells or different denominators"
    groups = defaultdict(list)
    for i, (m, _, _) in enumerate(a):
        groups[m].append(i)
    keys = sorted(groups)
    na = np.array([x[1] for x in a], float)
    nb = np.array([x[1] for x in b], float)
    den = np.array([x[2] for x in a], float)
    point = (na.sum() - nb.sum()) / den.sum() if den.sum() else float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        dd = den[idx].sum()
        vals[i] = (na[idx].sum() - nb[idx].sum()) / dd if dd else np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    loo = {}
    for k in keys:
        rest = [i for kk in keys if kk != k for i in groups[kk]]
        dd = sum(den[i] for i in rest)
        loo[k] = (sum(na[i] for i in rest) - sum(nb[i] for i in rest)) / dd if dd else None
    return {"delta": point, "ci95_meeting_cluster": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "leave_one_meeting_out": loo,
            "n_clusters": len(keys)}


def wilson(k: int, n: int, z: float = 1.96):
    """Naive token-level interval. Labelled naive on purpose: tokens inside a cell,
    a speaker and a meeting are not independent Bernoulli trials."""
    if n == 0 or not (0 <= k <= n):
        # not a proportion (an error count can exceed the reference length), so a
        # binomial interval is meaningless here
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [c - h, c + h]


# ------------------------------------------------------------------------- data
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load():
    man = json.loads((GOLD / "manifest.json").read_text())
    a_sha = sha(GOLD / "answers-user-2026-08-16.json")
    c_sha = sha(GOLD / "cells-frozen.json")
    assert a_sha == EXPECT_ANSWERS_SHA == man["answers_sha256"], "answers hash moved"
    assert c_sha == EXPECT_CELLS_SHA == man["cells_sha256"], "cells hash moved"
    ans = json.loads((GOLD / "answers-user-2026-08-16.json").read_text())
    cells = {c["id"]: c for c in json.loads((GOLD / "cells-frozen.json").read_text())["cells"]}
    with open(ROOT / "research/gold-set-2026-08/selection.csv", newline="", encoding="utf-8") as fh:
        sel = {r["cell_id"]: r for r in csv.DictReader(fh)}
    return man, ans, cells, sel


MEET_FILE = {
    ("chalandri", "jul29_2026"): "chalandri__jul29_2026.json",
    ("chania", "jun24_2026"): "chania__jun24_2026.json",
    ("samothraki", "jul6_2026"): "samothraki__jul6_2026.json",
    ("sparta", "jul1_2026"): "sparta__jul1_2026.json",
    ("xylokastro", "jun29_2026"): "xylokastro__jun29_2026.json",
    ("zografou", "jun18_2026"): "zografou__jun18_2026.json",
}
_pub_cache: dict[str, list] = {}


def pub_utterances(city: str, meeting: str) -> list[dict]:
    """Flat list of published utterances with absolute times and a speaker key."""
    key = f"{city}/{meeting}"
    if key in _pub_cache:
        return _pub_cache[key]
    f = MEET / MEET_FILE[(city, meeting)]
    d = json.loads(f.read_text())
    out = []
    for seg in d["transcript"]:
        tag = (seg.get("speakerTag") or {})
        spk = tag.get("personId") or tag.get("label") or seg.get("speakerTagId") or "?"
        for u in seg["utterances"]:
            out.append({"s": float(u["startTimestamp"]), "e": float(u["endTimestamp"]),
                        "text": u["text"], "spk": str(spk),
                        "uncertain": bool(u.get("uncertain"))})
    out.sort(key=lambda x: x["s"])
    _pub_cache[key] = out
    return out


# ---------------------------------------------------------------- per-cell views
REGIONS = ("core_strict", "core_envelope", "clip")


def region_blocks(blocks: list[dict], region: str) -> list[dict]:
    if region == "clip":
        return list(blocks)
    if region == "core_strict":
        return [b for b in blocks if b["s"] >= CORE[0] and b["e"] <= CORE[1]]
    return [b for b in blocks if b["e"] > CORE[0] and b["s"] < CORE[1]]


def region_span(blocks: list[dict], region: str) -> tuple[float, float] | None:
    """Hull of the blocks the region includes.

    Deliberately NOT the nominal clip or core interval. The human's blocks do not
    tile the whole 35 s clip, so a nominal span makes every system word spoken
    outside the annotated stretch a free insertion — measured, and it doubled
    every WER on this set before it was caught.
    """
    if not blocks:
        return None
    return (min(b["s"] for b in blocks), max(b["e"] for b in blocks))


def masked_intervals(blocks: list[dict], span: tuple[float, float],
                     also_cut: list[dict] | None = None) -> list[tuple[float, float]]:
    """`span` minus the time of every block whose text is not in the reference.

    Two reasons a block is cut: the human marked it `text_unc`, or the region does
    not include it. Without this the hypothesis words spoken over such a block come
    back as insertions and the reference is punished for the human's honesty. Named
    in the Codex plan review (job 75bfc337) as a required repair.
    """
    cuts = sorted([(b["s"], b["e"]) for b in blocks if b.get("text_unc")]
                  + [(b["s"], b["e"]) for b in (also_cut or [])])
    out, cur = [], span[0]
    for s, e in cuts:
        s, e = max(s, span[0]), min(e, span[1])
        if e <= cur:
            continue
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < span[1]:
        out.append((cur, span[1]))
    return out


def in_intervals(t: float, ivs: list[tuple[float, float]]) -> bool:
    return any(a <= t < b for a, b in ivs)


def gold_view(cell_ans: dict, region: str, drop_uncertain: bool = True):
    """(tokens in time order, tokens by speaker, overlap-exposed token list)."""
    blocks = sorted(region_blocks(cell_ans["b"]["blocks"], region), key=lambda b: (b["s"], b["id"]))
    kept = [b for b in blocks if not (drop_uncertain and b.get("text_unc"))]
    seq, by_spk, ov = [], defaultdict(list), []
    for b in kept:
        t = wtoks(b["text"])
        seq += t
        by_spk[b["spk"]] += t
        if b.get("ov_with"):
            ov += t
    return seq, dict(by_spk), ov, blocks, kept


def adp_tokens(cid: str, ivs):
    p = HYP / "adapter" / f"{cid}.json"
    d = json.loads(p.read_text())
    words = []
    for u in d["transcription"]["utterances"]:
        for w in u["words"]:
            words.append((float(w["start"]), w["word"]))
    words.sort()
    return [t for s, w in words if in_intervals(s, ivs) for t in wtoks(w)]


def turbo_view(cell: dict, ivs):
    seq, by_spk = [], defaultdict(list)
    for w in cell["words"]:
        if in_intervals(float(w["s"]), ivs):
            t = wtoks(w["t"])
            seq += t
            by_spk[w["spk"]] += t
    return seq, dict(by_spk)


def pub_view(city, meeting, clip_start: float, ivs):
    """Published utterances whose MIDPOINT falls in a scored interval."""
    us = pub_utterances(city, meeting)
    seq, by_spk, picked = [], defaultdict(list), []
    for u in us:
        s, e = u["s"] - clip_start, u["e"] - clip_start
        mid = (s + e) / 2.0
        if in_intervals(mid, ivs):
            t = wtoks(u["text"])
            seq += t
            by_spk[u["spk"]] += t
            picked.append({"s": s, "e": e, "spk": u["spk"], "n": len(t)})
    return seq, dict(by_spk), picked


def snx_tokens(cid: str):
    p = HYP / "soniox" / f"{cid}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return wtoks(d.get("text", "")) if d.get("ok") else None


# ------------------------------------------------------------------------ main
def main():
    man, ans, cells, sel = load()
    scored = man["scored_cell_ids"]
    res = {
        "protocol": "gold-set-2026-08-16",
        "answers_sha256": man["answers_sha256"],
        "cells_sha256": man["cells_sha256"],
        "n_cells": len(scored),
        "n_meetings": len({sel[c]["meeting_id"] for c in scored}),
        "regions": {},
    }

    # ---------------------------------------------------- census (region-free)
    census = {"blocks": 0, "blocks_text_unc": 0, "blocks_spk_unc": 0,
              "blocks_overlap": 0, "tokens_by_region": {}, "tokens_uncertain_by_region": {}}
    for cid in scored:
        for b in ans[cid]["b"]["blocks"]:
            census["blocks"] += 1
            census["blocks_text_unc"] += bool(b.get("text_unc"))
            census["blocks_spk_unc"] += bool(b.get("spk_unc"))
            census["blocks_overlap"] += bool(b.get("ov_with"))
    for region in REGIONS:
        n = nu = 0
        for cid in scored:
            for b in region_blocks(ans[cid]["b"]["blocks"], region):
                k = len(wtoks(b["text"]))
                n += k
                nu += k if b.get("text_unc") else 0
        census["tokens_by_region"][region] = n
        census["tokens_uncertain_by_region"][region] = nu
    res["census"] = census

    for region in REGIONS:
        res["regions"][region] = score_region(scored, ans, cells, sel, region)

    res["omission_detector"] = omission_detector(scored, ans)
    res["blind_sentinel"] = blind_sentinel(scored, ans, cells)
    res["pipeline_coverage"] = pipeline_coverage(scored, ans, sel)
    res["overlap_blocks"] = overlap_blocks(scored, ans, sel)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")
    return res


def score_region(scored, ans, cells, sel, region):
    """WER / cpWER / speaker accuracy / candidate audit inside one region."""
    per = {k: [] for k in ("adp_fid", "pub_fid", "turbo_fid", "adp_agree",
                           "adp_cp", "pub_cp", "turbo_cp",
                           "pub_spk_cond", "pub_spk_recall",
                           "pub_spk_cond_ov", "pub_spk_recall_ov",
                           "cand_support", "adp_miss_pub", "adp_miss_snx",
                           "adp_miss_any", "adp_del", "pub_del", "turbo_del",
                           "pub_ins", "adp_ins")}
    detail = []
    cand_bucket = Counter()
    for cid in scored:
        row = sel[cid]
        mid = row["meeting_id"]
        clip_start = float(row["clip_start"])
        seq, by_spk, ov, blocks, kept = gold_view(ans[cid], region)
        span = region_span(blocks, region)
        if span is None or not seq:
            continue
        outside = [b for b in ans[cid]["b"]["blocks"]
                   if b["id"] not in {x["id"] for x in blocks}]
        ivs = masked_intervals(blocks, span, outside)
        adp = adp_tokens(cid, ivs)
        tseq, tby = turbo_view(cells[cid], ivs)
        pseq, pby, _ = pub_view(row["city_id"], mid, clip_start, ivs)
        snx = snx_tokens(cid)

        # --- fidelity-to-audio, serialised by start time
        a = sdi_counts(seq, adp)
        p = sdi_counts(seq, pseq)
        t = sdi_counts(seq, tseq)
        per["adp_fid"].append((mid, a["err"], a["N"]))
        per["pub_fid"].append((mid, p["err"], p["N"]))
        per["turbo_fid"].append((mid, t["err"], t["N"]))
        per["adp_del"].append((mid, a["D"], a["N"]))
        per["pub_del"].append((mid, p["D"], p["N"]))
        per["turbo_del"].append((mid, t["D"], t["N"]))
        per["pub_ins"].append((mid, p["I"], p["N"]))
        per["adp_ins"].append((mid, a["I"], a["N"]))
        # --- agreement-with-OpenCouncil on the SAME audio (paired gap)
        ag = sdi_counts(pseq, adp)
        per["adp_agree"].append((mid, ag["err"], ag["N"]))

        # --- cpWER
        cp_a = cp_wer(by_spk, {"_": adp})
        cp_p = cp_wer(by_spk, pby or {"_": []})
        cp_t = cp_wer(by_spk, tby or {"_": []})
        per["adp_cp"].append((mid, cp_a["err"], cp_a["N"]))
        per["pub_cp"].append((mid, cp_p["err"], cp_p["N"]))
        per["turbo_cp"].append((mid, cp_t["err"], cp_t["N"]))

        # --- speaker attribution for PUB, permutation LOCKED once on the cell
        sp = speaker_accuracy(ans[cid], region, pby, cp_p["mapping"])
        per["pub_spk_cond"].append((mid, sp["correct_attr"], max(sp["correct_recog"], 0)))
        per["pub_spk_recall"].append((mid, sp["correct_attr"], sp["n_certain"]))
        per["pub_spk_cond_ov"].append((mid, sp["correct_attr_ov"], max(sp["correct_recog_ov"], 0)))
        per["pub_spk_recall_ov"].append((mid, sp["correct_attr_ov"], sp["n_certain_ov"]))

        # --- candidate audit: built from ADP vs PUB only, gold consulted after
        cand = [pseq[j] for op, _, j in align_ops(adp, pseq) if op == "I"]
        support = multiset_support(cand, seq)
        per["cand_support"].append((mid, support, len(cand)))

        # --- what ADP misses, and who witnesses it
        missed = [seq[i] for op, i, _ in align_ops(seq, adp) if op == "D"]
        w_pub = multiset_flags(missed, pseq)
        w_snx = multiset_flags(missed, snx) if snx is not None else [False] * len(missed)
        per["adp_miss_pub"].append((mid, sum(w_pub), len(missed)))
        per["adp_miss_snx"].append((mid, sum(w_snx), len(missed)))
        per["adp_miss_any"].append((mid, sum(x or y for x, y in zip(w_pub, w_snx)), len(missed)))
        for x, y in zip(w_pub, w_snx):
            cand_bucket[("pub" if x else "") + ("snx" if y else "") or "neither"] += 1

        detail.append({"cell": cid, "meeting": mid, "stream": row["draw_stream"],
                       "n_gold": len(seq), "n_adp": len(adp), "n_pub": len(pseq),
                       "n_turbo": len(tseq), "n_cand": len(cand),
                       "adp_wer": a["err"] / a["N"], "pub_wer": p["err"] / p["N"],
                       "adp_agree_wer": ag["err"] / ag["N"] if ag["N"] else None})

    out = {"n_cells_scored": len(detail)}
    for k, v in per.items():
        if not v:
            continue
        r = cluster_ci(v)
        r["naive_token_ci95"] = wilson(r["num"], r["den"])
        out[k] = r
    out["adp_missed_witness_buckets"] = dict(cand_bucket)
    out["paired"] = {
        "adp_minus_pub_fidelity": paired_delta(per["adp_fid"], per["pub_fid"]),
        "adp_minus_turbo_fidelity": paired_delta(per["adp_fid"], per["turbo_fid"]),
        "pub_minus_turbo_fidelity": paired_delta(per["pub_fid"], per["turbo_fid"]),
    }
    # paired agreement-fidelity gap, per cell
    gaps = [(d["meeting"], d["adp_agree_wer"] - d["adp_wer"]) for d in detail
            if d["adp_agree_wer"] is not None]
    out["paired_agreement_minus_fidelity_gap"] = {
        "per_cell_mean": sum(g for _, g in gaps) / len(gaps) if gaps else None,
        "n_cells": len(gaps),
        "n_cells_agreement_lower": sum(1 for _, g in gaps if g < 0),
    }
    out["cells"] = detail
    return out


def multiset_support(cand: list[str], gold: list[str]) -> int:
    """How many candidate OCCURRENCES the gold can support, one gold token each."""
    pool = Counter(gold)
    n = 0
    for w in cand:
        if pool[w] > 0:
            pool[w] -= 1
            n += 1
    return n


def multiset_flags(items: list[str], witness: list[str] | None) -> list[bool]:
    if witness is None:
        return [False] * len(items)
    pool = Counter(witness)
    out = []
    for w in items:
        ok = pool[w] > 0
        if ok:
            pool[w] -= 1
        out.append(ok)
    return out


def speaker_accuracy(cell_ans, region, pub_by_spk, mapping):
    """Words matched lexically AND attributed to the mapped speaker.

    The gold->hyp speaker mapping comes from the cell-level cpWER over the whole
    certain region and is reused unchanged inside the overlap subset.
    """
    blocks = sorted(region_blocks(cell_ans["b"]["blocks"], region), key=lambda b: (b["s"], b["id"]))
    certain = [b for b in blocks if not b.get("text_unc") and not b.get("spk_unc")]
    by_spk, ov_by_spk = defaultdict(list), defaultdict(list)
    for b in certain:
        t = wtoks(b["text"])
        by_spk[b["spk"]] += t
        if b.get("ov_with"):
            ov_by_spk[b["spk"]] += t
    n_certain = sum(len(v) for v in by_spk.values())
    n_certain_ov = sum(len(v) for v in ov_by_spk.values())
    all_hyp = [w for v in pub_by_spk.values() for w in v]
    all_ref = [w for v in by_spk.values() for w in v]
    recog = sdi_counts(all_ref, all_hyp)["M"]
    ov_ref = [w for v in ov_by_spk.values() for w in v]
    recog_ov = sdi_counts(ov_ref, all_hyp)["M"] if ov_ref else 0

    def attributed(ref_by):
        n = 0
        for g, toks in ref_by.items():
            h = pub_by_spk.get(mapping.get(g), [])
            n += sdi_counts(toks, h)["M"]
        return n

    return {"n_certain": n_certain, "n_certain_ov": n_certain_ov,
            "correct_recog": recog, "correct_recog_ov": recog_ov,
            "correct_attr": attributed(by_spk), "correct_attr_ov": attributed(ov_by_spk)}


def pipeline_coverage(scored, ans, sel):
    """How much of what was said is temporally uncovered by the published text.

    NOT a proof that applyDiarization took its SKIPPING branch: no production log
    is available here. It is observed absence of coverage.
    """
    unc_blocks = unc_sec = 0
    tot_blocks = tot_sec = 0
    per_cell = []
    for cid in scored:
        row = sel[cid]
        cs = float(row["clip_start"])
        us = pub_utterances(row["city_id"], row["meeting_id"])
        rel = [(u["s"] - cs, u["e"] - cs) for u in us]
        c_unc = c_tot = 0
        for b in region_blocks(ans[cid]["b"]["blocks"], "core_envelope"):
            tot_blocks += 1
            dur = b["e"] - b["s"]
            tot_sec += dur
            c_tot += 1
            if not any(e > b["s"] and s < b["e"] for s, e in rel):
                unc_blocks += 1
                unc_sec += dur
                c_unc += 1
        per_cell.append({"cell": cid, "meeting": row["meeting_id"],
                         "blocks": c_tot, "uncovered": c_unc})
    return {"blocks_total": tot_blocks, "blocks_temporally_uncovered": unc_blocks,
            "seconds_total": round(tot_sec, 2), "seconds_uncovered": round(unc_sec, 2),
            "per_cell": per_cell,
            "ci": cluster_ci([(p["meeting"], p["uncovered"], p["blocks"]) for p in per_cell])}


def overlap_blocks(scored, ans, sel):
    """The 'one speaker per utterance' question, at BLOCK level (n = blocks).

    For every overlap-participating gold block: how much of its text survives in
    the published transcript at all, and does the published text put it on the same
    speaker as its overlap partner (i.e. the two simultaneous voices merged).
    """
    rows = []
    for cid in scored:
        row = sel[cid]
        cs = float(row["clip_start"])
        us = pub_utterances(row["city_id"], row["meeting_id"])
        blocks = {b["id"]: b for b in ans[cid]["b"]["blocks"]}
        for b in ans[cid]["b"]["blocks"]:
            if not b.get("ov_with"):
                continue
            if not (b["e"] > CORE[0] and b["s"] < CORE[1]):
                continue
            gt = wtoks(b["text"])
            hit = [u for u in us if (u["e"] - cs) > b["s"] and (u["s"] - cs) < b["e"]]
            htoks = [w for u in hit for w in wtoks(u["text"])]
            matched = multiset_support(gt, htoks)
            spks = {u["spk"] for u in hit}
            partners = [blocks[i] for i in b["ov_with"] if i in blocks]
            same_spk_as_partner = None
            if partners and hit:
                phit = [u for u in us
                        for pb in partners
                        if (u["e"] - cs) > pb["s"] and (u["s"] - cs) < pb["e"]]
                pspks = {u["spk"] for u in phit}
                same_spk_as_partner = bool(spks and pspks and spks == pspks)
            rows.append({"cell": cid, "meeting": row["meeting_id"], "block": b["id"],
                         "n_tokens": len(gt), "matched_in_pub": matched,
                         "pub_speakers_here": len(spks),
                         "merged_with_partner_speaker": same_spk_as_partner})
    n = len(rows)
    return {
        "n_overlap_blocks_in_core": n,
        "blocks_with_zero_tokens_in_pub": sum(1 for r in rows if r["matched_in_pub"] == 0),
        "blocks_merged_onto_partner_speaker": sum(1 for r in rows
                                                  if r["merged_with_partner_speaker"]),
        "blocks_pub_shows_one_speaker": sum(1 for r in rows if r["pub_speakers_here"] == 1),
        "tokens_total": sum(r["n_tokens"] for r in rows),
        "tokens_matched_in_pub": sum(r["matched_in_pub"] for r in rows),
        "token_ci": cluster_ci([(r["meeting"], r["matched_in_pub"], r["n_tokens"])
                                for r in rows if r["n_tokens"]]),
        "rows": rows,
    }


def omission_detector(scored, ans):
    """PPV of the pass-3 flags under the HUMAN verdict.

    Two kinds, and they answer different questions:

    `nowords` — pyannote asserts >0.8 s of speech where the corrected text has no
    words. Same family of signal as the 0.320 precision of
    exp-2026-08-16-pyannote-transcription, but there the ground truth was our own
    published text and here it is a human who listened.

    `altonly` — the PUBLISHED OpenCouncil transcript has words inside the core. A
    verdict of `missing` means the human judged that speech IS there and their own
    corrected reference had not captured it: a direct, if small, bound on how far
    the reference itself under-covers the audio.

    A PPV among flags. Never a recall: nothing here enumerates the omissions the
    flags did not raise.
    """
    by_kind = defaultdict(Counter)
    per_meeting = defaultdict(Counter)
    for cid in scored:
        for f in (ans[cid].get("c") or {}).get("flags", []) or []:
            by_kind[f.get("kind", "?")][f.get("verdict", "unanswered")] += 1
            per_meeting[(f.get("kind", "?"), cid)][f.get("verdict", "unanswered")] += 1
    out = {}
    for kind, c in by_kind.items():
        n = sum(c.values())
        lost = c.get("missing", 0)
        out[kind] = {"n_flags": n, "verdicts": dict(c),
                     "ppv_speech_lost": lost / n if n else None,
                     "naive_token_ci95": wilson(lost, n)}
    return out


def blind_sentinel(scored, ans, cells):
    """The 3 calibration cells written from scratch before the prefill was shown.

    A sentinel, explicitly not a rate: 3 cells, one listener, and the memory of the
    blind pass changes the corrected pass that follows it.
    """
    rows = []
    for cid in scored:
        blank = ans[cid].get("blank")
        if not blank or not blank.get("text"):
            continue
        b = wtoks(blank["text"])
        final = []
        for blk in sorted(ans[cid]["b"]["blocks"], key=lambda x: x["s"]):
            final += wtoks(blk["text"])
        pre = []
        for w in cells[cid]["words"]:
            pre += wtoks(w["t"])
        rows.append({
            "cell": cid, "n_blind": len(b), "n_final": len(final), "n_prefill": len(pre),
            "blind_vs_final": sdi_counts(final, b),
            "prefill_vs_final": sdi_counts(final, pre),
            "blind_tokens_absent_from_prefill": len(b) - multiset_support(b, pre),
        })
    return {"n_cells": len(rows), "rows": rows}


if __name__ == "__main__":
    main()
