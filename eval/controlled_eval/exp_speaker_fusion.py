#!/usr/bin/env python3
"""Round 2 of wayfinder #17: does speaker information help the fusion vote choose?

Round 1 put the question out of reach: the service's default STT (Parakeet-TDT) scored
0.3567 WER on Greek council speech, so the word-level speaker attribution it returned
was attached to text nobody would use. The joint-fusion idea was never actually
tested. Round 2 re-runs the identical call with
`transcriptionConfig.model = faster-whisper-large-v3-turbo` and tests it.

TWO TIMELINES, and the primary one is NOT the exclusive one.
`exclusive: true` makes precision-2 return `diarization` AND `exclusiveDiarization`,
so both rounds already carry both; nothing was re-run to get this. Round 1 looked only
at the exclusive timeline, which was the wrong choice and there was measured evidence
for that at the time: `exp-2026-08-08-exclusive-diarization` found that resolving
overlap to a single speaker DROPS about 10 utterances for every 1 it recovers
(+1.94 per 100 against a +0.5 gate). The exclusive timeline manufactures exactly the
silent loss this project is chasing. So here the plain `diarization` is primary and
the exclusive one is secondary, and their DIFFERENCE is itself a signal: where the
plain timeline has two speakers at once and the exclusive one kept a single speaker,
that is detected overlap.

Overlap is worth conditioning on. It is 2.2% of the audio, but the worst overlap
quartile scores ~10 points worse on all seven systems (`exp-2026-08-03-overlap-screen`)
and the causal test priced it at 0.16 WER points (`exp-2026-08-03-synthetic-overlap`).

Measurements, frozen on issue #17 before any number was seen:

  A1  Are the vote's errors concentrated near speaker handovers? Descriptive.
  A2  THE MAIN ONE. Voting per speaker turn instead of per window, with a placebo
      partition control, because cutting a window into smaller pieces changes the
      vote's choices even when the cuts know nothing about speakers. The quantity of
      interest is the difference of differences. Reported again restricted to cells
      that contain detected overlap - that is the user's actual objective: use the
      text to separate speakers exactly where they talk over each other.
  A3  whisper-turbo as a standalone Greek ASR.
  A4  whisper-turbo as a fourth voter. A null is EXPECTED and was declared in advance.
  A5  Is `exclusiveDiarization` identical between the two rounds?
  A6  The omission detector on BOTH timelines, same time axis, plus an
      overlap-conditioned error breakdown.

Sample: the same 247 windows as round 1. Same scorer, same clustered bootstrap.

Writes results_speaker_fusion.json (aggregates only, no transcript text).

Env: SC (cache dir) N_BOOT (10000) N_PLACEBO (20) N_RANDOM (200)
"""
from __future__ import annotations

import collections
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.controlled_eval import bench_data as B                        # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import rates, sdi        # noqa: E402
from eval.controlled_eval.exp_parakeet_voter import (                   # noqa: E402
    MIN_DEL_RUN, MIN_SPEECH_SEC, align_ops, overlaps, token_times)
from eval.controlled_eval.parakeet_run import RUN_ID, sc, target_items  # noqa: E402
from eval.controlled_eval.scoring import (cluster_bootstrap, edist,     # noqa: E402
                                          head2head, wtoks)

TURBO = "pyannote-whisper-turbo"
BASE = "scribe-v2-clean"
SECOND = "soniox"
THIRD = "oc-runpod-fixed-2026-08-10"
TRIO = [BASE, SECOND, THIRD]
PRIORITY = [BASE, SECOND, THIRD, TURBO]      # frozen tie-break, same shape as round 1

TAU = 1.0            # seconds: a reference token is "near" a handover within this
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
N_PLACEBO = int(os.environ.get("N_PLACEBO", "20"))
N_RANDOM = int(os.environ.get("N_RANDOM", "200"))
OUT = Path(__file__).with_name("results_speaker_fusion.json")


def log(m):
    print(m, flush=True)


# ------------------------------------------------------- timeline -> active speakers
def active_intervals(segs: list[dict]) -> list[tuple[float, float, frozenset]]:
    """Sweep the segment list into maximal half-open intervals with a constant
    speaker SET.

    Written once and used for both timelines. On `exclusiveDiarization` every set has
    at most one member by construction, so this reduces to the plain segment list; on
    `diarization` it exposes the simultaneous-speaker intervals that the exclusive
    timeline has already collapsed away.

    Per-speaker REFERENCE COUNTS, not a set of labels. A set gets this wrong in ways
    that show up in real diarizer output: `A:[0,1]` followed by `A:[1,2]` would add
    and remove A at t=1 and lose the second segment entirely, and two overlapping
    `A` segments would end at the first `end`. Adjacent intervals carrying the same
    set are merged, because the docstring promises maximal intervals and because an
    unmerged redundant event would become a false placebo boundary.
    """
    delta: dict[float, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for s in segs:
        a, b = float(s["start"]), float(s["end"])
        if not (b > a):                       # zero-length or inverted: ignore
            continue
        delta[a][s["speaker"]] += 1
        delta[b][s["speaker"]] -= 1
    times = sorted(delta)
    live: collections.Counter = collections.Counter()
    raw = []
    for t, nxt in zip(times, times[1:]):
        live.update(delta[t])
        live += collections.Counter()         # drop non-positive counts
        if live and nxt > t:
            raw.append((t, nxt, frozenset(live)))
    out: list[tuple[float, float, frozenset]] = []
    for iv in raw:
        if out and out[-1][2] == iv[2] and out[-1][1] == iv[0]:
            out[-1] = (out[-1][0], iv[1], iv[2])
        else:
            out.append(iv)
    return out


def handover_cuts(segs: list[dict]) -> tuple[list[float], list[tuple[float, float]]]:
    """Times where the floor passes from one set of speakers to a disjoint one.

    Returns (cut times, the transitional spans they came from).

    The first version of this only accepted a DIRECT change to a disjoint set, and
    that misses the canonical conversational handover, which is overlap-mediated:

        {A} -> {A,B} -> {B}

    Neither of those two steps is disjoint, yet the floor plainly passes from A to B,
    and that transition is precisely the one this experiment cares about. So the rule
    walks forward from a stable set L until it reaches an interval sharing nobody with
    L, and places ONE cut in the middle of the span between the last interval that
    still contained someone from L and the first that contains none of them.

    On the exclusive timeline, where sets are singletons and never overlap, this
    degenerates to "the speaker label changed", so the two timelines stay comparable.
    """
    iv = active_intervals(segs)
    if not iv:
        return [], []
    cuts, spans = [], []
    L = iv[0][2]
    last_with_L = 0
    for j in range(1, len(iv)):
        if iv[j][2] & L:
            # someone who held the floor is still holding it: same turn continues.
            # L deliberately does NOT absorb the newcomer - if it did, {A}->{A,B}->{B}
            # would never register a handover, which is the exact case that matters.
            last_with_L = j
            continue
        span = (iv[last_with_L][1], iv[j][0])
        cuts.append((span[0] + span[1]) / 2)
        spans.append(span)
        L = iv[j][2]
        last_with_L = j
    return sorted(set(cuts)), spans


def other_boundary_times(segs: list[dict],
                         spans: list[tuple[float, float]]) -> list[float]:
    """The placebo pool: active-set change times that are NOT part of a handover.

    Overlap onsets and offsets that merely decorate a real handover are excluded via
    `spans`, because a cut inside a handover transition is speaker-informed and would
    make the control leak the very signal it is supposed to lack.
    """
    iv = active_intervals(segs)
    out = []
    for a, b in zip(iv, iv[1:]):
        t = (a[1] + b[0]) / 2
        if any(lo <= t <= hi for lo, hi in spans):
            continue
        out.append(t)
    return sorted(set(out))


def overlap_intervals(segs: list[dict]) -> list[tuple[float, float]]:
    """Intervals where two or more speakers are active at once."""
    return [(s, e) for s, e, sp in active_intervals(segs) if len(sp) >= 2]


def in_any(t: float, ivs: list[tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in ivs)


# ------------------------------------------------------------------ partitioning
def cells(cuts: list[float]) -> list[tuple[float, float]]:
    """Half-open cells tiling the whole timeline, so every token lands in exactly one.

    Deliberately NOT the diarization segments themselves: those leave gaps and a tail,
    and tokens falling in a gap would silently vanish, manufacturing deletions.
    """
    edges = [float("-inf")] + list(cuts) + [float("inf")]
    return list(zip(edges, edges[1:]))


def split_tokens(tokens: list[str], times: list[float],
                 cs: list[tuple[float, float]]) -> list[list[str]]:
    """Assign tokens to cells. Times are non-decreasing, so cells are index ranges."""
    out = [[] for _ in cs]
    j = 0
    for tok, t in zip(tokens, times):
        while j + 1 < len(cs) and t >= cs[j][1]:
            j += 1
        out[j].append(tok)
    return out


def pick_by_similarity(cands: dict[str, list[str]], order: list[str]) -> str:
    """Summed pairwise difflib similarity, argmax, first-wins on an exact tie.

    `order` is the frozen priority list, and max() returns the first maximal element,
    so passing it in that order IS the declared tie-break.
    """
    enc: dict = {}
    coded = {}
    for p in order:
        s = []
        for w in cands[p]:
            if w not in enc:
                enc[w] = chr(0x3000 + len(enc))
            s.append(enc[w])
        coded[p] = "".join(s)
    return max(order, key=lambda p: sum(
        SequenceMatcher(None, coded[p], coded[q], autojunk=False).ratio()
        for q in order if q != p))


def partition_vote(cand_toks, cand_times, cuts, order) -> tuple[list[str], dict]:
    """Vote inside each cell and concatenate the winners."""
    cs = cells(cuts)
    per = {p: split_tokens(cand_toks[p], cand_times[p], cs) for p in order}
    out, info = [], {"cells": len(cs), "disagree": 0, "picks": {}}
    for i in range(len(cs)):
        cands = {p: per[p][i] for p in order}
        if len({tuple(v) for v in cands.values()}) > 1:
            info["disagree"] += 1
        w = pick_by_similarity(cands, order)
        info["picks"][w] = info["picks"].get(w, 0) + 1
        out.extend(cands[w])
    return out, info


def greedy_local_oracle(ref_toks, ref_times, cand_toks, cand_times, cuts, order):
    """Best system per cell by local edit count against the reference slice.

    NOT a ceiling and never called one: whole-window Levenshtein can align across cell
    boundaries, so this greedy local choice is not guaranteed to minimise window WER.
    """
    cs = cells(cuts)
    per = {p: split_tokens(cand_toks[p], cand_times[p], cs) for p in order}
    rs = split_tokens(ref_toks, ref_times, cs)
    out = []
    for i in range(len(cs)):
        best = min(order, key=lambda p: edist(rs[i], per[p][i]))
        out.extend(per[best][i])
    return out


# ------------------------------------------------------------------ omission detector
def detector(segs, our_times, ref_times, ref_ops, rng) -> dict:
    """Flag long speech with none of our words; score against real deletion runs.

    Identical rule to round 1, but callable on either timeline and with the time axis
    supplied by the caller, so the two timelines are compared under one axis.
    """
    iv = active_intervals(segs)
    eligible = [(s, e) for s, e, _ in iv if e - s >= MIN_SPEECH_SEC]
    flags = [(s, e) for s, e in eligible
             if not any(s <= t <= e for t in our_times)]

    runs, i = [], 0
    while i < len(ref_ops):
        if ref_ops[i] == "D":
            j = i
            while j < len(ref_ops) and ref_ops[j] == "D":
                j += 1
            if j - i >= MIN_DEL_RUN:
                runs.append((ref_times[i], ref_times[j - 1]))
            i = j
        else:
            i += 1

    tp = sum(1 for f in flags if any(overlaps(f, r) for r in runs))
    hit = sum(1 for r in runs if any(overlaps(f, r) for f in flags))
    k = len(flags)
    rtp = rhit = 0
    for _ in range(N_RANDOM):
        if k and eligible:
            idx = rng.choice(len(eligible), size=min(k, len(eligible)), replace=False)
            rm = [eligible[i] for i in idx]
        else:
            rm = []
        rtp += sum(1 for f in rm if any(overlaps(f, r) for r in runs))
        rhit += sum(1 for r in runs if any(overlaps(f, r) for f in rm))
    return {"n_flags": k, "n_runs": len(runs), "n_eligible": len(eligible),
            "tp": tp, "hit": hit, "rtp": rtp / N_RANDOM, "rhit": rhit / N_RANDOM,
            "flag_sec": sum(e - s for s, e in flags)}


# ----------------------------------------------------------------------------- main
def main() -> None:
    import numpy as np

    wt_dir = sc() / "whisper_turbo"
    pk_dir = sc() / "parakeet"
    items = target_items()
    have = {p.stem for p in wt_dir.glob("*.json")}
    missing = [it["item_id"] for it in items if it["item_id"] not in have]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(items)} preregistered windows have no "
            f"whisper-turbo response, e.g. {missing[:3]}")
    log(f"{len(items)} windows, all with a whisper-turbo response")

    wt = {}
    for it in items:
        d = json.loads((wt_dir / f"{it['item_id']}.json")
                       .read_text(encoding="utf-8"))["output"]
        wt[it["item_id"]] = d
        it["hyp"][TURBO] = " ".join(w["text"] for w in d["wordLevelTranscription"])

    clusters = [it["cluster"] for it in items]
    quartet = list(PRIORITY)

    # ------------------------------------------------------------------ A5 first
    a5 = {"compared": 0, "identical": 0, "different": []}
    for it in items:
        p = pk_dir / f"{it['item_id']}.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text(encoding="utf-8"))["output"]["exclusiveDiarization"]
        b = wt[it["item_id"]]["exclusiveDiarization"]
        a5["compared"] += 1
        if a == b:
            a5["identical"] += 1
        else:
            a5["different"].append({
                "window": it["item_id"], "n_round1": len(a), "n_round2": len(b),
                "speech_sec_round1": round(sum(x["end"] - x["start"] for x in a), 2),
                "speech_sec_round2": round(sum(x["end"] - x["start"] for x in b), 2)})
    log(f"A5: {a5['identical']}/{a5['compared']} exclusiveDiarization identical")

    # --------------------------------------------------------------- single systems
    per_arm: dict[str, list] = {}
    for p in B.provider_ids(B.load_report(RUN_ID)) + [TURBO]:
        per_arm[f"single:{p}"] = [sdi(it["ref"], it["hyp"][p]) for it in items]

    # ------------------------------------------------- per-window token timelines
    diag = {p: {"anchors": 0, "tokens": 0, "outside_regular": 0,
                "zero_or_one_anchor_windows": 0} for p in quartet + ["REF"]}
    win = []
    for it in items:
        d = wt[it["item_id"]]
        pk_tokens, pk_times = [], []
        for w in d["wordLevelTranscription"]:
            tt = wtoks(w["text"])
            if not tt:
                continue
            mid = (float(w["start"]) + float(w["end"])) / 2
            for t in tt:
                pk_tokens.append(t)
                pk_times.append(mid)
        reg, exc = d["diarization"], d["exclusiveDiarization"]
        audio_end = max([float(s["end"]) for s in reg + exc]
                        + ([pk_times[-1]] if pk_times else []) + [0.0])
        cov = [(s, e) for s, e, _ in active_intervals(reg)]

        toks, times = {}, {}
        for p in quartet + ["REF"]:
            src = it["ref"] if p == "REF" else it["hyp"][p]
            tk = wtoks(src)
            tm, na = token_times(tk, pk_tokens, pk_times, audio_end)
            toks[p], times[p] = tk, tm
            diag[p]["anchors"] += na
            diag[p]["tokens"] += len(tk)
            if na <= 1:
                diag[p]["zero_or_one_anchor_windows"] += 1
            diag[p]["outside_regular"] += sum(1 for t in tm if not in_any(t, cov))
        win.append({"item": it, "reg": reg, "exc": exc, "toks": toks, "times": times,
                    "audio_end": audio_end,
                    "cuts": handover_cuts(reg),
                    "placebo_pool": other_boundary_times(reg),
                    "cuts_exc": handover_cuts(exc),
                    "ov": overlap_intervals(reg)})

    # ------------------------------------------------------- reconstruction invariant
    bad = []
    for w in win:
        cs = cells(w["cuts"])
        for p in quartet:
            flat = [t for part in split_tokens(w["toks"][p], w["times"][p], cs)
                    for t in part]
            if flat != w["toks"][p]:
                bad.append((w["item"]["item_id"], p))
    if bad:
        raise SystemExit(f"partition loses or reorders tokens in {len(bad)} cases, "
                         f"e.g. {bad[:3]} - A2 is not reportable")
    log("A2 invariant ok: every partition reconstructs each system exactly")

    # ------------------------------------------------------------------ arms
    rng = np.random.default_rng(7)
    arm_rows: dict[str, list] = {k: [] for k in (
        "vote3_window", "vote3_turn", "vote3_placebo", "vote3_turn_exclusive",
        "vote4_window", "greedy_turn_oracle")}
    a2 = {"turn_cells": 0, "disagree_cells": 0, "changed_choice_cells": 0,
          "changed_win": 0, "changed_tie": 0, "changed_loss": 0,
          "disagree_ref_tokens": 0, "total_ref_tokens": 0,
          "placebo_topped_up": 0, "turn_picks": {}, "window_picks": {},
          "overlap_cells": 0, "overlap_disagree_cells": 0,
          "overlap_changed_choice_cells": 0, "overlap_changed_win": 0,
          "overlap_changed_tie": 0, "overlap_changed_loss": 0}

    for w in win:
        it = w["item"]
        ref = it["ref"]
        ct = {p: w["toks"][p] for p in quartet}
        cm = {p: w["times"][p] for p in quartet}

        h_win, iw = partition_vote(ct, cm, [], TRIO)
        h_turn, itn = partition_vote(ct, cm, w["cuts"], TRIO)
        h_turn_x, _ = partition_vote(ct, cm, w["cuts_exc"], TRIO)
        arm_rows["vote3_window"].append(sdi(ref, " ".join(h_win)))
        arm_rows["vote3_turn"].append(sdi(ref, " ".join(h_turn)))
        arm_rows["vote3_turn_exclusive"].append(sdi(ref, " ".join(h_turn_x)))
        for k, v in itn["picks"].items():
            a2["turn_picks"][k] = a2["turn_picks"].get(k, 0) + v
        for k, v in iw["picks"].items():
            a2["window_picks"][k] = a2["window_picks"].get(k, 0) + v
        a2["turn_cells"] += itn["cells"]
        a2["total_ref_tokens"] += len(w["toks"]["REF"])

        h4, _ = partition_vote(ct, cm, [], quartet)
        arm_rows["vote4_window"].append(sdi(ref, " ".join(h4)))
        arm_rows["greedy_turn_oracle"].append(sdi(ref, " ".join(greedy_local_oracle(
            w["toks"]["REF"], w["times"]["REF"], ct, cm, w["cuts"], TRIO))))

        k = len(w["cuts"])
        pool = w["placebo_pool"]
        acc = []
        for _ in range(N_PLACEBO):
            if k == 0:
                pc = []
            elif len(pool) >= k:
                pc = sorted(float(x) for x in rng.choice(pool, size=k, replace=False))
            else:
                a2["placebo_topped_up"] += 1
                extra = rng.uniform(0, w["audio_end"], size=k - len(pool))
                pc = sorted([float(x) for x in pool] + [float(x) for x in extra])
            hp, _ = partition_vote(ct, cm, pc, TRIO)
            acc.append(sdi(ref, " ".join(hp)))
        arm_rows["vote3_placebo"].append(
            tuple(sum(a[i] for a in acc) / len(acc) for i in range(4)))

        # per-cell bookkeeping on the speaker partition, overall and inside overlap
        cs = cells(w["cuts"])
        per = {p: split_tokens(ct[p], cm[p], cs) for p in TRIO}
        rslice = split_tokens(w["toks"]["REF"], w["times"]["REF"], cs)
        win_pick = pick_by_similarity({p: ct[p] for p in TRIO}, TRIO)
        for i, (lo, hi) in enumerate(cs):
            cands = {p: per[p][i] for p in TRIO}
            has_ov = any(overlaps((max(lo, -1e9), min(hi, 1e9)), o) for o in w["ov"])
            if has_ov:
                a2["overlap_cells"] += 1
            if len({tuple(v) for v in cands.values()}) <= 1:
                continue
            a2["disagree_cells"] += 1
            a2["disagree_ref_tokens"] += len(rslice[i])
            if has_ov:
                a2["overlap_disagree_cells"] += 1
            tp = pick_by_similarity(cands, TRIO)
            if tp == win_pick:
                continue
            ca, cb = edist(rslice[i], cands[tp]), edist(rslice[i], cands[win_pick])
            key = "win" if ca < cb else ("tie" if ca == cb else "loss")
            a2["changed_choice_cells"] += 1
            a2[f"changed_{key}"] += 1
            if has_ov:
                a2["overlap_changed_choice_cells"] += 1
                a2[f"overlap_changed_{key}"] += 1

    # ------------------------------------------------------------------------- A1
    bnd = dict.fromkeys(("near_S", "near_D", "near_N", "far_S", "far_D", "far_N"), 0)
    ov_err = dict.fromkeys(("ov_S", "ov_D", "ov_N", "no_S", "no_D", "no_N"), 0)
    near_counts, far_counts = [], []
    ovc, noc = [], []
    for w in win:
        h_win, _ = partition_vote({p: w["toks"][p] for p in TRIO},
                                  {p: w["times"][p] for p in TRIO}, [], TRIO)
        ops = align_ops(w["toks"]["REF"], h_win)
        cuts, ov = w["cuts"], w["ov"]
        c = dict.fromkeys(bnd, 0)
        o = dict.fromkeys(ov_err, 0)
        for t, op in zip(w["times"]["REF"], ops):
            pre = "near" if any(abs(t - x) <= TAU for x in cuts) else "far"
            c[f"{pre}_N"] += 1
            if op in ("S", "D"):
                c[f"{pre}_{op}"] += 1
            pre2 = "ov" if in_any(t, ov) else "no"
            o[f"{pre2}_N"] += 1
            if op in ("S", "D"):
                o[f"{pre2}_{op}"] += 1
        for k in bnd:
            bnd[k] += c[k]
        for k in ov_err:
            ov_err[k] += o[k]
        near_counts.append((c["near_D"], c["near_N"]))
        far_counts.append((c["far_D"], c["far_N"]))
        ovc.append((o["ov_D"], o["ov_N"]))
        noc.append((o["no_D"], o["no_N"]))

    # ------------------------------------------------------------------------- A6
    det = {"regular": [], "exclusive": []}
    for w in win:
        ops = align_ops(w["toks"]["REF"], w["toks"][THIRD])
        for name, segs in (("regular", w["reg"]), ("exclusive", w["exc"])):
            det[name].append(detector(segs, w["times"][THIRD], w["times"]["REF"],
                                      ops, rng))

    def det_summary(rows):
        F = sum(r["n_flags"] for r in rows)
        R = sum(r["n_runs"] for r in rows)
        return {"n_flags": F, "n_deletion_runs": R,
                "n_eligible_segments": sum(r["n_eligible"] for r in rows),
                "flagged_seconds_total": round(sum(r["flag_sec"] for r in rows), 1),
                "precision": sum(r["tp"] for r in rows) / F if F else None,
                "recall": sum(r["hit"] for r in rows) / R if R else None,
                "random_matched_segments": {
                    "precision": sum(r["rtp"] for r in rows) / F if F else None,
                    "recall": sum(r["rhit"] for r in rows) / R if R else None}}

    # -------------------------------------------------------------------- assemble
    res = {
        "run_id": RUN_ID, "round": 2, "stt": "faster-whisper-large-v3-turbo",
        "n_items": len(items), "n_meetings": len(set(clusters)),
        "sample_note": "same 247 windows as round 1 (253 common minus 6 sealed)",
        "scorer": "eval/controlled_eval/scoring.py (not the benchmark app's)",
        "primary_timeline": "diarization (not exclusiveDiarization)",
        "tau_sec": TAU, "n_placebo_draws": N_PLACEBO,
        "tie_break_priority": PRIORITY,
        "A5_exclusive_diarization": a5,
        "arms": {}, "contrasts": {}, "A1_boundary": {}, "A2_info": a2,
        "A6_detector": {k: det_summary(v) for k, v in det.items()},
        "overlap": {}, "anchor_diagnostics": diag, "influence": {},
    }
    for k, v in list(per_arm.items()) + list(arm_rows.items()):
        res["arms"][k] = rates(v)

    def contrast(a_rows, b_rows, label):
        out = {}
        for metric, idx in (("wer", None), ("del_rate", 1), ("ins_rate", 2),
                            ("sub_rate", 0)):
            if metric == "wer":
                ca = [(r[0] + r[1] + r[2], r[3]) for r in a_rows]
                cb = [(r[0] + r[1] + r[2], r[3]) for r in b_rows]
            else:
                ca = [(r[idx], r[3]) for r in a_rows]
                cb = [(r[idx], r[3]) for r in b_rows]
            out[metric] = cluster_bootstrap(ca, cb, clusters, n_boot=N_BOOT)
        out["head2head_wer"] = head2head(
            [(r[0] + r[1] + r[2], r[3]) for r in a_rows],
            [(r[0] + r[1] + r[2], r[3]) for r in b_rows])
        res["contrasts"][label] = out
        return out

    contrast(arm_rows["vote3_turn"], arm_rows["vote3_window"], "turn vs window")
    contrast(arm_rows["vote3_placebo"], arm_rows["vote3_window"], "placebo vs window")
    contrast(arm_rows["vote3_turn"], arm_rows["vote3_placebo"], "turn vs placebo")
    contrast(arm_rows["vote3_turn_exclusive"], arm_rows["vote3_turn"],
             "turn(exclusive) vs turn(regular)")
    contrast(arm_rows["vote4_window"], arm_rows["vote3_window"], "vote4 vs vote3")
    contrast(per_arm[f"single:{TURBO}"], per_arm[f"single:{THIRD}"],
             "whisper-turbo vs ours")
    contrast(per_arm[f"single:{TURBO}"],
             per_arm["single:hf-openai-whisper-large-v3"],
             "whisper-turbo vs whisper-large-v3")

    res["A1_boundary"] = {
        "tau_sec": TAU,
        "near_ref_tokens": bnd["near_N"], "far_ref_tokens": bnd["far_N"],
        "near_share_of_ref_tokens": bnd["near_N"] / max(bnd["near_N"] + bnd["far_N"], 1),
        "near_del_rate": bnd["near_D"] / max(bnd["near_N"], 1),
        "far_del_rate": bnd["far_D"] / max(bnd["far_N"], 1),
        "near_sub_rate": bnd["near_S"] / max(bnd["near_N"], 1),
        "far_sub_rate": bnd["far_S"] / max(bnd["far_N"], 1),
        "near_share_of_reference_side_errors":
            (bnd["near_S"] + bnd["near_D"])
            / max(bnd["near_S"] + bnd["near_D"] + bnd["far_S"] + bnd["far_D"], 1),
        "note": "reference-side errors only (S+D); insertions have no reference token "
                "and therefore no time",
        "del_rate_near_minus_far_ci": strata_ci(near_counts, far_counts, clusters),
    }
    res["overlap"] = {
        "overlap_ref_tokens": ov_err["ov_N"], "clean_ref_tokens": ov_err["no_N"],
        "overlap_share_of_ref_tokens":
            ov_err["ov_N"] / max(ov_err["ov_N"] + ov_err["no_N"], 1),
        "overlap_del_rate": ov_err["ov_D"] / max(ov_err["ov_N"], 1),
        "clean_del_rate": ov_err["no_D"] / max(ov_err["no_N"], 1),
        "overlap_sub_rate": ov_err["ov_S"] / max(ov_err["ov_N"], 1),
        "clean_sub_rate": ov_err["no_S"] / max(ov_err["no_N"], 1),
        "overlap_share_of_reference_side_errors":
            (ov_err["ov_S"] + ov_err["ov_D"])
            / max(ov_err["ov_S"] + ov_err["ov_D"] + ov_err["no_S"] + ov_err["no_D"], 1),
        "del_rate_overlap_minus_clean_ci": strata_ci(ovc, noc, clusters),
        "definition": "regular diarization has >=2 simultaneous speakers where the "
                      "exclusive timeline kept one",
    }

    dd_turn = res["contrasts"]["turn vs window"]["wer"]["delta"]
    dd_plac = res["contrasts"]["placebo vs window"]["wer"]["delta"]
    res["A2_difference_of_differences_wer"] = {
        "turn_minus_window": dd_turn, "placebo_minus_window": dd_plac,
        "speaker_specific": dd_turn - dd_plac,
        "read_as": "the paired 'turn vs placebo' contrast is the test of this quantity",
    }

    a, b = arm_rows["vote3_turn"], arm_rows["vote3_window"]
    allk = list(range(len(items)))
    full = rates(a)["wer"] - rates(b)["wer"]
    worst, shift = None, 0.0
    for k in allk:
        keep = [i for i in allk if i != k]
        d = rates([a[i] for i in keep])["wer"] - rates([b[i] for i in keep])["wer"]
        if abs(d - full) > abs(shift):
            worst, shift = items[k]["item_id"], d - full
    res["influence"]["wer_turn_vs_window"] = {
        "delta": full, "max_shift_window": worst, "delta_without": full + shift}

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"-> {OUT}")
    for k in ("single:scribe-v2-clean", f"single:{THIRD}", "single:soniox",
              "single:hf-openai-whisper-large-v3", f"single:{TURBO}",
              "vote3_window", "vote3_turn", "vote3_placebo",
              "vote3_turn_exclusive", "vote4_window", "greedy_turn_oracle"):
        v = res["arms"][k]
        log(f"  {k:38s} wer={v['wer']:.4f}  del={v['del_rate']:.4f}  "
            f"ins={v['ins_rate']:.4f}  sub={v['sub_rate']:.4f}")
    for k, v in res["A6_detector"].items():
        log(f"  detector[{k:9s}] P={v['precision']} R={v['recall']} "
            f"null P={v['random_matched_segments']['precision']}")


def strata_ci(a_counts, b_counts, clusters):
    """Meeting-clustered CI on a difference of two pooled rates over DIFFERENT tokens.

    `cluster_bootstrap` asserts the two arms share denominators, which is right for a
    paired arm-vs-arm contrast and wrong here: near-boundary and far-from-boundary
    tokens are disjoint sets. Meetings are still the resampling unit and both pooled
    rates are recomputed inside every replicate.
    """
    import numpy as np
    groups = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = sorted(groups)
    a = np.array(a_counts, dtype=float)
    b = np.array(b_counts, dtype=float)
    rng = np.random.default_rng(7)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        da, db = a[idx, 1].sum(), b[idx, 1].sum()
        diffs[i] = (a[idx, 0].sum() / da if da else np.nan) - \
                   (b[idx, 0].sum() / db if db else np.nan)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    point = (a[:, 0].sum() / max(a[:, 1].sum(), 1)) - \
            (b[:, 0].sum() / max(b[:, 1].sum(), 1))
    return {"delta": float(point), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_clusters": len(keys),
            "note": "disjoint strata, meetings resampled"}


if __name__ == "__main__":
    main()
