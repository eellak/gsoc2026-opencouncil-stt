#!/usr/bin/env python3
"""Parakeet-TDT via pyannoteAI: single system, monotonicity, fourth voter, omission detector.

Four measurements, in the order wayfinder issue #17 froze them, because the first
can kill the rest:

  M1  Parakeet-TDT-0.6b-v3 as a standalone Greek council ASR. No measurement of it
      on this domain exists anywhere; the published FLEURS/Common Voice numbers are
      read speech and do not transfer.
  M2  Monotonicity. A transducer cannot silently skip audio the way an
      autoregressive decoder can, so the prediction is del_rate below every other
      system's. Confirmed or refused, not softened.
  M3  Fourth voter. Quartet vote against the incumbent trio. The tie-break rule was
      written into the issue before any number was seen: `consensus_pick` is argmax
      of summed pairwise similarity (well defined for any n >= 3), and exact ties go
      to the first system of the frozen priority order below.
  M4  Omission detector. Where `exclusiveDiarization` asserts speech and our own
      transcript has no words, how often is that a real deletion against the
      reference? Precision/recall for a detector that the 2026 scan found does not
      exist in the literature.

Sample: 247 windows — the 253 common to all 9 providers of the 2026-08-10 report,
minus the six sealed 2026-08 holdout windows that sit inside them. Every comparator
is recomputed here on those same 247; nothing is quoted from the 253-window run.

Writes results_parakeet_voter.json (aggregates only, no transcript text).

Env: SC (cache dir) N_BOOT (10000)
"""
from __future__ import annotations

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.controlled_eval import bench_data as B                       # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import rates, sdi       # noqa: E402
from eval.controlled_eval.parakeet_run import RUN_ID, sc, target_items  # noqa: E402
from eval.controlled_eval.scoring import (cluster_bootstrap, head2head,  # noqa: E402
                                          wtoks)

PARAKEET = "pyannote-parakeet"
BASE = "scribe-v2-clean"
SECOND = "soniox"
THIRD = "oc-runpod-fixed-2026-08-10"
OURS = THIRD

# Frozen before any number was seen (issue #17). max() returns the first maximal
# element, so passing the voters in this order IS the tie-break rule.
PRIORITY = [BASE, SECOND, THIRD, PARAKEET]

MIN_SPEECH_SEC = 1.5   # a flagged exclusive-speech interval must last at least this
MIN_DEL_RUN = 3        # a ground-truth deletion run must be at least this many words
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
N_RANDOM = int(os.environ.get("N_RANDOM", "200"))   # draws per window for the nulls
OUT = Path(__file__).with_name("results_parakeet_voter.json")


def log(m):
    print(m, flush=True)


# ----------------------------------------------------------------- alignment helpers
def align_ops(ref: list[str], hyp: list[str]) -> list[str]:
    """Per-reference-token operation from the Levenshtein backtrace: M, S or D.

    `sdi()` returns the counts; the omission detector needs to know WHICH reference
    words were dropped, so the backtrace is replayed here and insertions discarded.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], op[i][0] = i, "D"
    for j in range(1, m + 1):
        d[0][j], op[0][j] = j, "I"
    for i in range(1, n + 1):
        ai = ref[i - 1]
        di, dim1, opi = d[i], d[i - 1], op[i]
        for j in range(1, m + 1):
            if ai == hyp[j - 1]:
                di[j], opi[j] = dim1[j - 1], "M"
                continue
            sub, dele, ins = dim1[j - 1] + 1, dim1[j] + 1, di[j - 1] + 1
            best = min(sub, dele, ins)
            di[j] = best
            opi[j] = "S" if best == sub else ("D" if best == dele else "I")
    out = [""] * n
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o in ("M", "S"):
            out[i - 1] = o
            i, j = i - 1, j - 1
        elif o == "D":
            out[i - 1] = "D"
            i -= 1
        else:
            j -= 1
    return out


def _code(tokens: list[str], enc: dict) -> str:
    s = []
    for w in tokens:
        if w not in enc:
            enc[w] = chr(0x3000 + len(enc))
        s.append(enc[w])
    return "".join(s)


def token_times(tokens: list[str], pk_tokens: list[str], pk_times: list[float],
                audio_end: float) -> tuple[list[float], int]:
    """Give every token in `tokens` a time, by anchoring on words it shares with the
    Parakeet word stream and interpolating linearly between anchors.

    Tokens outside the anchor range are extrapolated at the mean token rate implied
    by the anchors, not clamped onto the nearest anchor. Clamping was the first
    version and it is wrong in a way that matters here: a whole leading or trailing
    deletion run would collapse onto a single instant belonging to the neighbouring
    recognised speech, which both invents flag intersections and destroys real ones.

    Returns (times, n_anchors). n_anchors is a diagnostic: if Parakeet's text is bad
    the anchors thin out and the interpolation is noise.
    """
    if not tokens:
        return [], 0
    enc: dict = {}
    a, b = _code(tokens, enc), _code(pk_tokens, enc)
    anchors = []
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            anchors.append((blk.a + k, pk_times[blk.b + k]))
    n = len(tokens)
    if not anchors:
        # No shared vocabulary at all: spread tokens evenly over the whole window.
        return [audio_end * i / max(n - 1, 1) for i in range(n)], 0

    times: list[float | None] = [None] * n
    for idx, t in anchors:
        times[idx] = t
    known = [i for i, t in enumerate(times) if t is not None]
    for lo, hi in zip(known, known[1:]):
        if hi - lo > 1:
            t0, t1 = times[lo], times[hi]
            for k in range(lo + 1, hi):
                times[k] = t0 + (t1 - t0) * (k - lo) / (hi - lo)
    first, last = known[0], known[-1]
    if last > first:
        rate = (times[last] - times[first]) / (last - first)
    else:
        rate = audio_end / max(n, 1)
    for i in range(first - 1, -1, -1):
        times[i] = max(0.0, times[i + 1] - rate)
    for i in range(last + 1, n):
        times[i] = min(audio_end, times[i - 1] + rate)
    return [float(t) for t in times], len(anchors)


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# ------------------------------------------------------------------------ M4 detector
def detector_window(pk: dict, ref: str, ours: str, rng) -> dict:
    """Flags, ground-truth deletion runs, and their intersection, for one window."""
    words = pk["wordLevelTranscription"]
    pk_tokens, pk_times = [], []
    for w in words:
        tt = wtoks(w["text"])
        if not tt:
            continue
        mid = (float(w["start"]) + float(w["end"])) / 2
        for t in tt:                       # a word entry can normalise to >1 token
            pk_tokens.append(t)
            pk_times.append(mid)

    segs = [(float(s["start"]), float(s["end"])) for s in pk["exclusiveDiarization"]]
    audio_end = max([e for _, e in segs] + [pk_times[-1] if pk_times else 0.0] + [0.0])

    ref_toks, our_toks = wtoks(ref), wtoks(ours)
    our_times, our_anchors = token_times(our_toks, pk_tokens, pk_times, audio_end)
    ref_times, ref_anchors = token_times(ref_toks, pk_tokens, pk_times, audio_end)

    # flags: exclusive-diarization speech segments >= MIN_SPEECH_SEC with no word of ours
    eligible = [(s, e) for s, e in segs if e - s >= MIN_SPEECH_SEC]
    flags = [(s, e) for s, e in eligible
             if not any(s <= t <= e for t in our_times)]

    # ground truth: runs of >= MIN_DEL_RUN consecutive deletions against the reference
    ops = align_ops(ref_toks, our_toks)
    # the detector's ground truth and the reported deletion rate must come from the
    # same alignment, or they are two different measurements wearing one name
    assert sum(1 for o in ops if o == "D") == sdi(ref, ours)[1], \
        "align_ops and sdi disagree on deletions - the backtraces have drifted apart"
    runs, i = [], 0
    while i < len(ops):
        if ops[i] == "D":
            j = i
            while j < len(ops) and ops[j] == "D":
                j += 1
            if j - i >= MIN_DEL_RUN:
                runs.append((ref_times[i], ref_times[j - 1]))
            i = j
        else:
            i += 1

    tp_flags = sum(1 for f in flags if any(overlaps(f, r) for r in runs))
    hit_runs = sum(1 for r in runs if any(overlaps(f, r) for f in flags))

    # Two nulls, because the first one I wrote is not a fair one.
    #
    # `uniform`: same count, same durations, placed anywhere in the window. This is
    #   what the preregistration said, and it is too weak - deletions happen during
    #   speech, and a uniform interval can land in silence. Kept for honesty.
    # `matched`: draw the same NUMBER of intervals from the eligible speech segments,
    #   uniformly at random. This is the null that isolates what the detector claims:
    #   given that you are going to point at k long speech segments, does "our
    #   transcript is empty here" beat pointing at random ones?
    rnd_u_tp = rnd_u_hit = rnd_m_tp = rnd_m_hit = 0
    k = len(flags)
    for _ in range(N_RANDOM):
        ru = []
        for s, e in flags:
            d = e - s
            st = float(rng.uniform(0, max(audio_end - d, 0.001)))
            ru.append((st, st + d))
        rnd_u_tp += sum(1 for f in ru if any(overlaps(f, r) for r in runs))
        rnd_u_hit += sum(1 for r in runs if any(overlaps(f, r) for f in ru))

        if k and eligible:
            idx = rng.choice(len(eligible), size=min(k, len(eligible)),
                             replace=False)
            rm = [eligible[i] for i in idx]
        else:
            rm = []
        rnd_m_tp += sum(1 for f in rm if any(overlaps(f, r) for r in runs))
        rnd_m_hit += sum(1 for r in runs if any(overlaps(f, r) for f in rm))

    return {"n_flags": len(flags), "n_runs": len(runs), "n_eligible": len(eligible),
            "tp_flags": tp_flags, "hit_runs": hit_runs,
            "rnd_u_tp": rnd_u_tp / N_RANDOM, "rnd_u_hit": rnd_u_hit / N_RANDOM,
            "rnd_m_tp": rnd_m_tp / N_RANDOM, "rnd_m_hit": rnd_m_hit / N_RANDOM,
            "our_anchors": our_anchors, "ref_anchors": ref_anchors,
            "n_our_tokens": len(our_toks), "n_pk_tokens": len(pk_tokens),
            "flag_sec": sum(e - s for s, e in flags)}


# ------------------------------------------------------------------------------- main
def main() -> None:
    import numpy as np

    pk_dir = sc() / "parakeet"
    items = target_items()
    have = {p.stem for p in pk_dir.glob("*.json")}
    missing = [it["item_id"] for it in items if it["item_id"] not in have]
    if missing:
        raise SystemExit(
            f"{len(missing)} of the {len(items)} preregistered windows have no "
            f"pyannote response, e.g. {missing[:3]}. Refusing to publish a "
            "different sample under the frozen sample note - rerun parakeet_run.py.")
    log(f"{len(items)} windows, all with a pyannote response")

    pk_out = {}
    for it in items:
        d = json.loads((pk_dir / f"{it['item_id']}.json")
                       .read_text(encoding="utf-8"))["output"]
        pk_out[it["item_id"]] = d
        it["hyp"][PARAKEET] = " ".join(w["text"] for w in d["wordLevelTranscription"])

    report = B.load_report(RUN_ID)
    providers = B.provider_ids(report) + [PARAKEET]
    clusters = [it["cluster"] for it in items]
    disjoint = [not it["in_training"] for it in items]

    trio = [BASE, SECOND, THIRD]
    quartet = [p for p in PRIORITY]

    per_arm: dict[str, list] = {}
    for p in providers:
        per_arm[f"single:{p}"] = [sdi(it["ref"], it["hyp"][p]) for it in items]

    picks: dict[str, dict] = {}
    for name, pool in (("vote3", trio), ("vote4", quartet)):
        # pool is passed in frozen priority order; max() takes the first maximum,
        # which is exactly the declared tie-break.
        pool = [p for p in PRIORITY if p in pool]
        rows, cnt = [], {}
        for it in items:
            pick = B.consensus_pick(it, pool)
            cnt[pick] = cnt.get(pick, 0) + 1
            rows.append(sdi(it["ref"], it["hyp"][pick]))
        per_arm[name] = rows
        picks[name] = cnt

    for name, pool in (("oracle3", trio), ("oracle4", quartet)):
        per_arm[name] = [
            min([sdi(it["ref"], it["hyp"][p]) for p in pool],
                key=lambda c: c[0] + c[1] + c[2])
            for it in items]

    res = {
        "run_id": RUN_ID,
        "n_items": len(items),
        "n_meetings": len(set(clusters)),
        "n_training_disjoint": sum(disjoint),
        "sample_note": "253 common-to-all-providers windows minus 6 sealed "
                       "2026-08 holdout windows; all comparators recomputed here",
        "scorer": "eval/controlled_eval/scoring.py (not the benchmark app's)",
        "trio": trio, "quartet": quartet, "tie_break_priority": PRIORITY,
        "consensus_picks": picks,
        "arms": {k: rates(v) for k, v in per_arm.items()},
        "contrasts": {},
        "influence": {},
        "by_contamination": {},
        "detector": {},
    }

    def contrast(a: str, b: str) -> dict:
        out = {}
        for metric, idx in (("wer", None), ("del_rate", 1), ("ins_rate", 2),
                            ("sub_rate", 0)):
            if metric == "wer":
                ca = [(r[0] + r[1] + r[2], r[3]) for r in per_arm[a]]
                cb = [(r[0] + r[1] + r[2], r[3]) for r in per_arm[b]]
            else:
                ca = [(r[idx], r[3]) for r in per_arm[a]]
                cb = [(r[idx], r[3]) for r in per_arm[b]]
            out[metric] = cluster_bootstrap(ca, cb, clusters, n_boot=N_BOOT)
        out["head2head_wer"] = head2head(
            [(r[0] + r[1] + r[2], r[3]) for r in per_arm[a]],
            [(r[0] + r[1] + r[2], r[3]) for r in per_arm[b]])
        return out

    for a, b in (("vote4", "vote3"),
                 ("vote4", f"single:{BASE}"),
                 ("vote3", f"single:{BASE}"),
                 (f"single:{PARAKEET}", f"single:{SECOND}"),
                 (f"single:{PARAKEET}", f"single:{OURS}")):
        res["contrasts"][f"{a} vs {b}"] = contrast(a, b)

    # single-item domination on both headline deltas of M3
    for metric, idx in (("wer", None), ("del_rate", 1)):
        def val(rows, keep):
            r = rates([rows[i] for i in keep])
            return r["wer"] if metric == "wer" else r["del_rate"]
        allk = list(range(len(items)))
        full = val(per_arm["vote4"], allk) - val(per_arm["vote3"], allk)
        worst, shift = None, 0.0
        for k in allk:
            keep = [i for i in allk if i != k]
            d = val(per_arm["vote4"], keep) - val(per_arm["vote3"], keep)
            if abs(d - full) > abs(shift):
                worst, shift = items[k]["item_id"], d - full
        res["influence"][f"{metric}_vote4_vs_vote3"] = {
            "delta": full, "max_shift_window": worst, "delta_without": full + shift}

    for name in (f"single:{PARAKEET}", f"single:{BASE}", f"single:{OURS}",
                 "vote3", "vote4"):
        rows = per_arm[name]
        split = {}
        for lbl, want in (("disjoint", True), ("contaminated", False)):
            sub = [r for r, d in zip(rows, disjoint) if d is want]
            split[lbl] = rates(sub) if sub else None
        res["by_contamination"][name] = split

    # ---------------------------------------------------------------- M4
    rng = np.random.default_rng(7)
    det = [detector_window(pk_out[it["item_id"]], it["ref"], it["hyp"][OURS], rng)
           for it in items]
    F = sum(d["n_flags"] for d in det)
    R = sum(d["n_runs"] for d in det)
    TP = sum(d["tp_flags"] for d in det)
    HIT = sum(d["hit_runs"] for d in det)
    res["detector"] = {
        "min_speech_sec": MIN_SPEECH_SEC, "min_del_run": MIN_DEL_RUN,
        "n_random_draws": N_RANDOM, "ours": OURS,
        "n_flags": F, "n_deletion_runs": R,
        "n_eligible_segments": sum(d["n_eligible"] for d in det),
        "precision": TP / F if F else None,
        "recall": HIT / R if R else None,
        "random_uniform": {
            "precision": sum(d["rnd_u_tp"] for d in det) / F if F else None,
            "recall": sum(d["rnd_u_hit"] for d in det) / R if R else None},
        "random_matched_segments": {
            "precision": sum(d["rnd_m_tp"] for d in det) / F if F else None,
            "recall": sum(d["rnd_m_hit"] for d in det) / R if R else None},
        "windows_with_a_flag": sum(1 for d in det if d["n_flags"]),
        "flagged_seconds_total": round(sum(d["flag_sec"] for d in det), 1),
        "median_our_anchors_per_window": float(
            np.median([d["our_anchors"] for d in det])),
        "median_our_tokens_per_window": float(
            np.median([d["n_our_tokens"] for d in det])),
        "median_ref_anchors_per_window": float(
            np.median([d["ref_anchors"] for d in det])),
    }
    # Paired bootstrap on micro-precision, clustered by meeting. cluster_bootstrap
    # recomputes sum(num)/sum(den) per replicate, so n_flags is a valid exposure
    # denominator here; windows with no flag carry no precision and are dropped from
    # both arms together, which leaves the point estimate untouched.
    # Caveat: the random arm enters as each window's N_RANDOM-draw mean, so the CI
    # covers meeting-level sampling, not the Monte-Carlo placement noise.
    ca = [(d["tp_flags"], d["n_flags"]) for d in det]
    keep = [i for i, c in enumerate(ca) if c[1] > 0]
    for lbl, key in (("uniform", "rnd_u_tp"), ("matched_segments", "rnd_m_tp")):
        cb = [(det[i][key], det[i]["n_flags"]) for i in range(len(det))]
        if keep:
            res["detector"][f"precision_vs_random_{lbl}_ci"] = cluster_bootstrap(
                [ca[i] for i in keep], [cb[i] for i in keep],
                [clusters[i] for i in keep], n_boot=N_BOOT)

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    log(f"-> {OUT}")
    for k, v in res["arms"].items():
        log(f"  {k:36s} wer={v['wer']:.4f}  del={v['del_rate']:.4f}  "
            f"ins={v['ins_rate']:.4f}  sub={v['sub_rate']:.4f}")
    d = res["detector"]
    log(f"  detector: P={d['precision']} R={d['recall']}  "
        f"| null uniform P={d['random_uniform']['precision']} "
        f"| null matched P={d['random_matched_segments']['precision']} "
        f"R={d['random_matched_segments']['recall']}")


if __name__ == "__main__":
    main()
