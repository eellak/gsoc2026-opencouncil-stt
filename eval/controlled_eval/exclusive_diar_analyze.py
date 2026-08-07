#!/usr/bin/env python3
"""Phase 1 analysis, exactly the three frozen metrics and the four-part gate.

Spec: docs/specs/exclusive-diarization-preregistration.md § Phase 1.
Input: ~/.cache/oc-overlap/exclusive_phase1.json (written by exclusive_diar_run.py).
Output: eval/controlled_eval/results_exclusive_phase1.json (aggregates only, no text).

Nothing here may be changed after the first result is looked at; the gate values
live in the freeze record and are re-read from it rather than retyped.
"""
from __future__ import annotations

import collections
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, log  # noqa: E402
from oc_merge_port import find_best_speaker, segments  # noqa: E402

STORE = SC / "exclusive_phase1.json"
FREEZE = SC / "exclusive_freeze.json"
MANIFEST = SC / "synth_overlap_manifest.json"
OUT = Path(__file__).resolve().parent / "results_exclusive_phase1.json"
GUARD = 0.5      # ± guard around the event for label mapping
MIN_MAP_SEC = 1.0
MERGE_GAP = 0.010
TILE = 5.0
MIN_TAIL_TILE = 1.0
N_BOOT = 10000
BOOT_SEED = 7


# ------------------------------------------------------------------ primitives
def merge_same_speaker(segs, gap=MERGE_GAP):
    per = collections.defaultdict(list)
    for s in segs:
        per[s.speaker].append((s.start, s.end))
    out = {}
    for spk, iv in per.items():
        iv.sort()
        m = []
        for s, e in iv:
            if m and s - m[-1][1] <= gap:
                m[-1] = (m[-1][0], max(m[-1][1], e))
            else:
                m.append((s, e))
        out[spk] = m
    return out


def time_in(segs, t0, t1):
    per = collections.Counter()
    for s in segs:
        a, b = max(t0, s.start), min(t1, s.end)
        if b > a:
            per[s.speaker] += b - a
    return per


def overlap_outside(segs_a, segs_b, ev0, ev1):
    """Pairwise speaker-time overlap between two timelines, event region excluded."""
    ma, mb = merge_same_speaker(segs_a), merge_same_speaker(segs_b)
    g0, g1 = ev0 - GUARD, ev1 + GUARD
    out = collections.Counter()
    for sa, iva in ma.items():
        for sb, ivb in mb.items():
            tot = 0.0
            for a0, a1 in iva:
                for b0, b1 in ivb:
                    lo, hi = max(a0, b0), min(a1, b1)
                    if hi <= lo:
                        continue
                    # subtract the guarded event region from the intersection
                    tot += (hi - lo) - max(0.0, min(hi, g1) - max(lo, g0))
            if tot > 0:
                out[(sa, sb)] = tot
    return out


def greedy_map(segs_a, segs_c, ev0, ev1):
    """Arm-A label -> arm-C label, greedy max overlap outside the guarded event."""
    pairs = overlap_outside(segs_a, segs_c, ev0, ev1)
    mapping, used_a, used_c = {}, set(), set()
    for (a, c), v in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if v < MIN_MAP_SEC or a in used_a or c in used_c:
            continue
        mapping[a] = c
        used_a.add(a)
        used_c.add(c)
    return mapping


def tiles(duration):
    out, t = [], 0.0
    while t < duration:
        end = min(t + TILE, duration)
        if end - t >= MIN_TAIL_TILE or not out:
            out.append((t, end))
        t += TILE
    return out


def tile_words(t0, t1):
    """Five equal sub-spans standing in for words; frozen, ASR-free."""
    step = (t1 - t0) / 5
    return [(t0 + i * step, t0 + (i + 1) * step) for i in range(5)]


def branch_counts(segs, duration):
    c = collections.Counter()
    for t0, t1 in tiles(duration):
        r = find_best_speaker(segs, t0, t1, tile_words(t0, t1))
        c["drop" if r is None else r.branch] += 1
    return c


def cluster_bootstrap_prop(flags, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """Two-sided 90% interval on a proportion, resampling whole clusters."""
    import random
    by = collections.defaultdict(list)
    for f, c in zip(flags, clusters):
        by[c].append(f)
    keys = list(by)
    if not keys:
        return (None, None)
    rng = random.Random(seed)
    props = []
    for _ in range(n_boot):
        picked = [by[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [x for g in picked for x in g]
        if flat:
            props.append(sum(flat) / len(flat))
    props.sort()
    return (round(props[int(0.05 * len(props))], 4),
            round(props[int(0.95 * len(props))], 4))


# ------------------------------------------------------------------ main
def main():
    store = json.loads(STORE.read_text())
    man = json.loads(MANIFEST.read_text())
    freeze = json.loads(FREEZE.read_text())
    gate = freeze["phase1"]["gate"]
    by_id = {it["item_id"]: it for it in man["items"]}

    def get(iid, arm, flag):
        return store.get(f"{iid}|{arm}|{flag}")

    ids = sorted({k.split("|")[0] for k in store})
    rows = []
    for iid in ids:
        it = by_id[iid]
        ev0 = float(it["event_start_sec"])
        ev1 = ev0 + float(it["event_dur_sec"])
        dur = float(it["window_dur_sec"])
        row = {"item_id": iid, "meeting_id": it["meeting_id"],
               "event": [round(ev0, 3), round(ev1, 3)], "failed": False,
               "reason": None}

        a, ce, cb = get(iid, "A", "excl"), get(iid, "C", "excl"), get(iid, "C", "base")
        if not a or not ce or "diarization" not in a or "diarization" not in ce:
            row.update(failed=True, reason="missing_job")
            rows.append(row)
            continue

        a_reg = segments(a["diarization"])
        c_reg = segments(ce["diarization"])
        c_exc_raw = ce.get("exclusiveDiarization")
        if not c_exc_raw:
            row.update(failed=True, reason="no_exclusive_key")
            rows.append(row)
            continue
        c_exc = segments(c_exc_raw)
        if not {s.speaker for s in c_exc} <= {s.speaker for s in c_reg}:
            row.update(failed=True, reason="label_not_subset")
            rows.append(row)
            continue

        # invariance check against the paired unflagged call
        if cb and "diarization" in cb:
            row["regular_invariant"] = (
                [(s.start, s.end, s.speaker) for s in segments(cb["diarization"])]
                == [(s.start, s.end, s.speaker) for s in c_reg])

        # --- metric 1: main-speaker absorption
        a_ev = time_in(a_reg, ev0, ev1)
        if not a_ev:
            row["absorption_excluded"] = "no_clean_speech_in_event"
        else:
            local_main_a = a_ev.most_common(1)[0][0]
            mapping = greedy_map(a_reg, c_reg, ev0, ev1)
            target = mapping.get(local_main_a)
            row["local_main_arm_a"] = local_main_a
            row["local_main_mapped"] = target
            if target is None:
                row.update(failed=True, reason="mapping_failure")
                row["absorb_exclusive"] = False
                row["absorb_regular"] = False
            else:
                e_ev, r_ev = time_in(c_exc, ev0, ev1), time_in(c_reg, ev0, ev1)
                row["absorb_exclusive"] = bool(e_ev) and e_ev.most_common(1)[0][0] == target
                row["absorb_regular"] = bool(r_ev) and r_ev.most_common(1)[0][0] == target

        # --- metric 2: fragmentation of the local main speaker
        tgt = row.get("local_main_mapped")
        if tgt:
            n_exc = len(merge_same_speaker(c_exc).get(tgt, []))
            n_reg = len(merge_same_speaker(c_reg).get(tgt, []))
            mapping_a = {v: k for k, v in greedy_map(a_reg, c_reg, ev0, ev1).items()}
            a_lbl = mapping_a.get(tgt)
            a_exc = segments(a.get("exclusiveDiarization") or [])
            n_a = len(merge_same_speaker(a_exc).get(a_lbl, [])) if a_lbl else 0
            row["frag_exc_over_reg"] = round(n_exc / n_reg, 4) if n_reg else None
            row["frag_C_over_A"] = round(n_exc / n_a, 4) if n_a else None
            if not n_reg or not n_a:
                row.update(failed=True, reason=row["reason"] or "zero_denominator")

        # --- metric 3: merge simulation
        row["tiles_regular"] = dict(branch_counts(c_reg, dur))
        row["tiles_exclusive"] = dict(branch_counts(c_exc, dur))
        rows.append(row)

    # ------------------------------------------------------------- aggregate
    def worst_case(r, field, good):
        """Failed items are scored worst-case for the proposal, never dropped."""
        if r["failed"]:
            return good is False
        return r.get(field)

    scored = [r for r in rows if "absorption_excluded" not in r]
    excluded = len(rows) - len(scored)
    absorb_e = [bool(worst_case(r, "absorb_exclusive", False)) for r in scored]
    absorb_r = [bool(r.get("absorb_regular")) if not r["failed"] else False
                for r in scored]
    clusters = [r["meeting_id"] for r in scored]

    rate_e = sum(absorb_e) / len(absorb_e) if absorb_e else 0.0
    rate_r = sum(absorb_r) / len(absorb_r) if absorb_r else 0.0

    def frag_list(field):
        out = []
        for r in rows:
            v = r.get(field)
            if r["failed"] or v is None:
                out.append(math.inf)  # worst case: counted as above 1.2
            else:
                out.append(v)
        return out

    f1, f2 = frag_list("frag_exc_over_reg"), frag_list("frag_C_over_A")

    def med(xs):
        finite = [x for x in xs if math.isfinite(x)]
        return round(statistics.median(xs) if all(math.isfinite(x) for x in xs)
                     else (statistics.median(finite) if finite else math.inf), 4)

    def frac_above(xs, thr=1.2):
        return round(sum(1 for x in xs if x > thr) / len(xs), 4) if xs else 1.0

    t_reg = collections.Counter()
    t_exc = collections.Counter()
    for r in rows:
        if r["failed"]:
            n = len(tiles(float(by_id[r["item_id"]]["window_dur_sec"])))
            t_exc["drop"] += n
            t_reg.update(r.get("tiles_regular") or {})
            continue
        t_reg.update(r.get("tiles_regular") or {})
        t_exc.update(r.get("tiles_exclusive") or {})
    bad_reg = t_reg["guess"] + t_reg["drop"]
    bad_exc = t_exc["guess"] + t_exc["drop"]

    inv = [r["regular_invariant"] for r in rows if "regular_invariant" in r]

    g = {
        "absorption_ge_0.80": rate_e >= gate["absorption_rate"],
        "absorption_beats_regular": rate_e >= rate_r,
        "frag_medians_le_1.2": (med(f1) <= gate["fragmentation_median"]
                                and med(f2) <= gate["fragmentation_median"]),
        "frag_frac_above_le_0.10": (frac_above(f1) <= gate["fragmentation_frac_above_1_2"]
                                    and frac_above(f2) <= gate["fragmentation_frac_above_1_2"]),
        "merge_sim_not_worse": bad_exc <= bad_reg,
    }

    res = {
        "n_items": len(rows),
        "n_failed": sum(1 for r in rows if r["failed"]),
        "failure_reasons": dict(collections.Counter(
            r["reason"] for r in rows if r["failed"])),
        "n_excluded_no_clean_speech": excluded,
        "regular_invariance": {"n_checked": len(inv), "n_identical": sum(1 for x in inv if x)},
        "metric1_absorption": {
            "exclusive": round(rate_e, 4),
            "regular": round(rate_r, 4),
            "n": len(absorb_e),
            "exclusive_ci90_cluster_meeting": cluster_bootstrap_prop(
                [int(x) for x in absorb_e], clusters),
            "n_meetings": len(set(clusters)),
        },
        "metric2_fragmentation": {
            "exc_over_reg": {"median": med(f1), "frac_above_1.2": frac_above(f1)},
            "excC_over_excA": {"median": med(f2), "frac_above_1.2": frac_above(f2)},
        },
        "metric3_merge_sim": {
            "regular": dict(t_reg), "exclusive": dict(t_exc),
            "guess_plus_drop": {"regular": bad_reg, "exclusive": bad_exc},
        },
        "gate": g,
        "gate_passed": all(g.values()),
        "freeze_record": FREEZE.name,
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"\n-> {OUT}")
    log("\nPHASE 1 GATE: " + ("PASS" if res["gate_passed"] else "FAIL"))


if __name__ == "__main__":
    main()
