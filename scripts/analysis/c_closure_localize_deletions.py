#!/usr/bin/env python3
"""Arm C closure: localize deletions in the 9 phase-2 windows using word
timestamps, and split them into (a) deletions sitting in uncovered VAD-positive
gaps (a shifted second pass could in principle recover them) vs (b) deletions
inside decoder-covered speech (a shifted pass cannot).

Inputs:
  - phase-1  : data/reports/finetune-research/c-preliminary-vad-2026-08-12.json
               (Silero islands per window, frozen config recorded there)
  - phase-2  : data/reports/finetune-research/c-phase2-timed-decode-2026-08-12.json
               (word-timestamped re-decode, eval-A config + word_timestamps=True)
  - refs     : data/reports/finetune-research/2026-08-10-corrected-adapter-report-full.json
  - tokenizer: eval.controlled_eval.eval_freeze.ftoks (frozen)
  - alignment: scripts/ds_wer.align (same tie-break as exp_same_stack.sdi)

Method: align ref vs phase-2 hyp tokens; group consecutive DELs into runs; each
run's audio location is the hole between the last aligned hyp word before it and
the first after it. Split the hole's time into (covered by an emitted segment)
vs (VAD speech not covered by any emitted segment). A run counts as
"uncovered_gap" when its hole contains > 1.0 s of uncovered VAD speech,
otherwise "within_covered".

Output: data/reports/finetune-research/c-closure-vad-2026-08-12.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from ds_wer import align, MATCH, SUB, DEL, INS  # noqa: E402

P1 = ROOT / "data/reports/finetune-research/c-preliminary-vad-2026-08-12.json"
P2 = ROOT / "data/reports/finetune-research/c-phase2-timed-decode-2026-08-12.json"
REPORT = ROOT / "data/reports/finetune-research/2026-08-10-corrected-adapter-report-full.json"
OUT = ROOT / "data/reports/finetune-research/c-closure-vad-2026-08-12.json"

UNCOVERED_RUN_THRESHOLD_S = 1.0  # frozen before looking at any number


def interval_intersection(a: list[list[float]], b: list[list[float]]) -> float:
    """Total overlap seconds between two interval lists."""
    total = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def subtract(spans: list[list[float]], minus: list[list[float]]) -> list[list[float]]:
    """spans − minus, as interval list."""
    out = []
    for s, e in spans:
        pieces = [[s, e]]
        for ms, me in minus:
            nxt = []
            for ps, pe in pieces:
                if me <= ps or ms >= pe:
                    nxt.append([ps, pe])
                else:
                    if ms > ps:
                        nxt.append([ps, ms])
                    if me < pe:
                        nxt.append([me, pe])
            pieces = nxt
        out.extend(pieces)
    return [p for p in out if p[1] - p[0] > 1e-6]


def main() -> None:
    p1 = json.loads(P1.read_text())
    p2 = json.loads(P2.read_text())
    refs = {it["itemId"]: it["referenceText"]
            for it in json.loads(REPORT.read_text())["items"]}
    islands_by_wid = {r["window_id"]: r["islands"] for r in p1["per_window"]}
    p1_dels = {r["window_id"]: r["del"] for r in p1["per_window"]}

    windows_out = {}
    tot_del = tot_gap = tot_cov = 0
    for wid, w in p2["windows"].items():
        islands = [[float(a), float(b)] for a, b in islands_by_wid[wid]]
        seg_spans = [[s["start"], s["end"]] for s in w["segments"]]

        # hyp tokens with word times (ftoks may split a word into 0..n tokens)
        hyp_toks, hyp_times = [], []
        for s in w["segments"]:
            for word in s["words"]:
                for t in ftoks(word["word"]):
                    hyp_toks.append(t)
                    hyp_times.append((word["start"], word["end"]))

        ref_toks = ftoks(refs[wid])
        ops = align(ref_toks, hyp_toks)
        n_del = sum(1 for op, _, _ in ops if op == DEL)

        # group consecutive DELs into runs, bracketed by timed hyp neighbours
        runs = []
        last_hyp_j = None
        i = 0
        while i < len(ops):
            op, ri, hj = ops[i]
            if op != DEL:
                if hj is not None:
                    last_hyp_j = hj
                i += 1
                continue
            run_ref = []
            while i < len(ops) and ops[i][0] == DEL:
                run_ref.append(ops[i][1])
                i += 1
            next_hyp_j = None
            for op2, _, hj2 in ops[i:]:
                if hj2 is not None:
                    next_hyp_j = hj2
                    break
            t0 = hyp_times[last_hyp_j][1] if last_hyp_j is not None else 0.0
            t1 = (hyp_times[next_hyp_j][0] if next_hyp_j is not None
                  else w["duration"])
            hole = [[t0, t1]] if t1 > t0 else []
            covered_s = interval_intersection(hole, seg_spans)
            uncovered_vad = subtract(
                [[max(s, t0), min(e, t1)] for s, e in islands
                 if min(e, t1) > max(s, t0)], seg_spans)
            uncovered_s = sum(e - s for s, e in uncovered_vad)
            klass = ("uncovered_gap" if uncovered_s > UNCOVERED_RUN_THRESHOLD_S
                     else "within_covered")
            runs.append({
                "n_del": len(run_ref),
                "ref_words": [ref_toks[r] for r in run_ref],
                "hole": [round(t0, 2), round(t1, 2)],
                "hole_s": round(t1 - t0, 2),
                "covered_in_hole_s": round(covered_s, 2),
                "uncovered_vad_in_hole_s": round(uncovered_s, 2),
                "class": klass,
            })

        gap_dels = sum(r["n_del"] for r in runs if r["class"] == "uncovered_gap")
        cov_dels = n_del - gap_dels

        # window-level robustness bound, independent of word-timestamp quality:
        # total VAD speech not covered by ANY emitted segment, times the
        # reference speech rate, caps coverage-recoverable deletions even if
        # run localization is wrong (word timestamps collapse near segment
        # boundaries, e.g. the 113-del run in jan30).
        vad_total = sum(e - s for s, e in islands)
        uncovered_all = subtract([list(x) for x in islands], seg_spans)
        uncovered_all_s = sum(e - s for s, e in uncovered_all)
        ref_rate = len(ref_toks) / vad_total if vad_total else 0.0
        window_bound = uncovered_all_s * ref_rate
        tot_del += n_del
        tot_gap += gap_dels
        tot_cov += cov_dels
        windows_out[wid] = {
            "phase2_deletions": n_del,
            "phase1_control_deletions": p1_dels[wid],
            "dels_in_uncovered_vad_gaps": gap_dels,
            "dels_within_covered_spans": cov_dels,
            "gap_fraction": round(gap_dels / n_del, 4) if n_del else None,
            "uncovered_vad_total_s": round(uncovered_all_s, 2),
            "ref_speech_rate_tok_per_s": round(ref_rate, 2),
            "window_level_recoverable_bound_dels": round(window_bound, 1),
            "n_runs": len(runs),
            "runs": runs,
        }

    result = {
        "generated": "2026-08-12",
        "method_note": ("run classified uncovered_gap when its inter-word hole "
                        f"contains > {UNCOVERED_RUN_THRESHOLD_S} s of VAD speech "
                        "not covered by any emitted segment; threshold frozen "
                        "before decoding results were inspected"),
        "aggregate": {
            "windows": len(windows_out),
            "total_deletions_phase2": tot_del,
            "deletions_in_uncovered_vad_gaps": tot_gap,
            "deletions_within_covered_spans": tot_cov,
            "fraction_recoverable_by_shifted_pass": round(tot_gap / tot_del, 4)
            if tot_del else None,
            "fraction_within_covered": round(tot_cov / tot_del, 4)
            if tot_del else None,
        },
        "robustness_note": (
            "word timestamps collapse near segment boundaries (the 113-del run "
            "in jan30 has a zero-width hole), so run localization alone is not "
            "trusted; window_level_recoverable_bound_dels caps recoverable "
            "deletions from total uncovered VAD time x reference speech rate, "
            "independent of word timing quality"),
        "per_window": windows_out,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    print(f"{'window':44s} {'p2del':>5} {'p1del':>5} {'gap':>4} {'cov':>4} frac_gap")
    for wid, r in sorted(windows_out.items(),
                         key=lambda kv: -kv[1]["phase2_deletions"]):
        print(f"{wid:44s} {r['phase2_deletions']:5d} "
              f"{r['phase1_control_deletions']:5d} "
              f"{r['dels_in_uncovered_vad_gaps']:4d} "
              f"{r['dels_within_covered_spans']:4d}  {r['gap_fraction']}")
    print("\naggregate:", json.dumps(result["aggregate"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
