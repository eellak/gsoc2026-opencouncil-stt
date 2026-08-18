#!/usr/bin/env python3
"""Assemble the bundle: audio, diarization, columns, deletions, three conditions.

Writes everything to `--out` (default `~/.cache/oc-public/tsfusion-2026-08/`). Nothing
it produces may enter git: `data.json` and the page carry verbatim council speech, the
same PII category as the 2026-07-21 history purge.

`eval/controlled_eval/msa.py` is imported and never touched. Its hash is recorded in
the manifest so a later reader can tell whether the alignment on the page is the one
the rest of the project is using.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from eval.controlled_eval import bench_data as B
from eval.controlled_eval.msa import align3, compose
from eval.controlled_eval.scoring import wtoks
from eval.tsfusion import conditions as CD
from eval.tsfusion import coords as CO
from eval.tsfusion import diarize as DZ
from eval.tsfusion import refalign as RA
from eval.tsfusion import speakers as SP
from eval.tsfusion import timing as TM
from eval.tsfusion.tokens import (soniox_timed_tokens, transfer_timestamps,
                                  whisper_timed_tokens)

ROOT = Path(__file__).resolve().parents[2]
SC = Path.home() / ".cache/oc-public"
RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
TRIO = ["scribe-v2-clean", "soniox", "oc-runpod-fixed-2026-08-10"]
SYSTEMS = ["scribe", "soniox", "whisper"]
BAND_FLOOR = 40


def band(a, b, c) -> int:
    return max(BAND_FLOOR, max(len(a), len(b), len(c)) - min(len(a), len(b), len(c)) + 20)


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------- inputs
def load_inputs():
    report = B.load_report(RUN_ID)
    items = {it["itemId"]: it for it in report["items"]}
    rw = json.loads((SC / "conf-substrate-2026-08/decode-rw.json").read_text())
    out = {}
    for w in CO.WINDOWS:
        it = items[w.item_id]
        sx = json.loads((SC / "composition-rt-2026-08/soniox-tokens" /
                         f"{w.item_id}.json").read_text())
        out[w.item_id] = {
            "span": w,
            "ref": wtoks(it["referenceText"]),
            "bench": {s: wtoks(it["perProvider"][p]["hypothesisText"])
                      for s, p in zip(SYSTEMS, TRIO)},
            "whisper_rw": wtoks(rw["windows"][w.item_id]["text"]),
            # timestamped streams, lifted into ABSOLUTE meeting time here and only here
            "soniox_timed": soniox_timed_tokens(sx["tokens"], t_offset=w.start),
            "whisper_timed": whisper_timed_tokens(
                rw["windows"][w.item_id]["segments"], t_offset=w.start),
        }
    return out, report


def pivot_for(texts: dict) -> int:
    """The whole-window consensus pick, exactly as `fusion_lab` computes it."""
    fake = {"hyp": {p: " ".join(texts[s]) for s, p in zip(SYSTEMS, TRIO)}}
    return TRIO.index(B.consensus_pick(fake, TRIO))


# --------------------------------------------------------------- per window
def build_window(win, diar_ex, diar_reg, bounds):
    """Everything the page needs for one benchmark window."""
    ref = win["ref"]
    hyps_c1 = [win["bench"][s] for s in SYSTEMS]
    hyps_c2 = [win["bench"]["scribe"], win["bench"]["soniox"], win["whisper_rw"]]

    # --- timestamps, transferred onto each hypothesis stream
    def transfers(hyps, whisper_stream_is_rw):
        tr = {}
        tr["soniox"] = transfer_timestamps(win["soniox_timed"], hyps[1])
        # C2/C3 use the rw text itself, so its timestamps are DIRECT, not transferred
        tr["whisper"] = transfer_timestamps(win["whisper_timed"], hyps[2])
        tr["_whisper_direct"] = whisper_stream_is_rw
        return tr

    tr_c1 = transfers(hyps_c1, False)
    tr_c2 = transfers(hyps_c2, True)

    cols_c1 = align3(*hyps_c1, band=band(*hyps_c1))
    cols_c2 = align3(*hyps_c2, band=band(*hyps_c2))
    w_c1, dec_c1 = compose(cols_c1, pivot=pivot_for(win["bench"]))
    p2 = pivot_for({"scribe": hyps_c2[0], "soniox": hyps_c2[1], "whisper": hyps_c2[2]})
    w_c2, dec_c2 = compose(cols_c2, pivot=p2)

    def column_sources(cols, tr, direct):
        """Per column, the interval each timestamped system offers for its entry."""
        cursor = {"scribe": 0, "soniox": 0, "whisper": 0}
        out, confs = [], []
        for col in cols:
            src, cf = {}, {}
            for k, sysname in enumerate(SYSTEMS):
                if col[k] is None:
                    continue
                j = cursor[sysname]
                cursor[sysname] += 1
                if sysname == "scribe":
                    continue
                t = tr[sysname]
                tok = t.intervals[j] if j < len(t.intervals) else None
                op = t.ops[j] if j < len(t.ops) else "unmatched"
                if tok is None:
                    src[sysname] = None
                    continue
                prov = tok.provenance
                if sysname == "whisper" and direct and prov == "observed_word":
                    prov = "observed_word"
                src[sysname] = {"start": tok.start, "end": tok.end,
                                "provenance": prov, "match": op,
                                "envelope": list(tok.envelope), "conf": tok.conf}
                if tok.conf is not None:
                    cf[sysname] = tok.conf
            out.append(src)
            confs.append(cf)
        return out, confs

    srcs_c1, confs_c1 = column_sources(cols_c1, tr_c1, False)
    srcs_c2, confs_c2 = column_sources(cols_c2, tr_c2, True)
    times_c1 = TM.place(srcs_c1, diar_ex, bounds)
    times_c2 = TM.place(srcs_c2, diar_ex, bounds)

    calls_c1 = [SP.assign(t.time_start, t.time_end, diar_ex, diar_reg,
                          unresolved=t.unresolved or t.time_start is None)
                for t in times_c1]
    calls_c2 = [SP.assign(t.time_start, t.time_end, diar_ex, diar_reg,
                          unresolved=t.unresolved or t.time_start is None)
                for t in times_c2]

    # --- C3, applying the frozen table
    c3_tokens, c3_changes, n_elig = [], [], 0
    for i, col in enumerate(cols_c2):
        ok, tok, why = CD.c3_eligible(col, times_c2[i], calls_c2[i], confs_c2[i])
        base = dec_c2[i]["token"]
        if ok:
            n_elig += 1
            if base is None:
                c3_changes.append({"col": i, "restored": tok,
                                   "time": [times_c2[i].time_start,
                                            times_c2[i].time_end],
                                   "speaker": calls_c2[i].speaker, "why": why})
                c3_tokens.append(tok)
                continue
        if base is not None:
            c3_tokens.append(base)

    # --- per system, aligned to the published text independently
    aligns = {}
    for k, s in enumerate(SYSTEMS):
        aligns[s] = RA.align_to_reference(s, ref, hyps_c1[k])
    aligns["W"] = RA.align_to_reference("W", ref, w_c1)

    return {
        "item_id": win["span"].item_id,
        "span": [win["span"].start, win["span"].end],
        "ref": ref,
        "hyps_c1": hyps_c1, "hyps_c2": hyps_c2,
        "cols_c1": cols_c1, "cols_c2": cols_c2,
        "dec_c1": dec_c1, "dec_c2": dec_c2,
        "w_c1": w_c1, "w_c2": w_c2, "w_c3": c3_tokens,
        "srcs_c1": srcs_c1, "times_c1": times_c1, "calls_c1": calls_c1,
        "confs_c1": confs_c1,
        "times_c2": times_c2, "calls_c2": calls_c2,
        "c3_changes": c3_changes, "c3_eligible": n_elig,
        "aligns": aligns,
        "transfer_stats": {
            s: {"target": tr_c1[s].n_target, "stable": tr_c1[s].n_stable,
                "ambiguous": tr_c1[s].n_ambiguous, "unmatched": tr_c1[s].n_unmatched}
            for s in ("soniox", "whisper")},
    }


# ------------------------------------------------------------------- assembly
def column_rows(w, diar_ex, diar_reg):
    """Display rows: one per C1 MSA column, plus everything known about it."""
    # map each system's token index -> the column it sits in
    col_of = {s: [] for s in SYSTEMS}
    for i, col in enumerate(w["cols_c1"]):
        for k, s in enumerate(SYSTEMS):
            if col[k] is not None:
                col_of[s].append(i)
    w_col = [i for i, d in enumerate(w["dec_c1"]) if d["token"] is not None]

    # reference words, via W's own alignment: which column each ref word landed in
    ref_at_col: dict[int, dict] = {}
    for op in w["aligns"]["W"].ops:
        if op.hyp_index is None:
            continue
        ci = w_col[op.hyp_index]
        ref_at_col[ci] = {"word": op.ref_word, "op": op.op,
                          "ambiguous": op.ambiguous, "ref_index": op.ref_index}

    # per-system op badges, projected onto columns
    sys_op: dict[str, dict[int, dict]] = {s: {} for s in SYSTEMS}
    for s in SYSTEMS:
        for op in w["aligns"][s].ops:
            if op.hyp_index is None:
                continue
            sys_op[s][col_of[s][op.hyp_index]] = {"op": op.op,
                                                  "ambiguous": op.ambiguous}

    rows = []
    for i, col in enumerate(w["cols_c1"]):
        t, call = w["times_c1"][i], w["calls_c1"][i]
        rep = t.representative
        entries = {s: col[k] for k, s in enumerate(SYSTEMS)}
        distinct = {e for e in col if e is not None}
        rows.append({
            "i": i,
            "window": w["item_id"],
            "scribe": entries["scribe"], "soniox": entries["soniox"],
            "whisper": entries["whisper"],
            "w": w["dec_c1"][i]["token"], "w_reason": w["dec_c1"][i]["reason"],
            "ref": ref_at_col.get(i),
            "agree": len(distinct) == 1 and all(e is not None for e in col),
            "occupancy": sum(e is not None for e in col),
            "time_start": t.time_start, "time_end": t.time_end,
            "time_method": t.time_method, "time_uncertainty": t.time_uncertainty,
            "time_conflict": t.time_conflict, "conflict_gap": t.conflict_gap,
            "unresolved": t.unresolved, "unresolved_reason": t.unresolved_reason,
            "sources": w["srcs_c1"][i],
            "page_t": (CO.to_page(rep) if rep is not None else None),
            "speaker_state": call.state, "speaker": call.speaker,
            "overlap_fraction": call.overlap_fraction,
            "speaker_runner_up": call.runner_up,
            "multiplicity": call.multiplicity,
            "phase30": (CO.whisper_phase(rep) if rep is not None else None),
            "phase30_meeting": (CO.chunk_phase(rep) if rep is not None else None),
            "in_seam": bool(rep is not None and CO.in_seam(rep)),
            "sys_op": {s: sys_op[s].get(i) for s in SYSTEMS},
        })
    return rows


def deletion_events(w, rows, diar_ex, bounds):
    """Edit events between columns, per system, placed by the fixed hierarchy."""
    col_of = {s: [] for s in SYSTEMS}
    for i, col in enumerate(w["cols_c1"]):
        for k, s in enumerate(SYSTEMS):
            if col[k] is not None:
                col_of[s].append(i)
    w_col = [i for i, d in enumerate(w["dec_c1"]) if d["token"] is not None]

    # a reference word is ANCHORED when some system matched it in a timed column
    anchor: dict[int, tuple] = {}
    proposals: dict[int, dict] = {}
    for s in SYSTEMS:
        for op in w["aligns"][s].ops:
            if op.op != "equal" or op.hyp_index is None:
                continue
            ci = col_of[s][op.hyp_index]
            t = w["times_c1"][ci]
            if t.time_method == "observed":
                anchor.setdefault(op.ref_index, (t.time_start, t.time_end))
                proposals.setdefault(op.ref_index, {})[s] = (t.time_start, t.time_end)

    # where in the display each reference index sits, so a gutter can be chosen
    ref_col: dict[int, int] = {}
    for op in w["aligns"]["W"].ops:
        if op.hyp_index is not None and op.ref_index is not None:
            ref_col[op.ref_index] = w_col[op.hyp_index]

    out = []
    for s in SYSTEMS + ["W"]:
        evs = RA.place_deletions(s, w["aligns"][s], anchor,
                                 {ri: p for ri, p in proposals.items()
                                  if s not in p},
                                 diar_ex, bounds)
        for e in evs:
            left = [r for r in ref_col if r < e.ref_index]
            e.after_column = ref_col[max(left)] if left else -1
            d = e.__dict__.copy()
            d["page_t"] = (CO.to_page(e.time_start)
                           if e.time_start is not None else None)
            d["page_t_end"] = (CO.to_page(e.time_end)
                               if e.time_end is not None else None)
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SC / "tsfusion-2026-08"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    msa_hash = sha16(ROOT / "eval/controlled_eval/msa.py")
    wins, report = load_inputs()

    diar = json.loads((out / "diar_padded.json").read_text())
    lo = CO.T0 - DZ.PAD
    ex = DZ.shift_and_crop(DZ.turns(diar, "exclusiveDiarization"), lo, CO.T0, CO.T1)
    reg = DZ.shift_and_crop(DZ.turns(diar, "diarization"), lo, CO.T0, CO.T1)
    bounds = (CO.T0, CO.T1)

    built = [build_window(wins[w.item_id], ex, reg, bounds) for w in CO.WINDOWS]

    rows, dels = [], []
    for w in built:
        base = len(rows)
        r = column_rows(w, ex, reg)
        for x in r:
            x["i"] += base
        d = deletion_events(w, r, ex, bounds)
        for x in d:
            x["after_column"] = (x["after_column"] + base) if \
                x["after_column"] is not None and x["after_column"] >= 0 else base - 1
        rows += r
        dels += d

    # ---- three conditions, pooled over the two windows
    ref_all = [t for w in built for t in w["ref"]]
    conds = []
    for name, key, label in (
            ("C1", "w_c1", "Το καθιερωμένο W, με τις παλιές υποθέσεις του benchmark"),
            ("C2", "w_c2", "Η ίδια ψηφοφορία, με τη νέα αποκωδικοποίηση Whisper. "
                           "Χρόνοι και διαρισμός αγνοούνται."),
            ("C3", "w_c3", "Η C2 συν τον παγωμένο κανόνα επαναφοράς που κοιτά "
                           "χρόνο και διαρισμό.")):
        toks = [t for w in built for t in w[key]]
        conds.append({"name": name, "label": label,
                      "stats": CD.sdi(ref_all, toks),
                      "n_eligible": sum(w["c3_eligible"] for w in built)
                      if name == "C3" else None,
                      "n_changed": sum(len(w["c3_changes"]) for w in built)
                      if name == "C3" else None,
                      "changes": [c for w in built for c in w["c3_changes"]]
                      if name == "C3" else [],
                      "per_window": [CD.sdi(w["ref"], w[key]) for w in built]})

    per_system = {}
    for s in SYSTEMS + ["W"]:
        hyp = [t for w in built
               for t in (w["w_c1"] if s == "W"
                         else w["hyps_c1"][SYSTEMS.index(s)])]
        a = RA.align_to_reference(s, ref_all, hyp)
        per_system[s] = {"counts": a.counts, "distance": a.distance,
                         "n_hyp": a.n_hyp, "wer": a.distance / len(ref_all)}

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "city": CO.CITY, "meeting": CO.MEETING,
        "windows": [{"item_id": w.item_id, "start": w.start, "duration": w.duration}
                    for w in CO.WINDOWS],
        "page_start_abs": CO.T0, "page_end_abs": CO.T1,
        "page_duration": CO.PAGE_DURATION, "seam_abs": list(CO.SEAM),
        "bench_run": RUN_ID, "trio": TRIO,
        "soniox_timestamp_model": "stt-rt-v4 (NOT the benchmark's stt-async-v5)",
        "whisper_timestamp_decode": "conf-substrate-2026-08/decode-rw.json "
                                    "(word_timestamps=True; not the benchmark decode)",
        "msa_sha256_16": msa_hash,
        "scoring_normalisation": "eval/controlled_eval/scoring.py wtoks "
                                 "(NFD, strip Mn, lower, \\w+)",
        "thresholds": {"time_tolerance_s": TM.TOLERANCE,
                       "max_bracket_s": TM.MAX_BRACKET,
                       "max_extrapolation_s": TM.MAX_EXTRAPOLATION,
                       "ambiguous_speaker_share": SP.AMBIGUOUS_SHARE,
                       "c3_conf_min": CD.CONF_MIN,
                       "diarization_pad_s": DZ.PAD},
        "pyannote": {"model": "precision-2", "exclusive": True,
                     "padded_span_abs": [CO.T0 - DZ.PAD, CO.T1 + DZ.PAD],
                     "padded_seconds": CO.PAGE_DURATION + 2 * DZ.PAD,
                     "smoke_seconds": 30.0,
                     "eur_per_hour": DZ.EUR_PER_HOUR,
                     "eur_spent": round(DZ.cost_eur(CO.PAGE_DURATION
                                                    + 2 * DZ.PAD + 30.0), 5)},
        "audio_check": json.loads((out / "audio_check.json").read_text()),
        "transfer_stats": {w["item_id"]: w["transfer_stats"] for w in built},
    }

    data = {
        "manifest": manifest,
        "diar": {"exclusive": [{**s, "page_start": CO.to_page(s["start"]),
                                "page_end": CO.to_page(s["end"])} for s in ex],
                 "regular": [{**s, "page_start": CO.to_page(s["start"]),
                              "page_end": CO.to_page(s["end"])} for s in reg],
                 "speakers": sorted({s["speaker"] for s in ex})},
        "rows": rows,
        "deletions": dels,
        "conditions": conds,
        "per_system": per_system,
        "seam_page": [CO.to_page(CO.SEAM[0]), CO.to_page(CO.SEAM[1])],
    }
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False))
    print(json.dumps({"rows": len(rows), "deletions": len(dels),
                      "msa_sha256_16": msa_hash,
                      "conditions": [{c["name"]: c["stats"]} for c in conds]},
                     ensure_ascii=False, indent=1))
    return data


if __name__ == "__main__":
    main()
