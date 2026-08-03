#!/usr/bin/env python3
"""Score the paired synthetic-overlap run against the frozen preregistration.

`docs/specs/synthetic-overlap-preregistration.md` fixes the estimand, the primary arm,
the endpoints and the DiCoW gate. This file computes them and nothing else; if a number
here is not in that document, it is exploratory and is labelled so in the output.

Aggregation is the word-weighted paired error-count difference over total reference
words, clustered on the TARGET meeting. Mean-of-per-window-WER would overweight the
short windows.

Usage:
  SC=~/.cache/oc-overlap python eval/controlled_eval/synth_overlap_analyze.py
Env: SC HYPS N_BOOT
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as S  # noqa: E402

ROOT = Path("/home/harold/opencouncil-fine-tuning")
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
HYPS = Path(os.environ.get("HYPS", SC / "synth_overlap_hyps.json"))
N_BOOT = int(os.environ.get("N_BOOT", "10000"))

PRIMARY_SYS = "finetune"
PRIMARY_ARM = "C"
GATE_BURDEN = 0.020        # C - A, absolute WER
GATE_SPEECH = 0.010        # C - E, absolute WER, one-sided 90% LB > 0


def log(*a):
    print(*a, flush=True)


def text_of(v):
    return v["text"] if isinstance(v, dict) else (v or "")


def counts_pair(items, hyps, sysname, arm, discount=None):
    """(errors, ref_words) per item for one arm, in a fixed item order."""
    out = []
    for it in items:
        h = text_of(hyps[f"{sysname}|{it['item_id']}|{arm}"])
        ht = S.wtoks(h)
        if discount is not None:
            ht = discount(it, ht)
        rt = S.wtoks(it["ref"])
        out.append((S.edist(rt, ht), len(rt)))
    return out


def sdi(ref, hyp):
    """Alignment decomposition. NOT a source attribution — see the preregistration."""
    a, b = S.wtoks(ref), S.wtoks(hyp)
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, s, dl, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            s += a[i - 1] != b[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dl, ins


def donor_discount(hyps, sysname):
    """Drop hypothesis tokens the donor itself said, before scoring the target.

    An OPTIMISTIC bound on target-speaker damage, never ground truth: simultaneous speech
    has no unique serialisation and two speakers can say the same word, so this can also
    delete words the target really said. Reported alongside the strict number, and the
    gap between them is what says whether the system is emitting the interjector.
    """
    def f(it, ht):
        d = hyps.get(f"{sysname}|{it['item_id']}|donor")
        if d is None:
            return ht
        pool = collections.Counter(S.wtoks(text_of(d)))
        keep = []
        for t in ht:
            if pool[t] > 0:
                pool[t] -= 1
            else:
                keep.append(t)
        return keep
    return f


def spillover(items, hyps, sysname, arm, pad=1.0):
    """How much of the change sits OUTSIDE the inserted event.

    Compares arm-vs-clean hypothesis text restricted to segments that do not touch the
    event window. No reference is involved: this asks whether Whisper's segmentation and
    context carried the disturbance into audio that is bit-identical between the arms.
    """
    tot_out = tot_words = 0
    for it in items:
        a = hyps.get(f"{sysname}|{it['item_id']}|A")
        b = hyps.get(f"{sysname}|{it['item_id']}|{arm}")
        if not isinstance(a, dict) or not isinstance(b, dict):
            return None
        s0 = it["event_start_sec"] - pad
        s1 = it["event_start_sec"] + it["event_dur_sec"] + pad
        def outside(v):
            return S.wtoks(" ".join(t for st, en, t in v["segments"] if en <= s0 or st >= s1))
        ao, bo = outside(a), outside(b)
        tot_out += S.edist(ao, bo)
        tot_words += len(ao)
    return {"changed_words_outside_event": tot_out, "words_outside_event": tot_words,
            "frac": tot_out / tot_words if tot_words else None}


def contrast(items, hyps, sysname, arm_a, arm_b, discount=None, label=""):
    ca = counts_pair(items, hyps, sysname, arm_a, discount)
    cb = counts_pair(items, hyps, sysname, arm_b, discount)
    clusters = [it["meeting_id"] for it in items]
    r = S.cluster_bootstrap(ca, cb, clusters, n_boot=N_BOOT)
    r["label"] = label or f"{arm_a} - {arm_b}"
    r["wer_a"], r["wer_b"] = S.wer(ca), S.wer(cb)
    return r


def one_sided_lb90(items, hyps, sysname, arm_a, arm_b):
    """One-sided 90% lower bound on WER(a) - WER(b), meeting-clustered."""
    ca = np.array(counts_pair(items, hyps, sysname, arm_a), dtype=float)
    cb = np.array(counts_pair(items, hyps, sysname, arm_b), dtype=float)
    groups = collections.defaultdict(list)
    for i, it in enumerate(items):
        groups[it["meeting_id"]].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(7)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        den = ca[idx, 1].sum()
        diffs[i] = (ca[idx, 0].sum() - cb[idx, 0].sum()) / den if den else np.nan
    return float(np.nanpercentile(diffs, 10))


def main():
    manifest = json.loads((SC / "synth_overlap_manifest.json").read_text())
    hyps = json.loads(HYPS.read_text())
    systems = sorted({k.split("|")[0] for k in hyps})

    items = [it for it in manifest["items"]
             if all(f"{s}|{it['item_id']}|{a}" in hyps
                    for s in systems for a in ("A", "B", "C", "D", "E", "F")
                    if not (a in ("G", "H") and s != PRIMARY_SYS))]
    log(f"{len(items)} items complete in every arm and system "
        f"({len(manifest['items'])} built)")

    n_target_meetings = len({it["meeting_id"] for it in items})
    n_donor_meetings = len({it["donor_meeting"] for it in items})

    out = {
        "preregistration": "docs/specs/synthetic-overlap-preregistration.md",
        "n_items": len(items), "n_target_meetings": n_target_meetings,
        "n_donor_meetings": n_donor_meetings, "n_boot": N_BOOT,
        "cluster_warning": (
            "Fewer than 20 independent target meetings: the clustered bootstrap is "
            "unstable and these intervals are indicative only."
            if n_target_meetings < 20 else None),
        "claim_limits": (
            "Causal for THIS additive-speech intervention on these windows, not for "
            "natural crosstalk. Two Whisper-family systems only."),
        "systems": {},
    }

    for sysname in systems:
        arms = ["B", "C", "D", "E", "F"] + (["G", "H"] if sysname == PRIMARY_SYS else [])
        block = {
            "wer_clean": S.wer(counts_pair(items, hyps, sysname, "A")),
            "vs_clean": {a: contrast(items, hyps, sysname, a, "A", label=f"{a} - A")
                         for a in arms},
            "speech_specific_C_minus_E": contrast(items, hyps, sysname, "C", "E",
                                                  label="C - E"),
            "voice_vs_reversed_C_minus_F": contrast(items, hyps, sysname, "C", "F",
                                                    label="C - F"),
            "donor_aware_bound": {
                a: contrast(items, hyps, sysname, a, "A",
                            discount=donor_discount(hyps, sysname),
                            label=f"{a} - A, donor words discounted")
                for a in ("C",)},
            "sdi_per_arm": {},
            "spillover_C": spillover(items, hyps, sysname, "C"),
        }
        for a in ["A"] + arms:
            tot = [0, 0, 0]
            for it in items:
                s, d, i = sdi(it["ref"], text_of(hyps[f"{sysname}|{it['item_id']}|{a}"]))
                tot[0] += s
                tot[1] += d
                tot[2] += i
            n = sum(len(S.wtoks(it["ref"])) for it in items)
            block["sdi_per_arm"][a] = {"sub": tot[0] / n, "del": tot[1] / n,
                                       "ins": tot[2] / n,
                                       "note": "alignment decomposition, not source attribution"}
        out["systems"][sysname] = block

    # ---------------------------------------------------------------- the frozen gate
    p = out["systems"].get(PRIMARY_SYS)
    if p:
        burden = p["vs_clean"][PRIMARY_ARM]["delta"]
        speech = p["speech_specific_C_minus_E"]["delta"]
        lb = one_sided_lb90(items, hyps, PRIMARY_SYS, "C", "E")
        out["gate"] = {
            "system": PRIMARY_SYS, "arm": PRIMARY_ARM,
            "burden_C_minus_A": burden, "threshold": GATE_BURDEN,
            "speech_specific_C_minus_E": speech, "threshold_speech": GATE_SPEECH,
            "one_sided_lb90_C_minus_E": lb,
            "pass": bool(burden >= GATE_BURDEN and speech >= GATE_SPEECH and lb > 0),
            "note": ("A pass still requires a frozen pretrained separator to recover "
                     ">=25% of the added errors before DiCoW is trained."),
        }

    dst = ROOT / "eval/controlled_eval/results_synth_overlap.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"-> {dst}")
    if p:
        g = out["gate"]
        log(f"gate: burden {g['burden_C_minus_A']:+.4f} (need >= {GATE_BURDEN}), "
            f"speech-specific {g['speech_specific_C_minus_E']:+.4f} "
            f"(need >= {GATE_SPEECH}, LB90 {g['one_sided_lb90_C_minus_E']:+.4f}) "
            f"=> {'PASS' if g['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
