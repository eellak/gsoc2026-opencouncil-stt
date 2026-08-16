"""Does the deletion-hard training supply teach omission? (issue #23, part 1-4)

For every `deletion_hard` row of the RUN1/RUN2 stage-2 training manifest we ask one
question with Soniox as an independent witness: **how much of the speech inside the
clip has no corresponding word in the training target?** If the answer is "a lot",
the mix literally taught the model to stay silent over audio.

Soniox was already run on every gap-verify row (exact clip + a 3 s-padded clip) and
only its TEXT was kept -- the wrapper emits no word timestamps and re-running it
would cost money, which this audit is not allowed to spend. So seconds are a
proxy: uncovered Soniox tokens x (clip duration / Soniox token count). The token
FRACTION is the primary statistic; seconds are the ticket's unit and are reported
next to it.

The same pipeline runs over the 300 audited `backbone` rows as a comparison group.
Backbone rows were selected for cleanliness, so they are a floor, not a fair
control: a deletion-hard number close to backbone is strong evidence of no problem,
a number far above it is only suggestive.

Writes JSON (counts and quantiles only -- never transcript text) to
~/.cache/oc-public/deletion-hard-audit/coverage.json

    python3 scripts/analysis/deletion_hard_coverage_audit.py
"""
from __future__ import annotations

import difflib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

CACHE = Path.home() / ".cache/oc-public"
MANIFEST = CACHE / "train-screens-2026-08/run1/manifest.jsonl"
GV = CACHE / "gap-verify"
GV_MANIFEST = GV / "manifest.jsonl"
ASR = GV / "asr"
BACKBONE = CACHE / "backbone-audit/manifest.jsonl"
BASELINE = CACHE / "export-snapshots/export-baseline-pre-gap-reviews-2026-08-13.jsonl"
OUT = CACHE / "deletion-hard-audit/coverage.json"

# --- normalization: identical to eval/gap2_verify_audio.py, which produced the
# found_frac numbers the training supply was selected on. Do not "improve" it.
FILLER_RE = re.compile(r"^(ε{2,}|μ{2,}|α{2,}|ο{3,}|χμ+|εμ+|μχ+|χ{2,})$")
EXTRA_FILLERS = {"ε"}
TOK_RE = re.compile(r"\w+")

UNCOVERED_SEC_GATE = 1.0  # the ticket's frozen threshold


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def ntoks(text: str, keep_fillers: bool = False) -> list[str]:
    out = []
    for m in TOK_RE.finditer(text or ""):
        t = norm(m.group())
        if not t:
            continue
        if not keep_fillers and (FILLER_RE.match(t) or t in EXTRA_FILLERS):
            continue
        out.append(t)
    return out


def dl_leq1(a: str, b: str) -> bool:
    """Damerau-Levenshtein distance <= 1 (same rule as the verifier's found_frac)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if a == b:
        return True
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        if len(diff) == 2 and diff[1] == diff[0] + 1:
            i = diff[0]
            return a[i] == b[i + 1] and a[i + 1] == b[i]
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def canonicalize(src: list[str], ref: list[str]) -> list[str]:
    """Fold ASR spelling variants onto the target's own spelling.

    Without this, `difflib` would score every Soniox mis-spelling of a target word
    as uncovered speech, which is exactly the artefact this audit must not create.
    """
    # sorted, not set-ordered: with several DL<=1 candidates `next` must pick the
    # same one on every run, and set iteration order moves with PYTHONHASHSEED.
    long_ref = sorted({t for t in ref if len(t) >= 4})
    out = []
    for t in src:
        if t in ref:
            out.append(t)
            continue
        if len(t) >= 5:
            hit = next((r for r in long_ref if dl_leq1(t, r)), None)
            if hit:
                out.append(hit)
                continue
        out.append(t)
    return out


def coverage(target: str, soniox: str, dur: float, keep_fillers: bool = False
             ) -> dict | None:
    """Soniox tokens with no counterpart in the target, in tokens and in seconds.

    Two bounds, because an equal-length `replace` block is genuinely ambiguous --
    it may be an ASR mis-hearing of a word the target has (covered) or entirely
    different speech (uncovered):

      lower  insert opcodes + the surplus of replace opcodes  (primary)
      upper  every Soniox token not in an `equal` opcode

    Both are reported everywhere. Picking whichever one clears the gate after the
    fact is exactly the move this audit must not make.
    """
    tgt = ntoks(target, keep_fillers)
    son_raw = ntoks(soniox, keep_fillers)
    if not son_raw:
        return None
    son = canonicalize(son_raw, tgt)
    sm = difflib.SequenceMatcher(a=tgt, b=son, autojunk=False)
    lower = upper = 0
    runs_lo, runs_hi = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            n = j2 - j1
            lower += n
            upper += n
            runs_lo.append(n)
            runs_hi.append(n)
        elif tag == "replace":
            net = (j2 - j1) - (i2 - i1)
            if net > 0:
                lower += net
                runs_lo.append(net)
            upper += j2 - j1
            runs_hi.append(j2 - j1)
    n_son = len(son)
    per_tok = dur / n_son if n_son else 0.0
    return {
        "n_target_tokens": len(tgt),
        "n_soniox_tokens": n_son,
        "dur": round(dur, 3),
        "sec_per_token": round(per_tok, 4),
        "uncovered_tokens": lower,
        "uncovered_frac": round(lower / n_son, 4),
        "uncovered_sec": round(lower * per_tok, 3),
        "uncovered_tokens_upper": upper,
        "uncovered_sec_upper": round(upper * per_tok, 3),
        "longest_run_tokens": max(runs_lo) if runs_lo else 0,
        "longest_run_sec": round((max(runs_lo) if runs_lo else 0) * per_tok, 3),
        "longest_run_sec_upper": round((max(runs_hi) if runs_hi else 0) * per_tok, 3),
    }


def edges(target: str, exact: str, padded: str) -> dict:
    """Does the clip cut its own first/last reference word?

    A target edge word that Soniox does not hear inside the exact clip but DOES
    hear once 3 s of context is added on both sides is a word the clip boundary
    sliced. The padded clip also carries genuine neighbour speech, so this test is
    only run on the target's own first and last token -- never on Soniox surplus.
    """
    tgt = ntoks(target)
    ex_t, pa_t = ntoks(exact), ntoks(padded)
    if not tgt or not ex_t or not pa_t:
        return {"head_cut": None, "tail_cut": None}

    def heard(tok: str, pool) -> bool:
        pool = set(pool)
        if tok in pool:
            return True
        return len(tok) >= 5 and any(dl_leq1(tok, t) for t in pool if len(t) >= 4)

    # Locate the exact clip inside the padded transcript, so that the 3 s of
    # genuine neighbour speech is separated from the clip's own content. Without
    # this, any word the neighbour happens to say would read as a cut edge.
    blocks = [b for b in difflib.SequenceMatcher(
        a=ex_t, b=pa_t, autojunk=False).get_matching_blocks() if b.size]
    if not blocks:
        return {"head_cut": None, "tail_cut": None}
    lo = blocks[0].b
    hi = blocks[-1].b + blocks[-1].size
    return {
        "head_cut": (not heard(tgt[0], ex_t)) and heard(tgt[0], pa_t[:lo]),
        "tail_cut": (not heard(tgt[-1], ex_t)) and heard(tgt[-1], pa_t[hi:]),
    }


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)

    def q(p: float) -> float:
        return round(s[min(len(s) - 1, int(p * len(s)))], 3)

    return {"n": len(s), "min": round(s[0], 3), "p10": q(0.10), "p25": q(0.25),
            "median": q(0.50), "p75": q(0.75), "p90": q(0.90), "p95": q(0.95),
            "p99": q(0.99), "max": round(s[-1], 3),
            "mean": round(statistics.fmean(s), 3)}


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def soniox_text(uid: str, kind: str) -> str | None:
    p = ASR / f"{uid}.{kind}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("soniox_text") if d.get("asr_ok") else None


def main() -> None:
    buckets: dict[str, list[dict]] = {}
    for r in read_jsonl(MANIFEST):
        buckets.setdefault(r["bucket"], []).append(r)
    dh = buckets["deletion_hard"]

    gv = {r["id"]: r for r in read_jsonl(GV_MANIFEST)}
    baseline_includes = {
        r["utterance_id"] for r in read_jsonl(BASELINE)
        if r.get("include_status") == "include"
    }

    per_row = []
    no_witness = 0
    for r in dh:
        uid = r["id"]
        g = gv.get(uid)
        ex = soniox_text(uid, "exact")
        pa = soniox_text(uid, "pad")
        if g is None or ex is None:
            no_witness += 1
            continue
        cov = coverage(r["text"], ex, r["dur"])
        if cov is None:
            no_witness += 1
            continue
        ff = g.get("found_frac")
        n_added = g.get("n_added") or 0
        # Reconstructed provenance. The auto-accept rules recorded on 2026-08-13/14
        # were found_frac>=0.85 AND n_added>=5, then found_frac>=0.85 AND n_added 2-4.
        # A row already an include BEFORE the wave was reviewed by a human.
        pre_existing = uid in baseline_includes
        rule_passer = (ff is not None and ff >= 0.85 and n_added >= 2)
        stratum = ("manual_pre_wave" if pre_existing
                   else "auto_verifier" if rule_passer
                   else "manual_wave")
        e = edges(r["text"], ex, pa or "")
        per_row.append({**cov, **e, "id": uid, "stratum": stratum, "tier": g.get("tier"),
                        "found_frac": ff, "n_added": n_added,
                        "city_id": r.get("city_id")})

    # --- comparison group: the 300 audited backbone rows, identical pipeline
    bb = []
    for r in read_jsonl(BACKBONE):
        if r.get("error") or not r.get("soniox_exact"):
            continue
        cov = coverage(r["text"], r["soniox_exact"], r["duration_s"])
        if cov is None:
            continue
        e = edges(r["text"], r["soniox_exact"], r.get("soniox_padded") or "")
        bb.append({**cov, **e})

    def summarize(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        n_edge = sum(x.get("head_cut") is not None for x in rs)
        secs = [x["uncovered_sec"] for x in rs]
        secs_hi = [x["uncovered_sec_upper"] for x in rs]
        fracs = [x["uncovered_frac"] for x in rs]
        runs = [x["longest_run_sec"] for x in rs]
        return {
            "n": len(rs),
            "uncovered_sec": quantiles(secs),
            "uncovered_sec_upper": quantiles(secs_hi),
            "uncovered_frac": quantiles(fracs),
            "longest_run_sec": quantiles(runs),
            "uncovered_sec_dose": round(
                sum(secs) / sum(x["dur"] for x in rs), 4),
            "share_uncovered_sec_gt_1.0": round(
                sum(s > UNCOVERED_SEC_GATE for s in secs) / len(rs), 4),
            "share_uncovered_sec_upper_gt_1.0": round(
                sum(s > UNCOVERED_SEC_GATE for s in secs_hi) / len(rs), 4),
            "share_longest_run_gt_1.0s": round(
                sum(s > UNCOVERED_SEC_GATE for s in runs) / len(rs), 4),
            "share_uncovered_frac_gt_0.5": round(
                sum(f > 0.5 for f in fracs) / len(rs), 4),
            "share_zero_uncovered": round(
                sum(x["uncovered_tokens"] == 0 for x in rs) / len(rs), 4),
            # Edge rates are over TESTABLE rows only. The test needs both a Soniox
            # exact and a Soniox padded transcript, and the backbone audit kept the
            # padded one for a minority of its rows -- counting those as "not cut"
            # would silently deflate the comparison group.
            "n_edge_testable": n_edge,
            "edge_test_unavailable": len(rs) - n_edge,
            "head_cut": round(sum(bool(x.get("head_cut")) for x in rs) / n_edge, 4)
                        if n_edge else None,
            "tail_cut": round(sum(bool(x.get("tail_cut")) for x in rs) / n_edge, 4)
                        if n_edge else None,
            "either_edge_cut": round(
                sum(bool(x.get("head_cut")) or bool(x.get("tail_cut")) for x in rs)
                / n_edge, 4) if n_edge else None,
            "median_dur": round(statistics.median(x["dur"] for x in rs), 3),
        }

    strata = {}
    for name in ("auto_verifier", "manual_pre_wave", "manual_wave"):
        strata[name] = summarize([x for x in per_row if x["stratum"] == name])

    # --- size of the correction: drop rows over the gate, re-derive the mixture
    n_total_dh = len(dh)
    flagged = [x for x in per_row if x["uncovered_sec"] > UNCOVERED_SEC_GATE]
    n_flag = len(flagged)
    measured = len(per_row)
    # optimistic: unmeasured rows are clean. pessimistic: they fail at the measured rate.
    proj_pess = n_flag + (n_total_dh - measured) * (n_flag / measured if measured else 0)

    def remix(drop: float) -> dict:
        """Realized presentation shares if `drop` deletion-hard rows are removed.

        The sampler caps deletion_hard at 2 presentations/row/epoch, so its
        presentation budget scales with its row count; the shortfall goes to
        backbone by the recorded cap_shortfall_rule.
        """
        meta = json.loads((MANIFEST.parent / "meta.json").read_text())
        b = dict(meta["presentation_budget_per_epoch"])
        cap = meta["presentation_cap_per_row_per_epoch"]["deletion_hard"]
        lost = min(b["deletion_hard"], drop * cap)
        b["deletion_hard"] -= lost
        b["backbone"] += lost
        tot = sum(b.values())
        return {k: round(v / tot, 4) for k, v in b.items()}

    # --- what the unwitnessed rows look like, on covariates that need no witness.
    # The gate is strongly duration-dependent (a 1.0 s absolute gap is nearly
    # impossible in a 1.5 s clip), so a duration-matched imputation is the least
    # unreasonable way to say something about them. It is an assumption, not a
    # measurement, and it is reported as such.
    dur_bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    measured_ids = {x["id"] for x in per_row}
    unwit = [r for r in dh if r["id"] not in measured_ids]
    rate_by_bin, imputed = {}, 0.0
    for lo, hi in dur_bins:
        sel = [x for x in per_row if lo <= x["dur"] < hi]
        rate = (sum(x["uncovered_sec"] > UNCOVERED_SEC_GATE for x in sel) / len(sel)
                if sel else 0.0)
        rate_by_bin[f"[{lo},{hi})"] = {"n_witnessed": len(sel), "rate": round(rate, 4),
                                       "n_unwitnessed": sum(
                                           1 for r in unwit if lo <= r["dur"] < hi)}
        imputed += rate * rate_by_bin[f"[{lo},{hi})"]["n_unwitnessed"]

    # --- sensitivity: keep hesitation fillers. A filler the target drops is itself
    # a short stretch the model is trained to stay silent through.
    with_fillers = []
    for r in dh:
        if r["id"] not in measured_ids:
            continue
        ex = soniox_text(r["id"], "exact")
        if ex is None:
            continue
        c = coverage(r["text"], ex, r["dur"], keep_fillers=True)
        if c:
            with_fillers.append(c)

    out = {
        "generated_for": "github issue #23 / exp-2026-08-13-targeted-deletion-training",
        "inputs": {
            "train_manifest": str(MANIFEST),
            "gap_verify_manifest": str(GV_MANIFEST),
            "backbone_audit": str(BACKBONE),
            "baseline_export": str(BASELINE),
        },
        "method": {
            "seconds_proxy": "uncovered_tokens * clip_duration / n_soniox_tokens "
                             "(no word timestamps exist; Soniox was text-only)",
            "primary_statistic": "uncovered token fraction; seconds reported for the "
                                 "ticket's 1.0 s gate",
            "uncovered": "difflib insert opcodes + replace surplus, after folding "
                         "Soniox spellings onto target tokens (DL<=1, len>=5)",
            "witness": "Soniox on the EXACT clip span only; the padded clip is used "
                       "only for the edge test",
        },
        "deletion_hard": {
            "rows_total": n_total_dh,
            "rows_measured": measured,
            "rows_without_witness": no_witness,
            "measured_share": round(measured / n_total_dh, 4),
            "all_measured": summarize(per_row),
            "by_stratum": strata,
            "by_tier": {t: summarize([x for x in per_row if x["tier"] == t])
                        for t in sorted({x["tier"] for x in per_row if x["tier"]})},
        },
        "backbone_comparison": summarize(bb) | {
            "caveat": "These 300 rows were SELECTED for cleanliness by the backbone "
                      "audit, and their median duration is far shorter. They are a "
                      "false-positive floor for the pipeline, not an unbiased "
                      "estimate of ordinary backbone coverage. Do not read the "
                      "contrast causally.",
        },
        "unwitnessed_rows": {
            "n": len(unwit),
            "median_dur": round(statistics.median([r["dur"] for r in unwit]), 3)
                          if unwit else None,
            "hours": round(sum(r["dur"] for r in unwit) / 3600, 3),
            "what_they_are": "deletion-hard rows that were already an `include` "
                             "before the gap wave, i.e. the HUMAN-reviewed stratum. "
                             "Soniox was never run on them, so this audit measures "
                             "the auto-verified stratum and cannot measure the "
                             "human one.",
        },
        "gate_rate_by_duration": rate_by_bin,
        "sensitivity_fillers_kept": summarize(with_fillers),
        "threshold": {
            "rule": ">=15% of deletion-hard rows with uncovered speech > 1.0 s",
            "witnessed_prevalence": round(n_flag / measured, 4) if measured else None,
            "cohort_lower_bound": round(n_flag / n_total_dh, 4),
            "cohort_upper_bound": round(
                (n_flag + (n_total_dh - measured)) / n_total_dh, 4),
            "cohort_mar_estimate": round(proj_pess / n_total_dh, 4),
            "cohort_duration_matched_estimate": round(
                (n_flag + imputed) / n_total_dh, 4),
            "rows_needed_for_a_guaranteed_cohort_pass": math.ceil(0.15 * n_total_dh),
            "positives_needed_among_unwitnessed": max(
                0, math.ceil(0.15 * n_total_dh) - n_flag),
            "implied_unwitnessed_rate_for_a_pass": round(
                max(0, math.ceil(0.15 * n_total_dh) - n_flag)
                / max(1, n_total_dh - measured), 4),
            "implied_unwitnessed_rate_duration_matched": round(
                imputed / max(1, n_total_dh - measured), 4),
            "upper_bound_note": "The unwitnessed rows are unmeasurable, so a cohort "
                                "FAILURE is never provable from these data: the "
                                "upper bound is above 15% by construction. Read the "
                                "witnessed prevalence and the duration-matched "
                                "estimate; 'gate not established' is not 'no gap "
                                "exists'.",
            "upper_bound_sensitivity": round(
                sum(x["uncovered_sec_upper"] > UNCOVERED_SEC_GATE for x in per_row)
                / measured, 4) if measured else None,
        },
        "correction_size": {
            "rows_dropped_measured": n_flag,
            "rows_dropped_projected_to_full_bucket": round(proj_pess),
            "mixture_now": json.loads((MANIFEST.parent / "meta.json").read_text())
                           ["realized_shares_per_epoch"],
            "mixture_after_drop_measured": remix(n_flag),
            "mixture_after_drop_projected": remix(proj_pess),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
