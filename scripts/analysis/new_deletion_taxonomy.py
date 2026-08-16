"""Where do the screens' NEW deletions fall? (issue #23, part 5)

Both training screens deleted more reference tokens than the control on the 39
frozen validation windows. This puts every reference token the arm deletes and the
control does not into one of four pre-specified buckets:

    name      the token belongs to a DS-WER v2 `entities` term for that city
    edge      the token sits at a mechanical boundary: the first/last 2 s of the
              window, or within 0.5 s of a >=0.8 s silence gap in pyannote's
              diarization (the decoder's own segment boundaries are not recorded,
              and silence gaps are where it puts them)
    overlap   pyannote places >=2 simultaneous speakers at the token's time
    ordinary  none of the above

Priority is name > edge > overlap > ordinary, fixed before the counts were read.
A category is only "mechanically explainable" if it beats what the same rule
assigns to the reference as a whole; the per-category BASE RATE over all reference
tokens is reported next to every share, and so is the control's own deletion mix.

Reference tokens are placed in time by aligning them to the cached
whisper-turbo word-level transcription (`$SC/whisper_turbo/<window>.json`) and
interpolating the positions that do not anchor.

Writes ~/.cache/oc-public/deletion-hard-audit/new-deletions.json (counts only).

    SC=~/.cache/oc-public .venv-eval/bin/python scripts/analysis/new_deletion_taxonomy.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.scoring import wtoks  # noqa: E402
from scripts.ds_wer import TermList  # noqa: E402

CACHE = Path(os.environ.get("SC", str(Path.home() / ".cache/oc-public")))
MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"
OUT = CACHE / "deletion-hard-audit/new-deletions.json"

ARMS = {
    "control": CACHE / "decode-ablation/eval-A.json",
    "run1": CACHE / "train-screens-2026-08/run1-eval/decode.json",
    "run2_stage2": CACHE / "train-screens-2026-08/run2-eval-stage2/decode.json",
}

FILLER_RE = re.compile(r"^(ε{2,}|μ{2,}|α{2,}|ο{3,}|χμ+|εμ+|μχ+|χ{2,})$")

EDGE_MARGIN_S = 2.0     # window head/tail
GAP_MIN_S = 0.8         # a silence this long is a plausible segment boundary
GAP_NEAR_S = 0.5        # ...and a token this close to one sits on that boundary


def ftoks(text: str) -> list[str]:
    """The frozen eval tokenizer (eval_freeze.ftoks)."""
    return [t for t in wtoks(text) if not FILLER_RE.match(t)]


# ----------------------------------------------------------------- alignment
def ref_ops(ref: list[str], hyp: list[str], prefer_del: bool = False) -> list[str]:
    """Per-reference-token operation from a Levenshtein backtrace: '=', 'S', 'D'.

    Deletion status is read off ONE alignment per arm. Ties in the DP are broken
    in a fixed order (match, then substitute, then delete, then insert), so two
    arms with identical hypotheses get identical labels.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                          d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops = [""] * n
    i, j = n, m
    while i > 0 and j > 0:
        cost = ref[i - 1] != hyp[j - 1]
        diag = d[i][j] == d[i - 1][j - 1] + cost
        up = d[i][j] == d[i - 1][j] + 1
        # `prefer_del` flips the only arbitrary choice in the backtrace, so the
        # audit can report how much of its answer is an alignment tie-break.
        if diag and not (prefer_del and up):
            ops[i - 1] = "S" if cost else "="
            i, j = i - 1, j - 1
        elif up:
            ops[i - 1] = "D"
            i -= 1
        else:
            j -= 1
    while i > 0:
        ops[i - 1] = "D"
        i -= 1
    return ops


# ------------------------------------------------------------------- timing
def turbo(wid: str) -> dict | None:
    p = CACHE / "whisper_turbo" / f"{wid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["output"]


def turbo_stream(pk: dict) -> tuple[list[str], list[float]]:
    toks, times = [], []
    for w in pk["wordLevelTranscription"]:
        tt = ftoks(w["text"])
        if not tt:
            continue
        span = (float(w["end"]) - float(w["start"])) / len(tt)
        for n, t in enumerate(tt):
            toks.append(t)
            times.append(float(w["start"]) + (n + 0.5) * span)
    return toks, times


def ref_times(ref: list[str], pk: dict, dur: float) -> tuple[list[float], list[bool]]:
    """A time for every reference token, plus whether it was really anchored."""
    ptoks, ptimes = turbo_stream(pk)
    times: list[float | None] = [None] * len(ref)
    if ptoks:
        for tag_, i1, i2, j1, j2 in SequenceMatcher(a=ref, b=ptoks,
                                                    autojunk=False).get_opcodes():
            if tag_ == "equal":
                for k in range(i2 - i1):
                    times[i1 + k] = ptimes[j1 + k]
    anchored = [t is not None for t in times]
    known = [(i, t) for i, t in enumerate(times) if t is not None]
    if not known:
        step = dur / max(1, len(ref))
        return [(i + 0.5) * step for i in range(len(ref))], anchored
    # linear interpolation between anchors, flat extrapolation at the ends
    out = []
    for i in range(len(ref)):
        if times[i] is not None:
            out.append(times[i])
            continue
        left = max((k for k, _ in known if k < i), default=None)
        right = min((k for k, _ in known if k > i), default=None)
        if left is None:
            out.append(times[right] * (i + 1) / (right + 1))
        elif right is None:
            lt = times[left]
            out.append(min(dur, lt + (i - left) * 0.3))
        else:
            lt, rt = times[left], times[right]
            out.append(lt + (rt - lt) * (i - left) / (right - left))
    return out, anchored


def overlap_intervals(pk: dict) -> list[tuple[float, float]]:
    """Times where >=2 DISTINCT pyannote speakers are active."""
    ev = []
    for s in pk["diarization"]:
        ev.append((float(s["start"]), 1, s["speaker"]))
        ev.append((float(s["end"]), -1, s["speaker"]))
    ev.sort(key=lambda e: (e[0], e[1]))
    active: dict[str, int] = {}
    out, start = [], None
    for t, delta, spk in ev:
        n_before = sum(1 for v in active.values() if v > 0)
        active[spk] = active.get(spk, 0) + delta
        n_after = sum(1 for v in active.values() if v > 0)
        if n_before < 2 <= n_after:
            start = t
        elif n_before >= 2 > n_after and start is not None:
            out.append((start, t))
            start = None
    return out


def silence_gaps(pk: dict, dur: float) -> list[tuple[float, float]]:
    """Gaps with no active speaker, >= GAP_MIN_S long."""
    segs = sorted((float(s["start"]), float(s["end"])) for s in pk["diarization"])
    merged: list[list[float]] = []
    for a, b in segs:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps, prev = [], 0.0
    for a, b in merged:
        if a - prev >= GAP_MIN_S:
            gaps.append((prev, a))
        prev = b
    if dur - prev >= GAP_MIN_S:
        gaps.append((prev, dur))
    return gaps


def inside(t: float, spans: list[tuple[float, float]], pad: float = 0.0) -> bool:
    return any(a - pad <= t <= b + pad for a, b in spans)


def term_positions(tokens: list[str], terms: TermList) -> list[bool]:
    """Per-position term membership. `ds_wer.tag` collapses a multi-token term into
    one symbol, which is right for DS-WER and wrong here: this audit needs one flag
    per reference token. Same longest-match-wins, leftmost-tie rule."""
    out = [False] * len(tokens)
    i, n = 0, len(tokens)
    while i < n:
        span_hit = 0
        for span in range(min(terms.max_len, n - i), 0, -1):
            if terms.by_alias.get(tuple(tokens[i:i + span])) is not None:
                span_hit = span
                break
        if span_hit:
            for k in range(span_hit):
                out[i + k] = True
            i += span_hit
        else:
            i += 1
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    man = json.loads(MANIFEST.read_text())
    from eval.controlled_eval import bench_data as B
    rep = B.load_report(man["source_run"])
    items = {it["itemId"]: it for it in rep["items"]}

    terms = {c: TermList([t for t in json.loads(
        (TERMS_DIR / f"{c}.v2.json").read_text())["terms"] if t["cut"] == "entities"])
        for c in ("argos", "orestiada")}

    hyps = {}
    for arm, path in ARMS.items():
        st = json.loads(path.read_text())
        hyps[arm] = {w: (v.get("text") or "") for w, v in st["windows"].items()}

    cats = ["name", "edge", "overlap", "ordinary"]
    mech = ["name", "edge", "overlap"]  # `ordinary` is the residual, never a cause
    counts = {a: {c: 0 for c in cats} for a in ARMS if a != "control"}
    control_del = {c: 0 for c in cats}
    arm_del_all = {a: {c: 0 for c in cats} for a in ARMS if a != "control"}
    base = {c: 0 for c in cats}
    recovered = {a: {c: 0 for c in cats} for a in ARMS if a != "control"}
    alt_prec = {a: {c: 0 for c in cats} for a in ARMS if a != "control"}
    alt_tie = {a: {c: 0 for c in cats} for a in ARMS if a != "control"}
    events = {a: {c: 0 for c in cats} for a in ARMS if a != "control"}
    n_new = {a: 0 for a in ARMS if a != "control"}
    per_window_new = {a: {} for a in ARMS if a != "control"}
    no_turbo, anchored_tot, tok_tot = [], 0, 0

    for w in man["eval_windows"]:
        wid, city, dur = w["window_id"], w["city"], w["duration_sec"]
        ref = ftoks(items[wid]["referenceText"])
        pk = turbo(wid)
        if pk is None:
            no_turbo.append(wid)
            times = [(i + 0.5) * dur / max(1, len(ref)) for i in range(len(ref))]
            anch = [False] * len(ref)
            ovl, gaps = [], []
        else:
            times, anch = ref_times(ref, pk, dur)
            ovl, gaps = overlap_intervals(pk), silence_gaps(pk, dur)
        anchored_tot += sum(anch)
        tok_tot += len(ref)

        is_term = term_positions(ref, terms[city])

        def category(i: int) -> str:
            if is_term[i]:
                return "name"
            t = times[i]
            if t <= EDGE_MARGIN_S or t >= dur - EDGE_MARGIN_S:
                return "edge"
            if inside(t, gaps, GAP_NEAR_S):
                return "edge"
            if inside(t, ovl):
                return "overlap"
            return "ordinary"

        def category_alt(i: int) -> str:
            """Codex's precedence (edge > overlap > name), reported as sensitivity."""
            t = times[i]
            if t <= EDGE_MARGIN_S or t >= dur - EDGE_MARGIN_S or inside(t, gaps, GAP_NEAR_S):
                return "edge"
            if inside(t, ovl):
                return "overlap"
            return "name" if is_term[i] else "ordinary"

        cat = [category(i) for i in range(len(ref))]
        cat_alt = [category_alt(i) for i in range(len(ref))]
        for c in cat:
            base[c] += 1

        ops = {a: ref_ops(ref, ftoks(hyps[a][wid])) for a in ARMS}
        ops_alt = {a: ref_ops(ref, ftoks(hyps[a][wid]), prefer_del=True) for a in ARMS}
        for a in counts:
            for i in range(len(ref)):
                if ops[a][i] == "D" and ops["control"][i] != "D":
                    alt_prec[a][cat_alt[i]] += 1
                if ops_alt[a][i] == "D" and ops_alt["control"][i] != "D":
                    alt_tie[a][cat[i]] += 1
            # deletion EVENTS: a run of consecutive new deletions counts once,
            # labelled by its first token, so one omitted phrase is one piece of
            # evidence and not ten.
            prev = False
            for i in range(len(ref)):
                now = ops[a][i] == "D" and ops["control"][i] != "D"
                if now and not prev:
                    events[a][cat[i]] += 1
                prev = now
        for c, o in zip(cat, ops["control"]):
            if o == "D":
                control_del[c] += 1
        for a in counts:
            new = 0
            for i, c in enumerate(cat):
                if ops[a][i] == "D":
                    arm_del_all[a][c] += 1
                    if ops["control"][i] != "D":
                        counts[a][c] += 1
                        new += 1
                elif ops["control"][i] == "D":
                    recovered[a][c] += 1
            n_new[a] += new
            per_window_new[a][wid] = new

    def shares(d: dict[str, int]) -> dict:
        tot = sum(d.values())
        return {c: {"n": d[c], "share": round(d[c] / tot, 4) if tot else None}
                for c in cats} | {"total": tot}

    out = {
        "generated_for": "github issue #23 part 5",
        "windows": len(man["eval_windows"]),
        "priority": "name > edge > overlap > ordinary (fixed before counting)",
        "params": {"edge_margin_s": EDGE_MARGIN_S, "silence_gap_min_s": GAP_MIN_S,
                   "silence_gap_near_s": GAP_NEAR_S,
                   "names": "DS-WER v2 entities cut"},
        "timing_quality": {
            "windows_without_pyannote": no_turbo,
            "ref_tokens": tok_tot,
            "anchored_share": round(anchored_tot / tok_tot, 4) if tok_tot else None,
            "note": "unanchored tokens are interpolated; their time is a guess and "
                    "the edge/overlap split for them is weaker than the name split, "
                    "which uses no timing at all",
        },
        "reference_base_rates": shares(base),
        "control_deletions": shares(control_del),
        "arm_deletions_all": {a: shares(arm_del_all[a]) for a in arm_del_all},
        "new_deletions": {a: shares(counts[a]) for a in counts},
        "recovered_by_arm": {a: shares(recovered[a]) for a in recovered},
        "sensitivity": {
            "alt_precedence_edge_overlap_name": {a: shares(alt_prec[a]) for a in alt_prec},
            "alt_alignment_tiebreak_prefer_deletion": {
                a: shares(alt_tie[a]) for a in alt_tie},
            "new_deletion_events": {a: shares(events[a]) for a in events},
        },
        "net_new": {a: n_new[a] for a in n_new},
        "single_window_domination": {
            a: {
                "max_window": max(per_window_new[a], key=per_window_new[a].get),
                "max_window_new": max(per_window_new[a].values()),
                "share_of_new": round(max(per_window_new[a].values()) / n_new[a], 4)
                if n_new[a] else None,
            } for a in per_window_new
        },
        "threshold": {
            "rule": ">40% of new deletions in ONE mechanically explainable category",
            "mechanical_categories": ["name", "edge", "overlap"],
            "residual_category": "ordinary",
            "max_mechanical_category": {
                a: max(mech, key=lambda c: counts[a][c]) for a in counts},
            "max_mechanical_share": {
                a: round(max(counts[a][c] for c in mech) / sum(counts[a].values()), 4)
                if sum(counts[a].values()) else None for a in counts},
            "largest_category_overall": {
                a: max(cats, key=lambda c: counts[a][c]) for a in counts},
            "largest_share_overall": {
                a: round(max(counts[a].values()) / sum(counts[a].values()), 4)
                if sum(counts[a].values()) else None for a in counts},
            "note": "`ordinary` is the residual bucket -- the absence of a mechanical "
                    "explanation, not one. It can never satisfy the gate, so the gate "
                    "reads max_mechanical_share.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
