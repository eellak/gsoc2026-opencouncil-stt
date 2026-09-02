#!/usr/bin/env python3
"""Where the three systems err on the post-June held-out set, counted not browsed.

Substrate: `atlas_substrate.py` — 391 held-out windows, 117 (city, meeting) clusters,
trio scribe / soniox / oc-cleanpack-cont-s47-b, the exact 3-way MSA of `msa.py`.

WHAT THIS IS NOT. Codex job 59b4f061 rejected the first draft of this script for a
reason worth repeating at the top of the file: **the MSA aligns three hypotheses, and
the reference is not one of them.** There is no intrinsic "reference token for column
i", so "which system is right at this column" is not a well-defined statistic, and
per-class WER is not additive because replacing one column realigns everything after
it. Every reference-aware number below is therefore computed the only honest way:

    opportunity(X) = WER(W) - WER( best path through the lattice when only the
                                   columns in X may deviate from W's vote )

rescored end to end by the frozen scorer. That quantity is a **clairvoyant in-sample
ceiling under a fixed alignment**, never an achievable gain, and the opportunities of
two class sets do not sum to the opportunity of their union.

The per-column diagnostics ("the oracle preferred the dissenter here") describe the
GLOBAL oracle path. They are descriptive, they read the reference, and they do not
imply any rule could find the same column at decode time.

Writes results_error_atlas.json — counts and rates only, never transcript text.

Env: REPORT (path to the downloaded report.json), WORKERS.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.atlas_substrate import TRIO, load             # noqa: E402
from eval.controlled_eval.column_classes import (                       # noqa: E402
    CLASSES, column_class, column_flags, split_merge_columns)
from eval.controlled_eval.exp_fusion_deletions import sdi               # noqa: E402
from eval.controlled_eval.greek_phonetics import phon                   # noqa: E402
from eval.controlled_eval.msa import oracle_select                      # noqa: E402
from eval.controlled_eval.scoring import edist, wtoks                   # noqa: E402

OUT = Path(__file__).with_name("results_error_atlas.json")
N_BOOT = int(os.environ.get("N_BOOT", "10000"))

# Published numbers this atlas must reproduce before any of its own are believed.
# Source: results_composition_postjune.json, run 2026-08-22-post-june-held-out-...
EXPECT = {"single:scribe": 0.13771297450629663,
          "single:soniox": 0.14752380436157334,
          "single:oc-cleanpack-cont-s47-b": 0.1795128913942942,
          "W": 0.12019621659710554,
          "oracle_msa": 0.061132491372612785}
TOL = 5e-6


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------- the lattice DP
def lattice_best(cands, ref):
    """Cheapest text obtainable by choosing one candidate per column, against `ref`.

    Generalises `msa.oracle_select` by taking the candidate lists directly, so a
    caller can FORCE a column to a single token (W's vote) and free only the columns
    it wants to price. Same recurrence, same boundary row; verified against
    `oracle_select` on the real substrate in `_selftest`.

    `cands[i]` is a list of candidates for column i; None means emit nothing.
    """
    n, m = len(cands), len(ref)
    NEG = -1
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bk = [[(0, 0, NEG)] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        dp[0][j] = j
        bk[0][j] = (0, j - 1, NEG)
    for i in range(1, n + 1):
        cl = cands[i - 1]
        has_eps = bool(cl) and cl[0] is None
        dp[i][0] = dp[i - 1][0] + (0 if has_eps else 1)
        bk[i][0] = (i - 1, 0, 0)
        row, prev = dp[i], dp[i - 1]
        bki = bk[i]
        for j in range(1, m + 1):
            best = row[j - 1] + 1
            bkb = (i, j - 1, NEG)
            rj = ref[j - 1]
            for ci, e in enumerate(cl):
                if e is None:
                    v = prev[j]
                    if v < best:
                        best, bkb = v, (i - 1, j, ci)
                    continue
                v = prev[j - 1] + (0 if e == rj else 1)
                if v < best:
                    best, bkb = v, (i - 1, j - 1, ci)
                v = prev[j] + 1
                if v < best:
                    best, bkb = v, (i - 1, j, ci)
            row[j] = best
            bki[j] = bkb
    out = [None] * n
    i, j = n, m
    while i or j:
        pi, pj, ci = bk[i][j]
        if ci != NEG:
            out[i - 1] = cands[i - 1][ci]
        i, j = pi, pj
    return out


def col_cands(col):
    """Frozen candidate order: epsilon first if the column has one, then a, b, c."""
    seen, cl = set(), []
    if any(e is None for e in col):
        cl.append(None)
    for e in col:
        if e is not None and e not in seen:
            seen.add(e)
            cl.append(e)
    return cl


# ------------------------------------------------------------------------- islands
def islands(cols):
    """Maximal runs of consecutive non-`agree` columns, as (start, end) half-open.

    This is a description of the frozen MSA, not of a linguistic error span: an
    `agree` column can sit inside a wider boundary disagreement and cut it in two,
    and two unrelated errors one column apart merge into one island. `split_merge`
    is reported per island so the second failure mode is at least visible.
    """
    out, s = [], None
    for i, c in enumerate(cols):
        if column_class(c) != "agree":
            if s is None:
                s = i
        elif s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(cols)))
    return out


def bucket(n):
    return str(n) if n <= 3 else "4+"


# ------------------------------------------------------------------ per-window work
def analyse(win):
    cols = [tuple(c) for c in win["cols"]]
    ref, w_toks = win["ref"], win["w"]
    # per-column oracle choices; recomputed here rather than cached so the
    # substrate stays a pure alignment artifact with no reference-aware fields
    win = dict(win, oracle_sel=oracle_select(cols, ref))
    n = len(cols)
    klass = [column_class(c) for c in cols]
    sm = split_merge_columns(cols)

    # W's per-column choice, needed to force columns closed.
    from eval.controlled_eval.msa import vote_column
    w_choice = [vote_column(c, win["pivot"])[0] for c in cols]
    assert [t for t in w_choice if t is not None] == list(w_toks)

    free = [col_cands(c) for c in cols]
    forced = [[t] for t in w_choice]

    def restricted(idx):
        """WER counts when only columns in `idx` may deviate from W."""
        cands = [free[i] if i in idx else forced[i] for i in range(n)]
        toks = [t for t in lattice_best(cands, ref) if t is not None]
        return sdi(" ".join(ref), " ".join(toks))

    res = {"id": win["id"], "city": win["city"], "meeting": win["meeting"],
           "n_cols": n, "ref_tokens": len(ref)}

    # ---- baselines, all rescored by the frozen scorer
    res["sdi"] = {f"single:{p}": list(sdi(" ".join(ref), " ".join(h)))
                  for p, h in zip(TRIO, win["hyps"])}
    res["sdi"]["W"] = list(sdi(" ".join(ref), " ".join(w_toks)))
    res["sdi"]["oracle_msa"] = list(sdi(" ".join(ref), " ".join(win["oracle"])))

    # ---- class census
    res["classes"] = dict(Counter(klass))
    fl = defaultdict(int)
    for i, c in enumerate(cols):
        if klass[i] in ("unresolved_two", "unresolved_three", "exact_2_of_3"):
            f = column_flags(c)
            for k in ("strict_homophone", "loose_homophone", "partial_homophone"):
                if f[k]:
                    fl[f"{klass[i]}:{k}"] += 1
            fl[f"{klass[i]}:chardist_{min(f['max_char_dist'], 3)}"] += 1
    res["flags"] = dict(fl)
    res["split_merge_cols"] = len(sm)

    # ---- clairvoyant opportunity, one class at a time (NOT additive)
    res["opp_class"] = {}
    for k in CLASSES:
        idx = {i for i in range(n) if klass[i] == k}
        if idx:
            res["opp_class"][k] = list(restricted(idx))

    # ---- islands
    isl = islands(cols)
    res["islands"] = []
    per_len = defaultdict(set)
    for (s, e) in isl:
        L = e - s
        per_len[bucket(L)] |= set(range(s, e))
        toks_by_sys = [sum(1 for i in range(s, e) if cols[i][k] is not None)
                       for k in range(3)]
        res["islands"].append({
            "len": L,
            "sys_tokens": toks_by_sys,
            "w_tokens": sum(1 for i in range(s, e) if w_choice[i] is not None),
            "split_merge": any(i in sm for i in range(s, e)),
            # descriptive: did the GLOBAL fixed-lattice oracle path deviate from the
            # vote anywhere inside this island?
            "oracle_deviates": any(win["oracle_sel"][i] != w_choice[i]
                                   for i in range(s, e)),
        })
    # ---- clairvoyant opportunity by island length (this is the (d) question)
    res["opp_island_len"] = {b: list(restricted(idx)) for b, idx in per_len.items()}

    # ---- where the global oracle preferred the dissenter, inside exact_2_of_3
    mino = Counter()
    for i, c in enumerate(cols):
        if klass[i] != "exact_2_of_3":
            continue
        cnt = Counter(e for e in c if e is not None)
        maj, _ = cnt.most_common(1)[0]
        who = [k for k in range(3) if c[k] != maj][0]
        pair = "+".join(TRIO[k][:14] for k in range(3) if k != who)
        o = win["oracle_sel"][i]
        mino[f"n:{pair}|{TRIO[who][:14]}"] += 1
        if o is not None and o != maj:
            mino[f"oracle_took_minority:{pair}|{TRIO[who][:14]}"] += 1
        if o is None:
            mino[f"oracle_took_eps:{pair}|{TRIO[who][:14]}"] += 1
    res["minority"] = dict(mino)

    # ---- per-system substitution profile, from the SCORER's own alignment
    res["subprofile"] = {}
    for p, h in zip(TRIO, win["hyps"]):
        prof = Counter()
        for op, a, b in sdi_pairs(ref, h):
            if op != "S":
                continue
            if phon(a) == phon(b):
                prof["homophone_strict"] += 1
            elif phon(a, loose=True) == phon(b, loose=True):
                prof["homophone_loose"] += 1
            elif edist(list(a), list(b)) <= 2:
                prof["near_char"] += 1
            else:
                prof["far"] += 1
        res["subprofile"][p] = dict(prof)
    return res


def sdi_pairs(a, b):
    """The frozen scorer's traceback as (op, ref_tok, hyp_tok) triples.

    Same recurrence and same tie precedence as `exp_fusion_deletions.sdi`, so the
    op counts here equal that function's S/D/I exactly. Substitution SUBCLASSES
    inherit whatever traceback that precedence picks; when several optimal paths
    exist the split between classes can move even though S does not. `traceback_ties`
    in the output counts how often that ambiguity was present.
    """
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], op[i][0] = i, "D"
    for j in range(1, m + 1):
        d[0][j], op[0][j] = j, "I"
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                d[i][j], op[i][j] = d[i - 1][j - 1], "M"
                continue
            sub, dele, ins = d[i - 1][j - 1] + 1, d[i - 1][j] + 1, d[i][j - 1] + 1
            best = min(sub, dele, ins)
            d[i][j] = best
            op[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    out, i, j = [], n, m
    while i or j:
        o = op[i][j] if (i and j) else ("D" if i else "I")
        if o in ("M", "S"):
            out.append((o, a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif o == "D":
            out.append((o, a[i - 1], None))
            i -= 1
        else:
            out.append((o, None, b[j - 1]))
            j -= 1
    out.reverse()
    return out


# ---------------------------------------------------------------------- aggregation
def wer_of(t):
    S, D, I, R = t
    return (S + D + I) / R if R else 0.0


def add(t, u):
    return [t[i] + u[i] for i in range(4)]


def pooled(rows, key, sub=None):
    tot = [0, 0, 0, 0]
    for r in rows:
        v = r[key] if sub is None else r[key].get(sub)
        if v:
            tot = add(tot, v)
    return tot


def boot_delta(rows, a_key, b_key, seed=20260825):
    """Paired (city, meeting)-clustered bootstrap on WER(a) - WER(b)."""
    by = defaultdict(list)
    for r in rows:
        by[(r["city"], r["meeting"])].append(r)
    cl = sorted(by)
    rng = random.Random(seed)

    def d(sample):
        ta = [0, 0, 0, 0]
        tb = [0, 0, 0, 0]
        for r in sample:
            ta = add(ta, a_key(r))
            tb = add(tb, b_key(r))
        return wer_of(ta) - wer_of(tb)

    point = d(rows)
    out = []
    for _ in range(N_BOOT):
        sel = [rng.choice(cl) for _ in cl]
        out.append(d([r for c in sel for r in by[c]]))
    out.sort()
    return point, out[int(0.025 * N_BOOT)], out[int(0.975 * N_BOOT)]


def domination(rows, a_key, b_key):
    """Share of the |numerator delta| carried by the largest single cluster.

    This project has quoted a headline where one window supplied 67% of the effect.
    The check runs before any delta is written, never after.
    """
    by = defaultdict(float)
    for r in rows:
        a, b = a_key(r), b_key(r)
        by[(r["city"], r["meeting"])] += (sum(a[:3]) - sum(b[:3]))
    tot = sum(abs(v) for v in by.values())
    if not tot:
        return {"top_cluster_share": 0.0, "n_clusters": len(by)}
    top = max(by.values(), key=abs)
    return {"top_cluster_share": abs(top) / tot, "n_clusters": len(by)}


def main():
    report = Path(os.environ["REPORT"])
    sub = load(report)
    wins = sub["windows"]
    workers = int(os.environ.get("WORKERS", str(min(12, os.cpu_count() or 4))))
    log(f"analysing {len(wins)} windows on {workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(analyse, wins, chunksize=1), 1):
            rows.append(r)
            if k % 50 == 0:
                log(f"  {k}/{len(wins)}")

    # ---------- RECONCILIATION GATE. Nothing below is believed until this passes.
    recon = {}
    for arm in EXPECT:
        recon[arm] = {"got": wer_of(pooled(rows, "sdi", arm)),
                      "expected": EXPECT[arm]}
    bad = [a for a, v in recon.items() if abs(v["got"] - v["expected"]) > TOL]
    log("reconciliation: " + json.dumps(
        {a: round(v["got"], 6) for a, v in recon.items()}))
    if bad:
        raise SystemExit(f"RECONCILIATION FAILED for {bad} — refusing to write. "
                         f"The atlas is describing a different substrate than the "
                         f"published composition numbers.")
    log("reconciliation OK — all five published arms reproduce")

    out = {
        "substrate": sub["meta"],
        "code_commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "boot_seed": 20260825,
        "n_boot": N_BOOT,
        "reconciliation": recon,
        "reading_note": (
            "Every 'opportunity' below is WER(W) minus the WER of the best path "
            "through ONE fixed hypothesis alignment when only the named columns may "
            "deviate from the vote. It reads the reference. It is a clairvoyant "
            "in-sample ceiling, not an achievable gain, and opportunities do not sum "
            "across sets."),
    }

    # ---------- arms
    out["arms"] = {a: {"wer": wer_of(t), "sub": t[0] / t[3], "del": t[1] / t[3],
                       "ins": t[2] / t[3], "ref_tokens": t[3]}
                   for a in EXPECT for t in [pooled(rows, "sdi", a)]}

    # ---------- column census
    ncol = sum(r["n_cols"] for r in rows)
    cc = Counter()
    for r in rows:
        cc.update(r["classes"])
    out["n_columns"] = ncol
    out["classes"] = {k: {"n": cc[k], "share": cc[k] / ncol,
                          "windows_containing": sum(1 for r in rows
                                                    if r["classes"].get(k))}
                      for k in CLASSES if cc[k]}
    fl = Counter()
    for r in rows:
        fl.update(r["flags"])
    out["flags"] = dict(fl)
    out["split_merge_columns"] = sum(r["split_merge_cols"] for r in rows)

    # ---------- clairvoyant opportunity per class
    wkey = lambda r: r["sdi"]["W"]                                    # noqa: E731
    opp = {}
    for k in CLASSES:
        if not cc[k]:
            continue
        key = lambda r, k=k: r["opp_class"].get(k, r["sdi"]["W"])     # noqa: E731
        pt, lo, hi = boot_delta(rows, wkey, key)
        opp[k] = {"n_columns": cc[k], "wer_with_class_free": wer_of(pooled(
            [{"x": key(r)} for r in rows], "x")),
            "opportunity": pt, "ci": [lo, hi], **domination(rows, wkey, key)}
    out["opportunity_by_class"] = opp

    # ---------- clairvoyant opportunity by island length  (question (d))
    ilen = Counter()
    isl_stats = defaultdict(lambda: Counter())
    for r in rows:
        for i in r["islands"]:
            b = bucket(i["len"])
            ilen[b] += 1
            isl_stats[b]["n"] += 1
            isl_stats[b]["split_merge"] += int(i["split_merge"])
            isl_stats[b]["oracle_deviates"] += int(i["oracle_deviates"])
            for k in range(3):
                isl_stats[b][f"tok_{TRIO[k][:14]}"] += i["sys_tokens"][k]
            isl_stats[b]["tok_W"] += i["w_tokens"]
            # a "ragged" island is the user's 2-vs-3 case: the systems put a
            # different NUMBER of tokens inside the same island
            if len(set(i["sys_tokens"])) > 1:
                isl_stats[b]["ragged"] += 1
    oi = {}
    for b in sorted(ilen):
        key = lambda r, b=b: r["opp_island_len"].get(b, r["sdi"]["W"])  # noqa: E731
        pt, lo, hi = boot_delta(rows, wkey, key)
        oi[b] = {**{k: v for k, v in isl_stats[b].items()},
                 "opportunity": pt, "ci": [lo, hi], **domination(rows, wkey, key)}
    out["islands_by_length"] = oi
    out["n_islands"] = sum(ilen.values())

    # ---------- minority
    mino = Counter()
    for r in rows:
        mino.update(r["minority"])
    out["minority_in_exact_2_of_3"] = dict(mino)
    out["minority_note"] = (
        "'oracle_took_minority' means the GLOBAL fixed-lattice oracle path, which "
        "reads the reference, emitted the dissenting token at a column where two "
        "systems agreed. It is grouped by the agreeing PAIR because two agreeing "
        "systems are not two independent votes if they share a lineage. It is not "
        "evidence that any decode-time rule could find those columns.")

    # ---------- per-system substitution profile
    prof = {p: Counter() for p in TRIO}
    for r in rows:
        for p in TRIO:
            prof[p].update(r["subprofile"][p])
    out["substitution_profile"] = {
        p: {**dict(prof[p]), "S_total": pooled(rows, "sdi", f"single:{p}")[0]}
        for p in TRIO}

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(f"wrote {OUT}")
    log(json.dumps(out["arms"], indent=1))


if __name__ == "__main__":
    main()
