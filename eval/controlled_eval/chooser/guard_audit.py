# -*- coding: utf-8 -*-
"""Price the vote/negation guard on the RULES-ONLY arm, over the 391 windows.

The published pricing ("rescues 26, breaks 31, net -5") was measured on the POLICY
arm (rules + LLM chooser) on the confirmation cities. What ships is R1+R2 + the
guard, and `guard: true` is hardcoded in FusionTranscriber. This recomputes the
same four questions for the rules-only arm:

  1. the 75 islands where the frozen guard fires, and which token each rescues
  2. how many of those rescues are right by the reference (island oracle) and how
     many are wrong -- plus the WER-truth version, a per-island counterfactual
  3. what the rules-only arm deletes at islands, per word type
  4. whether one meeting or window dominates the guard's effect

The guard definition is READ from make_fixtures.py, not restated here. Aggregates
only: no transcript text is written, and the only word types printed are the
critical lexicon and the function-word deletion census, both counts.

Env: REPORT, SC. Writes $SC/chooser-2026-08-25/results_guard_rules_only.json
(counts only, never git).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/harold/opencouncil-fine-tuning")

from eval.controlled_eval.atlas_substrate import load, sc                # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import sdi                # noqa: E402
from eval.controlled_eval.msa import oracle_select                       # noqa: E402
from reconstruct import window_parts                                     # noqa: E402
from make_fixtures import apply_guard, CRIT                              # noqa: E402

# exploratory only: the lexicon the ticket assumed, against the frozen two
WIDE = ("ναι", "δεν", "όχι", "οχι", "μη", "μην")


def guard_with(lex, spans, out):
    core = [[t for t in sp if t not in lex] for sp in spans]
    if not (core[0] == core[1] == core[2]):
        return out, False
    n_crit = [sum(t in lex for t in sp) for sp in spans]
    if not any(n_crit):
        return out, False
    if any(t in lex for t in out):
        return out, False
    k = max(range(3), key=lambda i: (n_crit[i], len(spans[i]), -i))
    return list(spans[k]), True


def job(w):
    isl, tail, wsel, cols = window_parts(w)
    osel = oracle_select(cols, w["ref"])
    ref = " ".join(w["ref"])

    base = [list(i["r12"]) for i in isl]
    guarded = [apply_guard([list(x) for x in i["spans"]], list(i["r12"])) for i in isl]
    wide = [guard_with(WIDE, [list(x) for x in i["spans"]], list(i["r12"])) for i in isl]

    def text(sel, alt):
        out = []
        for k, i in enumerate(isl):
            out += i["before"]
            out += (alt[k][0] if k in sel else base[k])
        return " ".join(out + tail)

    fired = {k for k in range(len(isl)) if guarded[k][1]}
    fired_wide = {k for k in range(len(isl)) if wide[k][1]}
    off = sdi(ref, text(set(), guarded))
    on = sdi(ref, text(fired, guarded))
    on_wide = sdi(ref, text(fired_wide, wide))

    rows = []
    for k in sorted(fired):
        i = isl[k]
        tgt = [t for t in (osel[j] for j in range(i["s"], i["e"])) if t is not None]
        single = sdi(ref, text({k}, guarded))
        piece, _ = guarded[k]
        add = Counter(t for t in piece if t in CRIT) - Counter(t for t in base[k]
                                                              if t in CRIT)
        rows.append({
            "win": w["id"], "city": w["city"], "meeting": w["meeting"], "k": k,
            "cat": i["cat"], "ilen": i["e"] - i["s"],
            "rescued": dict(add),
            "src": max(range(3), key=lambda x: (
                sum(t in CRIT for t in i["spans"][x]), len(i["spans"][x]), -x)),
            "n_base": len(base[k]), "n_guard": len(piece),
            "noncrit_changed": int([t for t in piece if t not in CRIT]
                                   != [t for t in base[k] if t not in CRIT]),
            "base_empty": int(not base[k]),
            "crit_in_oracle": {t: sum(1 for x in tgt if x == t) for t in CRIT},
            "base_ok": int(base[k] == tgt), "guard_ok": int(piece == tgt),
            "oracle_len": len(tgt),
            "dedits": sum(single[:3]) - sum(off[:3]),
        })

    # whole-island deletion census for the guard-OFF rules arm
    dele = []
    for k, i in enumerate(isl):
        if base[k] or not any(i["spans"]):
            continue
        tgt = [t for t in (osel[j] for j in range(i["s"], i["e"])) if t is not None]
        dele.append({"win": w["id"], "city": w["city"], "meeting": w["meeting"],
                     "cat": i["cat"], "oracle": tgt, "vote": list(i["vote"]),
                     "wrong": int(bool(tgt))})

    return {"id": w["id"], "city": w["city"], "meeting": w["meeting"],
            "off": off, "on": on, "on_wide": on_wide,
            "n_isl": len(isl), "n_fired": len(fired), "n_fired_wide": len(fired_wide),
            "rows": rows, "dele": dele}


def wer(t):
    return (t[0] + t[1] + t[2]) / t[3]


def main():
    sub = load(Path(os.environ["REPORT"]))
    wins = list(sub["windows"])
    print(f"{len(wins)} windows", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", "14"))) as p:
        for r in p.map(job, wins, chunksize=4):
            res.append(r)

    tot = {a: [0, 0, 0, 0] for a in ("off", "on", "on_wide")}
    for r in res:
        for a in tot:
            for j in range(4):
                tot[a][j] += r[a][j]
    n_isl = sum(r["n_isl"] for r in res)
    rows = [x for r in res for x in r["rows"]]
    dele = [x for r in res for x in r["dele"]]

    print(f"\nislands {n_isl}   guard fired {len(rows)}   "
          f"wide-lexicon fired {sum(r['n_fired_wide'] for r in res)}")
    for a in ("off", "on", "on_wide"):
        print(f"  {a:8s} {wer(tot[a]):.5f}  sidn {tot[a]}")

    # --- 1. what is rescued
    resc = Counter()
    for x in rows:
        resc.update(x["rescued"])
    print("\n[1] rescued token types over the fired islands")
    for t, n in resc.most_common():
        print(f"  {t:6s} {n}")
    print(f"  islands whose rules output was EMPTY before the guard: "
          f"{sum(x['base_empty'] for x in rows)} of {len(rows)}")
    print(f"  islands where the guard also changed a NON-critical token: "
          f"{sum(x['noncrit_changed'] for x in rows)}")
    print("  span the guard adopted: " + ", ".join(
        f"{n}x {['scribe','soniox','ours'][k]}"
        for k, n in Counter(x["src"] for x in rows).most_common()))
    print("  island category: " + ", ".join(
        f"{c} {n}" for c, n in Counter(x["cat"] for x in rows).most_common()))

    # --- 2. right or wrong
    print("\n[2a] by the island oracle (the policy pricing's unit)")
    resc_n = sum(1 for x in rows if not x["base_ok"] and x["guard_ok"])
    brk_n = sum(1 for x in rows if x["base_ok"] and not x["guard_ok"])
    both_w = sum(1 for x in rows if not x["base_ok"] and not x["guard_ok"])
    both_r = sum(1 for x in rows if x["base_ok"] and x["guard_ok"])
    print(f"  rescued (was wrong, now right): {resc_n}")
    print(f"  broken  (was right, now wrong): {brk_n}")
    print(f"  still wrong both ways: {both_w}   right both ways: {both_r}")
    print(f"  net islands: {resc_n - brk_n:+d}")

    by_verdict = defaultdict(Counter)
    for x in rows:
        v = "correct" if x["guard_ok"] else "wrong"
        by_verdict[v].update(x["rescued"])
    print("  rescued token by verdict: " + "; ".join(
        f"{v}: " + ", ".join(f"{t} {n}" for t, n in c.most_common())
        for v, c in sorted(by_verdict.items())))
    print("  island length (columns): " + ", ".join(
        f"{L}->{n}" for L, n in sorted(Counter(x["ilen"] for x in rows).items())))
    print("  tokens the guard adds per island: " + ", ".join(
        f"{L}->{n}" for L, n in sorted(Counter(x["n_guard"] for x in rows).items())))

    print("\n[2b] does the reference actually contain the rescued token?")
    have = sum(1 for x in rows
               if sum(x["crit_in_oracle"].values()) >= sum(x["rescued"].values()))
    part = sum(1 for x in rows
               if 0 < sum(x["crit_in_oracle"].values()) < sum(x["rescued"].values()))
    none = sum(1 for x in rows if sum(x["crit_in_oracle"].values()) == 0)
    print(f"  oracle has at least as many critical tokens as the guard added: {have}")
    print(f"  oracle has some but fewer: {part}")
    print(f"  oracle has none (the guard inserted a word nobody said): {none}")

    print("\n[2c] by WER, per-island counterfactual (Δ edit operations)")
    good = [x for x in rows if x["dedits"] < 0]
    bad = [x for x in rows if x["dedits"] > 0]
    flat = [x for x in rows if x["dedits"] == 0]
    print(f"  islands that lower edits: {len(good)}  (sum {sum(x['dedits'] for x in good)})")
    print(f"  islands that raise edits: {len(bad)}  (sum +{sum(x['dedits'] for x in bad)})")
    print(f"  islands with no effect:   {len(flat)}")
    print(f"  sum of single-island Δ: {sum(x['dedits'] for x in rows):+d}   "
          f"measured all-on Δ: {sum(tot['on'][:3]) - sum(tot['off'][:3]):+d}")

    # --- 3. deletion census at islands, guard-off rules arm
    print(f"\n[3] whole-island deletions by the rules-only arm (guard off): {len(dele)}")
    wrong = [d for d in dele if d["wrong"]]
    print(f"  wrong by the island oracle (material that should have stayed): "
          f"{len(wrong)}  = {len(wrong)/max(1,len(dele)):.1%}")
    lost = Counter(t for d in wrong for t in d["oracle"])
    print("  word types lost in those islands (oracle target), top 20:")
    for t, n in lost.most_common(20):
        print(f"    {t:12s} {n}")
    print(f"  distinct types {len(lost)}, tokens {sum(lost.values())}")
    for t in CRIT:
        occ = sum(1 for d in dele if t in d["vote"] or t in d["oracle"])
        w_isl = sum(1 for d in wrong if t in d["oracle"])
        print(f"  «{t}»: islands emptied that involve it {occ}, of which wrong "
              f"{w_isl}, tokens lost {lost[t]}")

    # --- 4. domination
    print("\n[4] single-item domination of the guard's effect")
    for key, name in ((lambda x: (x["city"], x["meeting"]), "meeting"),
                      (lambda x: x["win"], "window")):
        agg = defaultdict(int)
        for x in rows:
            agg[key(x)] += x["dedits"]
        T = sum(abs(v) for v in agg.values()) or 1
        net = sum(agg.values())
        top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:5]
        print(f"  by {name}: {len(agg)} carriers, net {net:+d}, "
              f"gross {T}, largest {abs(top[0][1])}/{T} = {abs(top[0][1])/T:.1%}")
        print("    top 5 |Δ|: " + ", ".join(str(abs(v)) for _, v in top))
        # leave-one-out on the sign of the net. Compare against the sign of `net`
        # itself: hardcoding `<= 0` only tests a flip when net is positive, and
        # silently reports every carrier as a flip when it is negative
        # (CodeRabbit, 2026-09-06).
        sign = (net > 0) - (net < 0)
        flips = sum(1 for _, v in agg.items()
                    if sign == 0 or ((net - v) > 0) - ((net - v) < 0) != sign)
        print(f"    carriers whose removal would flip the sign of the net: {flips}")

    out = {
        "n_windows": len(res), "n_islands": n_isl,
        "totals": tot, "wer": {a: wer(tot[a]) for a in tot},
        "guard_lexicon": list(CRIT), "wide_lexicon": list(WIDE),
        "fired": len(rows), "fired_wide": sum(r["n_fired_wide"] for r in res),
        "rescued_types": dict(resc),
        "rescued_by_verdict": {v: dict(c) for v, c in by_verdict.items()},
        "island_oracle": {"rescued": resc_n, "broken": brk_n,
                          "wrong_both": both_w, "right_both": both_r},
        "wer_counterfactual": {"lower": len(good), "raise": len(bad),
                               "flat": len(flat),
                               "sum": sum(x["dedits"] for x in rows)},
        "deletions": {"n": len(dele), "wrong": len(wrong),
                      "lost_types": dict(lost.most_common())},
    }
    dest = sc() / "chooser-2026-08-25" / "results_guard_rules_only.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
