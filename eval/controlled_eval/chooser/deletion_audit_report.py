#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read the deletion-audit verdicts, join them back to the reference, report.

The join is by clip filename, server-side, so the browser never held `oracle`.

READ THE NUMBERS THE WAY THIS SAMPLE WAS BUILT. cutclips.py drew every deletion
the reference calls wrong, plus an equal number it calls right, as blind controls
(100 / 103 among the 203 that got audio). So the pooled "share of deletions the
human calls harmful" IS NOT A POPULATION RATE and must never be quoted as one. The
estimable quantities are the two conditional rates -- harmful given the reference
says wrong, harmful given the reference says right -- and they are reported apart.

The unit is a deletion EVENT, not a word: answer 1 means at least some of the
deleted material was said and belongs in the text.

    python3 deletion_audit_report.py [--json]
"""
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
DATA_DIR = Path(os.environ.get("AUDIT_DATA", SC / "chooser-2026-08-25"))
SAMPLE = DATA_DIR / os.environ.get("AUDIT_SAMPLE", "listen_sample.json")
SMALL = DATA_DIR / "listen_sample_small.json"
VERDICTS = DATA_DIR / os.environ.get("AUDIT_VERDICTS", "deletion_audit_verdicts.json")
CRIT = ("ναι", "δεν")            # the frozen production guard lexicon, exactly

ANS = {1: "1 said, must stay      (deletion WRONG)",
       2: "2 said, filler/repeat  (deletion right)",
       3: "3 not said             (deletion right)",
       4: "4 unjudgeable          (excluded)"}


# ---------------------------------------------------------------- statistics

def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k + 1))


def upper95(k, n):
    """One-sided exact (Clopper-Pearson) 95% upper bound on p: solve P(X<=k)=0.05.

    Binomial, so it assumes independent clips. They are not: several clips come
    from one meeting and one speaker. Treat it as optimistic, and read it next to
    the concentration block below."""
    if n == 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if binom_cdf(k, n, mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def pct(x, n):
    return "   n/a" if not n else f"{100.0 * x / n:5.1f}%"


# ---------------------------------------------------------------- slices

def fingerprint(schema, seed, deck):
    """Byte-for-byte the server's. If the sample changed under the same filenames,
    the answers were given to a different question and must not be reported."""
    h = hashlib.sha256()
    h.update(f"{schema}|{seed}|{len(deck)}".encode("utf-8"))
    for o in deck:
        h.update(json.dumps([o["clip"], o["L"], o["R"], o["lost"]],
                            ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def must_validate(v, by_clip):
    order = v.get("order")
    if not isinstance(order, list) or not order:
        raise SystemExit("refusing: the verdict file records no clip order")
    if len(set(order)) != len(order):
        raise SystemExit("refusing: the recorded order has duplicates, so clips "
                         "would be counted more than once")
    missing = [c for c in order if c not in by_clip]
    if missing:
        raise SystemExit(f"refusing: {len(missing)} clip(s) in the recorded order are "
                         f"absent from {SAMPLE.name}, e.g. {missing[0]}")
    if len(order) != len(by_clip):
        raise SystemExit(f"refusing: the recorded order covers {len(order)} clips but "
                         f"{SAMPLE.name} has {len(by_clip)}")
    stored = v.get("fingerprint")
    if stored:
        live = fingerprint(v.get("schema"), v.get("seed"), [by_clip[c] for c in order])
        if live != stored:
            raise SystemExit(
                "refusing: the sample no longer matches the fingerprint recorded with "
                "these answers. The context or the deleted words shown to the judge "
                "have changed, so the answers are answers to a different question.\n"
                f"  recorded {stored}\n  now      {live}")
    else:
        print("!! no fingerprint recorded -- cannot prove these answers describe "
              "this sample", file=sys.stderr)
    for clip, rec in v.get("verdicts", {}).items():
        if not isinstance(rec, dict):
            raise SystemExit(f"refusing: malformed verdict for {clip!r}")
        a = rec.get("a")
        if a is not None and (type(a) is not int or not 1 <= a <= 4):
            raise SystemExit(f"refusing: verdict for {clip!r} has answer {a!r}, which is "
                             f"not 1-4. Silently treating it as a judgement would "
                             f"invent a result.")


def tally(rows):
    """rows: list of (sample_row, answer). Answer may be None (unanswered)."""
    t = {"n": len(rows), "counts": Counter(), "unanswered": 0,
         "agree": 0, "disagree": 0,
         "harmful": 0, "benign": 0, "determinate": 0, "unjudgeable": 0,
         "ref_wrong_n": 0, "ref_wrong_harmful": 0,
         "ref_right_n": 0, "ref_right_harmful": 0}
    for o, a in rows:
        if a is None:
            t["unanswered"] += 1
            continue
        if a not in (1, 2, 3, 4):
            raise SystemExit(f"refusing: answer {a!r} on {o['clip']!r} is not 1-4")
        t["counts"][a] += 1
        if a == 4:
            t["unjudgeable"] += 1
            continue
        ref_wrong = bool(o["oracle"])
        harmful = (a == 1)
        t["determinate"] += 1
        t["harmful" if harmful else "benign"] += 1
        t["agree" if harmful == ref_wrong else "disagree"] += 1
        if ref_wrong:
            t["ref_wrong_n"] += 1
            t["ref_wrong_harmful"] += harmful
        else:
            t["ref_right_n"] += 1
            t["ref_right_harmful"] += harmful
    return t


def line(label, t, width=34):
    d, h = t["determinate"], t["harmful"]
    return (f"  {label:<{width}} n={t['n']:>4}  judged={d:>4}  "
            f"harmful={h:>4} {pct(h, d)}  "
            f"concord={pct(t['agree'], d)}  "
            f"unjudg={t['unjudgeable']:>3}  unans={t['unanswered']:>3}")


def block(title, t, out):
    out(f"\n{title}")
    out(f"  clips in slice        {t['n']}")
    for a in (1, 2, 3, 4):
        out(f"    {ANS[a]}   {t['counts'][a]:>4}")
    out(f"    unanswered                          {t['unanswered']:>4}")
    d = t["determinate"]
    out(f"  judged (1|2|3)        {d}      unjudgeable (4) {t['unjudgeable']}")
    if not d:
        out("  nothing judged yet in this slice.")
        return
    out(f"  HARMFUL (answer 1)    {t['harmful']} / {d} = {pct(t['harmful'], d)}"
        f"   <- descriptive only; this denominator mixes the two strata")
    out(f"  range if every unjudged clip went the other way: "
        f"{pct(t['harmful'], t['n'])} .. "
        f"{pct(t['harmful'] + t['unjudgeable'] + t['unanswered'], t['n'])} of the slice")
    out("")
    out("  human vs reference (concordance, NOT accuracy -- the reference never listened)")
    out("  bounds below: exact binomial, one-sided 95%, over ANSWERED and JUDGEABLE")
    out("  events only. They assume independent clips; clips share meetings and")
    out("  speakers, so coverage is optimistic -- read them next to CONCENTRATION.")
    out("  While the audit is incomplete they also assume what is unanswered is")
    out("  not skipped selectively, which nobody has checked.")
    out(f"                          ref says WRONG   ref says RIGHT")
    out(f"    human: harmful  (1)   {t['ref_wrong_harmful']:>8}       "
        f"{t['ref_right_harmful']:>8}")
    out(f"    human: benign (2|3)   {t['ref_wrong_n'] - t['ref_wrong_harmful']:>8}       "
        f"{t['ref_right_n'] - t['ref_right_harmful']:>8}")
    out(f"    concordance           {t['agree']} / {d} = {pct(t['agree'], d)}")
    out(f"    P(harmful | ref WRONG) {t['ref_wrong_harmful']}/{t['ref_wrong_n']}"
        f" = {pct(t['ref_wrong_harmful'], t['ref_wrong_n'])}"
        f"   [upper {upper95(t['ref_wrong_harmful'], t['ref_wrong_n']):.3f}]"
        if t["ref_wrong_n"] else "    P(harmful | ref WRONG) n/a")
    out(f"    P(harmful | ref RIGHT) {t['ref_right_harmful']}/{t['ref_right_n']}"
        f" = {pct(t['ref_right_harmful'], t['ref_right_n'])}"
        f"   [upper {upper95(t['ref_right_harmful'], t['ref_right_n']):.3f}]"
        if t["ref_right_n"] else "    P(harmful | ref RIGHT) n/a")


def concentration(rows, out):
    """One window has previously supplied 67% of a headline effect here."""
    for label, key in (("meeting", lambda o: f"{o['city']}/{o['meeting']}"),
                       ("window", lambda o: f"{o['city']}/{o['meeting']}/{o['win']}")):
        harm = Counter()
        judged = Counter()
        for o, a in rows:
            if a in (1, 2, 3):
                judged[key(o)] += 1
                if a == 1:
                    harm[key(o)] += 1
        H = sum(harm.values())
        J = sum(judged.values())
        if not H:
            out(f"  by {label}: no harmful judgements yet")
            continue
        top, c = harm.most_common(1)[0]
        rest_h, rest_j = H - c, J - judged[top]
        out(f"  by {label}: {len(harm)} carry harm; top = {top} with {c}/{H}"
            f" = {pct(c, H)} of harmful; excluding it the rate goes"
            f" {pct(H, J)} -> {pct(rest_h, rest_j)}")


# ---------------------------------------------------------------- main

def main():
    as_json = "--json" in sys.argv
    if not VERDICTS.exists():
        raise SystemExit(f"no verdicts yet at {VERDICTS} -- run deletion_audit_server.py")
    v = json.loads(VERDICTS.read_text(encoding="utf-8"))
    sample = [o for o in json.loads(SAMPLE.read_text(encoding="utf-8")) if o.get("clip")]
    by_clip = {o["clip"]: o for o in sample}
    small = {o["clip"] for o in json.loads(SMALL.read_text(encoding="utf-8")) if o.get("clip")}
    crit = {o["clip"] for o in sample if any(w in CRIT for w in o["lost"])}

    must_validate(v, by_clip)
    order = v["order"]
    answers = {c: (r.get("a") if isinstance(r, dict) else None)
               for c, r in v.get("verdicts", {}).items()}
    unknown = [c for c in answers if c not in by_clip]
    rows = [(by_clip[c], answers.get(c)) for c in order]

    buf = []
    out = buf.append
    out("=" * 78)
    out("DELETION AUDIT -- human ear vs the fusion's deletions")
    out("=" * 78)
    out(f"verdicts   {VERDICTS}")
    out(f"sample     {SAMPLE}  ({len(sample)} clips)")
    out(f"seed       {v.get('seed')}   schema {v.get('schema')}   order recorded: "
        f"{'yes' if v.get('order') else 'NO'}")
    if v.get("fingerprint"):
        out(f"fingerprint {v['fingerprint'][:16]}…")
    if unknown:
        out(f"!! {len(unknown)} verdict(s) name clips absent from the sample; ignored")
    notes = sum(1 for r in v.get("verdicts", {}).values()
                if isinstance(r, dict) and r.get("note"))
    out(f"free-text comments left: {notes}")
    out("")
    out("HOW TO READ THIS. The sample is stratified by the reference, roughly 50/50:")
    nw = sum(1 for o in sample if o["oracle"])
    out(f"  {nw} deletions the reference calls WRONG, {len(sample) - nw} it calls RIGHT.")
    out("  That was deliberate, so the judge could not infer an answer from the base")
    out("  rate. The consequence is that the POOLED harmful share below is NOT the")
    out("  rate at which the fusion destroys speech in production. The two")
    out("  conditional rates are the estimable quantities. Do not pool them, and do")
    out("  not quote the overall percentage as a production harm rate.")
    out("")
    out("  unit = one deletion event (answer 1 = some deleted material was said and")
    out("  belongs in the text). Not words, not seconds.")

    whole = tally(rows)
    block("── ALL CLIPS " + "─" * 50, whole, out)

    out("\n── CONCENTRATION (single-item domination check) " + "─" * 20)
    concentration(rows, out)

    for field, title in (("kind", "BY kind"), ("rule", "BY rule"), ("cat", "BY cat")):
        out(f"\n── {title} " + "─" * (62 - len(title)))
        groups = defaultdict(list)
        for o, a in rows:
            groups[o[field]].append((o, a))
        for k in sorted(groups, key=lambda k: -len(groups[k])):
            out(line(k, tally(groups[k])))

    out("\n" + "=" * 78)
    out("SUBGROUPS THAT PRICE THE GUARD -- reported apart, never pooled with the above")
    out("=" * 78)
    out(f"These are TWO DIFFERENT SETS and the difference is real: the 36-row pilot")
    out(f"sample is not the same thing as the rows the frozen guard lexicon")
    out(f"{CRIT} would actually fire on. |36-sample|={len(small)}, "
        f"|lexicon rows|={len(crit)}, overlap={len(small & crit)}.")
    out("Neither is stratified the way the whole sample is; check the split shown.")

    for name, keys in (("A. the 36-clip pilot sample (listen_sample_small.json)", small),
                       ("B. rows whose deleted text contains ναι or δεν", crit)):
        sub = [(o, a) for o, a in rows if o["clip"] in keys]
        t = tally(sub)
        w = sum(1 for o, _ in sub if o["oracle"])
        block(f"── {name} " + "─" * max(1, 60 - len(name)) +
              f"\n  reference split in this subgroup: {w} wrong / {len(sub) - w} right",
              t, out)

    txt = "\n".join(buf)
    if as_json:
        def d(t):
            return {k: (dict(v_) if isinstance(v_, Counter) else v_) for k, v_ in t.items()}
        print(json.dumps({
            "verdicts_file": str(VERDICTS), "seed": v.get("seed"),
            "fingerprint": v.get("fingerprint"), "notes": notes,
            "all": d(whole),
            "by_kind": {k: d(tally([(o, a) for o, a in rows if o["kind"] == k]))
                        for k in {o["kind"] for o, _ in rows}},
            "by_rule": {k: d(tally([(o, a) for o, a in rows if o["rule"] == k]))
                        for k in {o["rule"] for o, _ in rows}},
            "small36": d(tally([(o, a) for o, a in rows if o["clip"] in small])),
            "lexicon": d(tally([(o, a) for o, a in rows if o["clip"] in crit])),
        }, ensure_ascii=False, indent=1))
    else:
        print(txt)


if __name__ == "__main__":
    main()
