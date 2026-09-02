# -*- coding: utf-8 -*-
"""Do few-shot examples help on THREE|COMPOSITE, the one category with real mass?

THREE|COMPOSITE is 2,196 islands worth 0.0211 of WER if perfected - 40% of all
remaining headroom - and the only tested category where the LLM gained exactly
nothing. Its candidate ceiling is 71.9%, so 72% is the wall, not 100%.

EXAMPLES ARE SELECTED BY A FROZEN RULE, never by hand and never by whether the model
got them right: stratify by which system the oracle chose and by island length, then
take the lowest hash inside each stratum. Examples come from a disjoint pool of the
search cities, so no test island can appear as its own demonstration.

Long spans are capped: a THREE|COMPOSITE island can run 50+ columns and a handful of
those would dominate the prompt.

Env: N (test islands, 80), SHOTS (comma list, "0,8,20"), MODEL, BATCH, THINK
"""
import hashlib, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_chooser as RC
import autoprompt as A
import prompts_v2 as V

CAT = "THREE|COMPOSITE"
MAXW = 14                      # words per span shown in an example


def which(r):
    """Which system's whole span the oracle took, or 'mix' / 'eps'."""
    o = tuple(r["oracle"])
    if not o:
        return "eps"
    for i, s in enumerate(r["spans"]):
        if tuple(s) == o:
            return RC.SHORT[i]
    return "mix"


def short(toks):
    t = list(toks)
    return " ".join(t[:MAXW]) + (" …" if len(t) > MAXW else "")


def demo(r, n):
    ps = RC.paths_of(r)
    b = [f"Παράδειγμα {n}", f"  αριστερά: …{r['L'][-40:]}"]
    pick = None
    for k, (toks, sup) in enumerate(ps):
        b.append(f"  P{k}: «{short(toks) if toks else '(τίποτα)'}»")
        if tuple(toks) == tuple(r["oracle"]):
            pick = k
    b.append(f"  δεξιά: {r['R'][:40]}…")
    b.append(f"  ΣΩΣΤΟ: P{pick}")
    return "\n".join(b)


def build(instr, shots):
    if not shots:
        return instr
    return (instr + "\n\nΠΑΡΑΔΕΙΓΜΑΤΑ ΛΥΜΕΝΑ ΣΩΣΤΑ:\n\n"
            + "\n\n".join(demo(r, i + 1) for i, r in enumerate(shots))
            + "\n\nΤώρα κρίνε τα δικά σου σημεία με την ίδια λογική.")


def score(pool, instr, model, batch, think):
    picks = {}
    for s in range(0, len(pool), batch):
        b = pool[s:s + batch]
        txt, err = RC.call(RC.render(b, instr), model, think)
        if err:
            print(f"    ERROR {err[:90]}", flush=True)
            continue
        for i, v in RC.parse(txt, len(b)).items():
            picks[s + i] = v
    c = Counter()
    for i, r in enumerate(pool):
        ps = RC.paths_of(r)
        fb = tuple(r["r12"])
        p = picks.get(i)
        if not p or p[0] == "ABSTAIN":
            chosen, k = fb, ("abstain" if p else "missing")
        else:
            m = re.fullmatch(r"P(\d+)", p[0])
            chosen, k = ((tuple(ps[int(m.group(1))][0]), "chose")
                         if m and int(m.group(1)) < len(ps) else (fb, "invalid"))
        c[k] += 1
        c["ok"] += int(chosen == tuple(r["oracle"]))
    return c


def main():
    N = int(os.environ.get("N", "80"))
    shots_list = [int(x) for x in os.environ.get("SHOTS", "0,8,20").split(",")]
    model = os.environ.get("MODEL", "sonnet")
    batch = int(os.environ.get("BATCH", "40"))
    think = int(os.environ.get("THINK", "4000"))

    all_ = A.load_pool(CAT, A.SEARCH_CITIES, 10**6)
    h = lambda r: hashlib.md5(f"{r['win']}{r['st']}".encode()).hexdigest()
    all_.sort(key=h)
    test, demos = all_[:N], all_[N:]
    print(f"{CAT}: {len(all_)} νησιά αναζήτησης · test {len(test)} · pool παραδειγμάτων {len(demos)}")

    # frozen stratified selection: by oracle source x length bucket, lowest hash first
    strata = defaultdict(list)
    for r in demos:
        if sum(len(s) for s in r["spans"]) > 3 * MAXW:
            continue                      # skip monsters, they would dominate
        strata[(which(r), min(r["len"], 4))].append(r)
    for k in strata:
        strata[k].sort(key=h)
    keys = sorted(strata)
    print(f"στρώματα παραδειγμάτων: {len(keys)}  " +
          ", ".join(f"{a}/{b}:{len(strata[(a,b)])}" for a, b in keys[:6]))

    def pick_shots(n):
        out, i = [], 0
        while len(out) < n and any(len(strata[k]) > i for k in keys):
            for k in keys:
                if len(strata[k]) > i and len(out) < n:
                    out.append(strata[k][i])
            i += 1
        return out

    base = V.P[CAT]["instr"]
    f = lambda g: sum(1 for r in test if g(r)) / len(test)
    print(f"\nnull στα ίδια νησιά: R1+R2 {f(lambda r: r['r12ok']):.1%} · "
          f"ψήφος {f(lambda r: r['ok']):.1%} · "
          f"Scribe {f(lambda r: r['spans'][0]==r['oracle']):.1%} · "
          f"ταβάνι {f(lambda r: tuple(r['oracle']) in [tuple(p[0]) for p in RC.paths_of(r)]):.1%}")
    print(f"\n{'shots':>6s} {'σωστά':>7s} {'αποφ.':>6s} {'αποχή':>6s} {'άκυρα':>6s} {'~tokens':>8s}")
    for n in shots_list:
        sh = pick_shots(n)
        instr = build(base, sh)
        c = score(test, instr, model, batch, think)
        print(f"{n:6d} {c['ok']/len(test):6.1%} {c['chose']:6d} {c['abstain']:6d} "
              f"{c['invalid']:6d} {len(instr)//3:8d}", flush=True)


if __name__ == "__main__":
    main()
