# -*- coding: utf-8 -*-
"""The decisive run: does edit-based composition beat the rule, and is it noise?

Three configurations, three repeats each, on the same 200 islands. Repeats are the
point: on 80 islands a single run's 95% band at 50% is +/-11 points, and this
project has seen 25 points of category-accuracy drift between identical runs. One
number here would be worthless.

  base   selection only, no examples          - what we had
  B      Fable's edit format, 16 examples     - position-addressed edits
  D      same plus redundant addressing       - position AND word, cross-checked

D exists because a mis-counted position in B produces VALID JSON that edits the
wrong token, and nothing detects it. D's parser rejects the mismatch and counts it
separately, so the silent failure becomes a visible one.

Env: N (200), REPS (3), MODEL, BATCH, THINK
"""
import hashlib, json, os, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_chooser as RC
import autoprompt as A
from prompts_three_v3 import V as V3
from prompts_three_v4 import D as V4

SHORT = ["Σύστημα Α", "Σύστημα Β", "Σύστημα Γ"]
MAXW = 16
SYS = RC.SYSTEM.replace(
 "Διαλέγεις ΜΟΝΟ μία από τις προσφερόμενες διαδρομές.",
 "Δηλώνεις μία διαδρομή ως βάση και, αν χρειάζεται, το πολύ δύο μικρές διορθώσεις "
 "πάνω της. Κάθε λέξη που εισάγεις πρέπει να υπάρχει ήδη σε κάποια διαδρομή αυτού "
 "του σημείου.")


def avail(r):
    a = set()
    for s in r["spans"]:
        a |= set(s)
    return a


def apply_edits(base, e):
    """Returns (tokens, reason). reason is None on success."""
    if len(e) > 2:
        return None, "invalid"
    out, byq = [], defaultdict(list)
    for op in e:
        if not isinstance(op, dict):
            return None, "invalid"
        if "+" in op:
            q = op["+"]
            if not isinstance(q, int) or not 0 <= q <= len(base) or "w" not in op:
                return None, "invalid"
        elif "x" in op:
            q = op["x"]
            if not isinstance(q, int) or not 0 <= q < len(base):
                return None, "invalid"
            if "was" in op and base[q] != op["was"]:
                return None, "misindexed"
        else:
            return None, "invalid"
        byq[q].append(op)
    for q in range(len(base) + 1):
        for op in byq.get(q, []):
            if "+" in op:
                out.append(op["w"])
        if q == len(base):
            break
        xs = [op for op in byq.get(q, []) if "x" in op]
        if xs:
            if "w" in xs[0]:
                out.append(xs[0]["w"])
        else:
            out.append(base[q])
    return out, None


def render(rows, instr, fmt, shots):
    b = ["ΚΑΝΟΝΑΣ ΓΙΑ ΑΥΤΗ ΤΗΝ ΚΑΤΗΓΟΡΙΑ:", "", instr, "",
         "ΠΛΑΙΣΙΟ: συνεδρίαση ελληνικού δημοτικού συμβουλίου. Προφορικός λόγος, "
         "σπαστός, επαναληπτικός, συχνά αντιγραμματικός.", ""]
    if shots:
        b.append("ΛΥΜΕΝΑ ΠΑΡΑΔΕΙΓΜΑΤΑ:")
        for n, r in enumerate(shots, 1):
            ps = RC.paths_of(r)
            b.append(f"\nΠαράδειγμα {n}")
            b.append(f"  αριστερά: …{' '.join(r['L'].split()[-18:])}")
            for k, (t, _) in enumerate(ps):
                b.append(f"  P{k}: «{' '.join(t[:MAXW]) if t else '(τίποτα)'}»")
            b.append(f"  δεξιά: {' '.join(r['R'].split()[:18])}…")
            o = tuple(r["oracle"])
            hit = next((k for k, (t, _) in enumerate(ps) if tuple(t) == o), None)
            if hit is not None:
                b.append(f'  ΣΩΣΤΟ: {{"p":"P{hit}","e":[]}}')
            else:
                k = min(range(len(ps)), key=lambda i: abs(len(ps[i][0]) - len(o)))
                b.append(f'  ΣΩΣΤΟ: βάση P{k} με διόρθωση → «{" ".join(o)}»')
        b.append("")
    b.append(f"Ακολουθούν {len(rows)} σημεία.\n")
    for n, r in enumerate(rows):
        b.append(f"### σημείο {n}")
        b.append(f"αριστερά: …{r['L']}")
        for k, (t, sup) in enumerate(RC.paths_of(r)):
            who = " + ".join(SHORT[RC.SHORT.index(s)] for s in sup) if sup else "παράλειψη"
            b.append(f"  P{k}: «{' '.join(t) if t else '(τίποτα)'}»   [{who}]")
        b.append(f"δεξιά: {r['R']}…")
        b.append("")
    b.append(fmt)
    return "\n".join(b)


def call(prompt, model, think, system):
    env = dict(os.environ, MAX_THINKING_TOKENS=str(think))
    p = subprocess.run(["claude", "-p", prompt, "--model", model, "--output-format",
                        "json", "--append-system-prompt", system, "--allowed-tools", "",
                        "--strict-mcp-config", "--mcp-config",
                        json.dumps({"mcpServers": {}})],
                       capture_output=True, text=True, env=env, timeout=1500)
    try:
        d = json.loads(p.stdout)
        if isinstance(d, list):
            d = [x for x in d if x.get("type") == "result"][-1]
        if d.get("is_error") or "[" not in (d.get("result") or ""):
            return None, str(d.get("result"))[:120]
        return d["result"], None
    except Exception as e:
        return None, f"{e}"


def run(test, instr, fmt, shots, model, batch, think, edits):
    got = {}
    for s in range(0, len(test), batch):
        chunk = test[s:s + batch]
        txt, err = call(render(chunk, instr, fmt, shots), model, think,
                        SYS if edits else RC.SYSTEM)
        if err:
            print(f"      ERROR {err[:80]}", flush=True); continue
        m = re.search(r"\[.*\]", txt, re.S)
        if not m: continue
        try: arr = json.loads(m.group(0))
        except Exception: continue
        for o in arr:
            try: i = int(o["i"])
            except Exception: continue
            if 0 <= i < len(chunk): got[s + i] = o
    c = Counter()
    for i, r in enumerate(test):
        ps = RC.paths_of(r); fb = tuple(r["r12"]); o = tuple(r["oracle"])
        v = got.get(i)
        if not v: chosen, k = fb, "missing"
        elif str(v.get("p", "")).upper() == "ABSTAIN": chosen, k = fb, "abstain"
        else:
            m = re.fullmatch(r"P(\d+)", str(v.get("p", "")).upper())
            if not m or int(m.group(1)) >= len(ps): chosen, k = fb, "invalid"
            else:
                base = list(ps[int(m.group(1))][0]); e = v.get("e") or []
                if not edits or not e: chosen, k = tuple(base), "picked"
                else:
                    toks, why = apply_edits(base, e)
                    if why: chosen, k = fb, why
                    elif not set(toks) <= avail(r): chosen, k = fb, "invented"
                    else: chosen, k = tuple(toks), "edited"
        c[k] += 1; c["ok"] += int(chosen == o)
        if k == "edited": c["edited_ok"] += int(chosen == o)
    return c


def main():
    rows = [r for r in json.loads((HERE / "three_composite_wide.json").read_text())
            if r["city"] in A.SEARCH_CITIES]
    h = lambda r: hashlib.md5(f"{r['win']}{r['st']}".encode()).hexdigest()
    rows.sort(key=h)
    N = int(os.environ.get("N", "200")); REPS = int(os.environ.get("REPS", "3"))
    test, pool = rows[:N], rows[N:]
    st = defaultdict(list)
    for r in pool:
        if sum(len(s) for s in r["spans"]) > 3 * MAXW: continue
        st[(tuple(r["oracle"]) in [tuple(s) for s in r["spans"]], min(r["len"], 4))].append(r)
    for k in st: st[k].sort(key=h)
    keys = sorted(st, key=str)
    def shots(n):
        out, i = [], 0
        while len(out) < n and any(len(st[k]) > i for k in keys):
            for k in keys:
                if len(st[k]) > i and len(out) < n: out.append(st[k][i])
            i += 1
        return out
    f = lambda g: sum(1 for r in test if g(r)) / len(test)
    model = os.environ.get("MODEL", "sonnet")
    batch = int(os.environ.get("BATCH", "40")); think = int(os.environ.get("THINK", "6000"))
    print(f"THREE|COMPOSITE · test {len(test)} · {model} · {REPS} επαναλήψεις")
    print(f"null: R1+R2 {f(lambda r: tuple(r['r12'])==tuple(r['oracle'])):.1%} · "
          f"ταβάνι επιλογής {f(lambda r: tuple(r['oracle']) in [tuple(s) for s in r['spans']]):.1%}")
    import prompts_v2 as P2
    # base is dropped: R1+R2 is deterministic, so the null needs no runs at all.
    CFG = [("B 16-shot", V3["B"]["instr"], V3["B"]["format_note"], 16, True),
           ("D 16-shot", V4["instr"], V4["format_note"], 16, True)]
    res = {}
    print(f"\n{'config':14s} {'runs':>22s} {'μέση':>7s} {'min':>6s} {'max':>6s} "
          f"{'edited':>7s} {'ed.ok':>6s} {'misidx':>7s} {'inv':>5s}")
    for name, instr, fmt, ns, ed in CFG:
        sh = shots(ns); accs = []; agg = Counter()
        for rep in range(REPS):
            c = run(test, instr, fmt, sh, model, batch, think, ed)
            accs.append(c["ok"] / len(test)); agg.update(c)
            print(f"    {name} rep{rep}: {c['ok']/len(test):.1%}", flush=True)
        res[name] = accs
        eo = f"{agg['edited_ok']}/{agg['edited']}" if agg["edited"] else "—"
        print(f"{name:14s} {' '.join(f'{a:.1%}' for a in accs):>22s} "
              f"{sum(accs)/len(accs):6.1%} {min(accs):5.1%} {max(accs):5.1%} "
              f"{agg['edited']:7d} {eo:>6s} {agg['misindexed']:7d} {agg['invented']:5d}",
              flush=True)
    print()
    null = f(lambda r: tuple(r["r12"]) == tuple(r["oracle"]))
    for k, accs in res.items():
        d = sum(accs)/len(accs) - null
        print(f"{k}: {sum(accs)/len(accs):.1%} vs R1+R2 {null:.1%}  ->  {d:+.1f} μονάδες"
              f"{'   (κάτω από τη ζώνη ±7, αναποφάσιστο)' if abs(d)<0.07 else ''}")
    json.dump(res, open("decide_result.json", "w"))


if __name__ == "__main__":
    main()
