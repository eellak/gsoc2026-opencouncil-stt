# -*- coding: utf-8 -*-
"""Let the model COMPOSE, not only pick, on THREE|COMPOSITE.

The measurement that motivated this: in this category the correct answer uses ONLY
words some system emitted 100% of the time, but is exactly one system's whole span
only 63.3% of the time. Whole-path selection is therefore capped at 63%; composition
from the same words is capped at 100%.

THE GUARD. Free text would let the model write fluent Greek nobody said. So every
token of a composed answer must appear in some system's span for this island, and
violations are counted and reported, never silently accepted. A violating answer
falls back to R1+R2, exactly like an abstention.

Env: N (test islands), SHOTS, MODEL, BATCH, THINK, MODE (pick|compose)
"""
import hashlib, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_chooser as RC
import prompts_v2 as V

SHORT = ["Σύστημα Α", "Σύστημα Β", "Σύστημα Γ"]
MAXW = 16

SYSTEM_COMPOSE = RC.SYSTEM.replace(
 "Διαλέγεις ΜΟΝΟ μία από τις προσφερόμενες διαδρομές.",
 "Μπορείς είτε να διαλέξεις μία διαδρομή, είτε να ΣΥΝΘΕΣΕΙΣ μία νέα από τις λέξεις "
 "των διαδρομών.\nΚΑΝΟΝΑΣ ΑΠΑΡΑΒΑΤΟΣ: κάθε λέξη της σύνθεσής σου πρέπει να υπάρχει "
 "ήδη σε κάποια από τις διαδρομές αυτού του σημείου. Δεν προσθέτεις λέξεις, δεν "
 "διορθώνεις κλίση, δεν ομαλοποιείς. Συνδυάζεις μόνο ό,τι κάποιο σύστημα άκουσε.")

INSTR_TAIL = """
Σε αυτή την κατηγορία τα τρία συστήματα άκουσαν διαφορετικά πράγματα ΚΑΙ διαφορετικό
μήκος. Συχνά κανένα δεν έχει ολόκληρο δίκιο: το ένα πιάνει την αρχή, το άλλο το τέλος.

Διάβασε τα συμφραζόμενα αριστερά και δεξιά. Βρες ποια ακολουθία λέξεων συνεχίζει
φυσικά το αριστερό και οδηγεί φυσικά στο δεξί, χρησιμοποιώντας ΜΟΝΟ λέξεις που
εμφανίζονται στις διαδρομές.

Αν μια ολόκληρη διαδρομή ταιριάζει, διάλεξέ την με το αναγνωριστικό της.
Αν το σωστό είναι συνδυασμός, γράψε τον.
"""


def avail(r):
    a = set()
    for s in r["spans"]:
        a |= set(s)
    return a


def render(rows, instr, shots, compose):
    b = ["ΚΑΝΟΝΑΣ ΓΙΑ ΑΥΤΗ ΤΗΝ ΚΑΤΗΓΟΡΙΑ:", "", instr, ""]
    if compose:
        b.append(INSTR_TAIL)
    b += ["ΠΛΑΙΣΙΟ: συνεδρίαση ελληνικού δημοτικού συμβουλίου. Προφορικός λόγος, "
          "κατακερματισμένος, επαναληπτικός, συχνά αντιγραμματικός.", ""]
    if shots:
        b.append("ΛΥΜΕΝΑ ΠΑΡΑΔΕΙΓΜΑΤΑ:")
        for n, r in enumerate(shots, 1):
            ps = RC.paths_of(r)
            b.append(f"\nΠαράδειγμα {n}")
            b.append(f"  αριστερά: …{' '.join(r['L'].split()[-18:])}")
            for k, (toks, _) in enumerate(ps):
                b.append(f"  P{k}: «{' '.join(toks[:MAXW]) if toks else '(τίποτα)'}»")
            b.append(f"  δεξιά: {' '.join(r['R'].split()[:18])}…")
            o = tuple(r["oracle"])
            hit = next((k for k, (t, _) in enumerate(ps) if tuple(t) == o), None)
            if hit is not None:
                b.append(f"  ΣΩΣΤΟ: P{hit}  (ολόκληρη μία διαδρομή)")
            else:
                b.append(f"  ΣΩΣΤΟ: σύνθεση → «{' '.join(o)}»")
                b.append("         (κάθε λέξη της υπάρχει σε κάποια διαδρομή)")
        b.append("")
    b.append(f"Ακολουθούν {len(rows)} σημεία.\n")
    for n, r in enumerate(rows):
        b.append(f"### σημείο {n}")
        b.append(f"αριστερά: …{r['L']}")
        for k, (toks, sup) in enumerate(RC.paths_of(r)):
            who = " + ".join(SHORT[RC.SHORT.index(s)] for s in sup) if sup else "παράλειψη"
            b.append(f"  P{k}: «{' '.join(toks) if toks else '(τίποτα)'}»   [{who}]")
        b.append(f"δεξιά: {r['R']}…")
        b.append("")
    if compose:
        b += ['Απάντησε με πίνακα JSON. Για κάθε σημείο, είτε {"i":0,"p":"P2"} για '
              'ολόκληρη διαδρομή, είτε {"i":0,"t":"οι λεξεις που συνθετεις"} για σύνθεση.',
              'Παράδειγμα: [{"i":0,"p":"P1"},{"i":1,"t":"και θα παμε στο δημαρχειο"}]',
              "Καμία άλλη έξοδος."]
    else:
        b += ['Απάντησε με πίνακα JSON: [{"i":0,"p":"P0"}, {"i":1,"p":"ABSTAIN"}]',
              "Καμία άλλη έξοδος."]
    return "\n".join(b)


def parse(txt, n):
    m = re.search(r"\[.*\]", txt or "", re.S)
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return {}
    out = {}
    for o in arr:
        try:
            i = int(o["i"])
        except Exception:
            continue
        if 0 <= i < n:
            out[i] = (str(o.get("p") or "").strip().upper(), str(o.get("t") or "").strip())
    return out


def run(test, instr, shots, model, batch, think, compose):
    picks = {}
    for s in range(0, len(test), batch):
        b = test[s:s + batch]
        env = dict(os.environ, MAX_THINKING_TOKENS=str(think))
        import subprocess
        p = subprocess.run(
            ["claude", "-p", render(b, instr, shots, compose), "--model", model,
             "--output-format", "json", "--append-system-prompt",
             SYSTEM_COMPOSE if compose else RC.SYSTEM,
             "--allowed-tools", "", "--strict-mcp-config",
             "--mcp-config", json.dumps({"mcpServers": {}})],
            capture_output=True, text=True, env=env, timeout=1200)
        try:
            d = json.loads(p.stdout)
            if isinstance(d, list):
                d = [x for x in d if x.get("type") == "result"][-1]
            if d.get("is_error") or "[" not in (d.get("result") or ""):
                print(f"    ERROR {str(d.get('result'))[:90]}", flush=True); continue
            txt = d["result"]
        except Exception as e:
            print(f"    ERROR {e}", flush=True); continue
        for i, v in parse(txt, len(b)).items():
            picks[s + i] = v
    c = Counter()
    for i, r in enumerate(test):
        ps = RC.paths_of(r); fb = tuple(r["r12"]); o = tuple(r["oracle"])
        v = picks.get(i)
        if not v:
            chosen, k = fb, "missing"
        elif v[1]:
            toks = tuple(v[1].split())
            if set(toks) <= avail(r):
                chosen, k = toks, "composed"
            else:
                chosen, k = fb, "invented"       # used words nobody said
        elif v[0] == "ABSTAIN":
            chosen, k = fb, "abstain"
        else:
            m = re.fullmatch(r"P(\d+)", v[0])
            chosen, k = ((tuple(ps[int(m.group(1))][0]), "picked")
                         if m and int(m.group(1)) < len(ps) else (fb, "invalid"))
        c[k] += 1
        c["ok"] += int(chosen == o)
        if k == "composed":
            c["composed_ok"] += int(chosen == o)
        if k == "picked":
            c["picked_ok"] += int(chosen == o)
    return c


def main():
    rows = json.loads((HERE / "three_composite_wide.json").read_text())
    import autoprompt as A
    rows = [r for r in rows if r["city"] in A.SEARCH_CITIES]
    h = lambda r: hashlib.md5(f"{r['win']}{r['st']}".encode()).hexdigest()
    rows.sort(key=h)
    N = int(os.environ.get("N", "80"))
    test, pool = rows[:N], rows[N:]
    strata = defaultdict(list)
    for r in pool:
        if sum(len(s) for s in r["spans"]) > 3 * MAXW:
            continue
        exact = tuple(r["oracle"]) in [tuple(s) for s in r["spans"]]
        strata[(exact, min(r["len"], 4))].append(r)
    for k in strata:
        strata[k].sort(key=h)
    keys = sorted(strata, key=str)

    def shots(n):
        out, i = [], 0
        while len(out) < n and any(len(strata[k]) > i for k in keys):
            for k in keys:
                if len(strata[k]) > i and len(out) < n:
                    out.append(strata[k][i])
            i += 1
        return out

    f = lambda g: sum(1 for r in test if g(r)) / len(test)
    print(f"THREE|COMPOSITE · test {len(test)} · wide context")
    print(f"null: R1+R2 {f(lambda r: tuple(r['r12'])==tuple(r['oracle'])):.1%} · "
          f"ταβάνι επιλογής {f(lambda r: tuple(r['oracle']) in [tuple(s) for s in r['spans']]):.1%} · "
          f"ταβάνι σύνθεσης {f(lambda r: set(r['oracle'])<=avail(r)):.1%}")
    base = V.P["THREE|COMPOSITE"]["instr"]
    model = os.environ.get("MODEL", "sonnet")
    batch = int(os.environ.get("BATCH", "40"))
    think = int(os.environ.get("THINK", "6000"))
    print(f"\n{'τρόπος':22s} {'σωστά':>7s} {'επέλεξε':>8s} {'συνέθεσε':>9s} "
          f"{'σύνθ.σωστές':>12s} {'επινόησε':>9s}")
    for label, shot_n, comp in [("επιλογή, 0 shots", 0, False),
                                ("σύνθεση, 0 shots", 0, True),
                                ("σύνθεση, 8 shots", 8, True),
                                ("σύνθεση, 16 shots", 16, True)]:
        c = run(test, base, shots(shot_n), model, batch, think, comp)
        cs = f"{c['composed_ok']}/{c['composed']}" if c["composed"] else "—"
        print(f"{label:22s} {c['ok']/len(test):6.1%} {c['picked']:8d} {c['composed']:9d} "
              f"{cs:>12s} {c['invented']:9d}", flush=True)


if __name__ == "__main__":
    main()
