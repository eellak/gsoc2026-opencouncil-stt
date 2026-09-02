# -*- coding: utf-8 -*-
"""Prompt search per category, with the multiplicity defence taken from the repo.

WHY THE CEREMONY. Scoring a prompt is cheap and `oracle` reads the reference, so a
loop that tries fifty prompts and keeps the best one is a machine for manufacturing
a false positive. This project's expensive mistakes have all been measurement
mistakes. So the city partition of `eval/controlled_eval/autoresearch.py`
(PROTOCOL autoresearch-2026-08-16b) is reused verbatim and ENFORCED here:

  SEARCH   athens chalandri chania orestiada vrilissia zografou   - iterate freely
  CONFIRM  argos samothraki sparta xylokastro                     - touched ONCE, at the end

`argithea` did not exist in the pinned 10-city split (it arrives with the 391-window
substrate). It is assigned to CONFIRM, because it has never had a rule fitted on it
and confirmation is the scarcer resource. Declared here, before any number.

WHAT THE LOOP DOES, per category and per round:
  1. run the current prompt on the SEARCH sample
  2. hand Codex the prompt, the score, and the actual failures
  3. Codex returns a revised prompt, or says no change is worth making
  4. run the revision on the SAME sample; keep whichever scores higher on SEARCH
  5. leave-one-search-city-out each round, so a prompt that wins on one city is visible

WHAT THE SEARCH NUMBERS ARE. Exploratory. 21 categories x N rounds is a large family;
several prompts will beat their predecessor by chance. Nothing here ships. The frozen
prompt set gets exactly ONE confirmation run, afterwards, in one batch.

Env: CATS (comma list or ALL), MODEL, ROUNDS (3), N (90), BATCH (15), THINK (4000),
     PROMPTS (starting module), EFFORT (codex effort, high)
"""
import json, os, re, subprocess, sys, time, hashlib
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_chooser as RC

PROTOCOL = "autoprompt-2026-08-25a"
SEARCH_CITIES = ("athens", "chalandri", "chania", "orestiada", "vrilissia", "zografou")
CONFIRM_CITIES = ("argos", "samothraki", "sparta", "xylokastro", "argithea")
CB = "/home/harold/codex-bridge"
STATE = HERE / "autoprompt_state.json"


def log(m): print(m, flush=True)


def key(k):
    return k.replace("|COMPOSITE", "") if k.startswith(("TWO_", "SOLO_", "PAIRE_")) else k


def load_pool(cat, cities, n, seed=20260825):
    """`cat` may be several categories joined by '&' - a POOLED category.

    Fable's finding: the PAIRE_* family shares one logic and differs only in which
    system is silent, and separately none of them has enough search islands to
    measure alone (42, 47, 18). Pooling is the honest fix; measuring 18 islands is
    not, because one island moves the rate by 5.6 points.
    """
    import random
    rows = json.loads((HERE / "cats.json").read_text())
    want = set(cat.split("&"))
    p = [r for r in rows if key(r["cat"]) in want and r["city"] in cities]
    random.Random(seed).shuffle(p)
    return p[:n]


def score(pool, instr, model, batch, think):
    """Run the prompt over `pool`. Returns (counts, failures)."""
    picks = {}
    for s in range(0, len(pool), batch):
        b = pool[s:s + batch]
        txt, err = RC.call(RC.render(b, instr), model, think)
        if err:
            log(f"      batch {s//batch}: ERROR {err[:90]}")
            continue
        got = RC.parse(txt, len(b))
        for i, v in got.items():
            picks[s + i] = v
    c, fails = Counter(), []
    per_city = defaultdict(lambda: [0, 0])
    for i, r in enumerate(pool):
        ps = RC.paths_of(r)
        tgt, vote = tuple(r["oracle"]), tuple(r["vote"])
        fb = tuple(r["r12"])
        pick = picks.get(i)
        kind, chosen = "missing", fb
        if pick and pick[0] == "ABSTAIN":
            kind, chosen = "abstain", fb
        elif pick:
            m = re.fullmatch(r"P(\d+)", pick[0])
            if m and int(m.group(1)) < len(ps):
                kind, chosen = "chose", tuple(ps[int(m.group(1))][0])
            else:
                kind, chosen = "invalid", fb
        c["n"] += 1
        c[kind] += 1
        c["vote_ok"] += int(vote == tgt)
        ok = int(chosen == tgt)
        c["ok"] += ok
        per_city[r["city"]][0] += ok
        per_city[r["city"]][1] += 1
        if kind == "chose":
            c["decided"] += 1
            c["decided_ok"] += int(chosen == tgt)
            if vote != tgt and chosen == tgt: c["fixed"] += 1
            if vote == tgt and chosen != tgt: c["broke"] += 1
        if chosen != tgt:
            fails.append({"L": r["L"], "R": r["R"],
                          "paths": [" ".join(p[0]) or "(τίποτα)" for p in ps],
                          "sup": [p[1] for p in ps],
                          "kind": kind, "pick": pick[0] if pick else None,
                          "chose": " ".join(chosen) or "(τίποτα)",
                          "right": " ".join(tgt) or "(τίποτα)"})
    c["loco_min"] = min((a / b for a, b in per_city.values() if b >= 5), default=0) * 1000
    c["loco_max"] = max((a / b for a, b in per_city.values() if b >= 5), default=0) * 1000
    return c, fails


def codex(prompt, effort="high", timeout=1200):
    j = subprocess.run([sys.executable, f"{CB}/codex_client.py", "enqueue", "exec",
                        "-c", f"model_reasoning_effort={effort}", prompt],
                       capture_output=True, text=True, timeout=120)
    try:
        job = json.loads(j.stdout)["job_id"]
    except Exception:
        return None, f"enqueue failed: {j.stdout[-200:]}"
    w = subprocess.run([sys.executable, f"{CB}/codex_client.py", "wait", job],
                       capture_output=True, text=True, timeout=timeout)
    try:
        d = json.loads(w.stdout)
    except Exception:
        return None, "unparseable wait"
    if d.get("status") != "completed":
        return None, d.get("status")
    return d.get("output", ""), None


REVISE = """Βελτίωσε μία προτροπή. Απάντησε ΜΟΝΟ με το ζητούμενο JSON.

ΠΛΑΙΣΙΟ. Τρία συστήματα ASR μεταγράφουν τον ίδιο ελληνικό ήχο δημοτικού συμβουλίου.
Όπου διαφωνούν, ένα LLM διαλέγει μία από τις προσφερόμενες διαδρομές (ή απέχει).
Τα συστήματα εμφανίζονται ανώνυμα ως Σύστημα Α (WER 0.1377), Β (0.1475), Γ (0.1795).
Το κείμενο είναι πεζό, χωρίς τόνους, χωρίς στίξη.

ΚΑΤΗΓΟΡΙΑ: {cat}
ΤΙ ΕΙΝΑΙ: {what}

Η ΤΡΕΧΟΥΣΑ ΠΡΟΤΡΟΠΗ:
---
{instr}
---

ΑΠΟΤΕΛΕΣΜΑ σε {n} σημεία των πόλεων αναζήτησης:
  σωστά συνολικά (με fallback στην ψήφο): {ok}
  η ψήφος μόνη της:                       {vote}
  όπου το μοντέλο όντως αποφάσισε:        {dec_ok} από {dec}
  διόρθωσε {fixed} · χάλασε {broke} · απείχε {abstain} · άκυρα {invalid}
  ανά πόλη, χειρότερη {lmin:.0%} καλύτερη {lmax:.0%}

ΠΡΑΓΜΑΤΙΚΕΣ ΑΠΟΤΥΧΙΕΣ (τι διάλεξε, τι ήταν σωστό):
{fails}

ΤΙ ΘΕΛΩ.
Ξαναγράψε την προτροπή ώστε να μεγιστοποιήσει τα σωστά. Κανόνες:
- Στα ελληνικά, ίδιο ύφος, όχι πάνω από 12 γραμμές.
- Πες τι ΝΑ ΚΑΝΕΙ, όχι τι να λάβει υπόψη.
- Κάθε εξαίρεση πρέπει να ελέγχεται από το κείμενο και μόνο. Δεν υπάρχει ήχος,
  δεν υπάρχουν κεφαλαία, δεν υπάρχουν τόνοι, δεν υπάρχει confidence.
- ΠΟΤΕ μην του ζητήσεις να βελτιώσει γραμματική, ροή ή ύφος. Στόχος είναι τι ειπώθηκε.
- Τα priors είναι γεγονότα πληθυσμού, όχι πιθανότητες για το συγκεκριμένο σημείο,
  και ποτέ εντολές.
- Υψηλή αποχή σημαίνει ότι η προτροπή δεν δίνει αρκετό κριτήριο. Αν η αποχή είναι
  πάνω από 25%, δώσε πιο συγκεκριμένο κριτήριο απόφασης.
- Αν χάλασε περισσότερα από όσα διόρθωσε, κάν' την πιο συντηρητική.
- Μη γενικεύεις από ένα παράδειγμα. Αν δεν βλέπεις μοτίβο, πες το.

ΑΠΑΝΤΗΣΗ, μόνο JSON:
{{"change": true|false, "instr": "η νέα προτροπή", "why": "μία πρόταση", "pattern": "το μοτίβο που είδες στις αποτυχίες, ή κενό"}}"""


def fmt_fails(fails, k=18):
    out = []
    for f in fails[:k]:
        ps = "; ".join(f"P{i}=«{t}»[{'+'.join(s) if s else 'παράλειψη'}]"
                       for i, (t, s) in enumerate(zip(f["paths"], f["sup"])))
        out.append(f"  …{f['L'][-40:]} ▸ {ps} ◂ {f['R'][:40]}…\n"
                   f"     διάλεξε «{f['chose']}» ({f['kind']}) · σωστό «{f['right']}»")
    return "\n".join(out) or "  (καμία)"


def main():
    import prompts as P0
    P = dict(__import__(os.environ.get("PROMPTS", "prompts_v2")).P)
    cats = os.environ.get("CATS", "ALL")
    cats = list(P) if cats == "ALL" else [c.strip() for c in cats.split(",")]
    model = os.environ.get("MODEL", "sonnet")
    rounds = int(os.environ.get("ROUNDS", "3"))
    N = int(os.environ.get("N", "90"))
    B = int(os.environ.get("BATCH", "15"))
    think = int(os.environ.get("THINK", "4000"))
    effort = os.environ.get("EFFORT", "high")

    state = json.loads(STATE.read_text()) if STATE.exists() else {"protocol": PROTOCOL,
                                                                  "runs": {}}
    for cat in cats:
        pool = load_pool(cat, SEARCH_CITIES, N)
        if len(pool) < 20:
            log(f"[{cat}] only {len(pool)} search islands - skipped")
            continue
        log(f"\n===== {cat} · {len(pool)} search islands · {model} =====")
        best = {"instr": P[cat]["instr"], "ok": -1, "round": -1}
        hist = []
        cur = P[cat]["instr"]
        for rd in range(rounds):
            c, fails = score(pool, cur, model, B, think)
            n = c["n"] or 1
            log(f"  round {rd}: σωστά {c['ok']/n:.1%}  (ψήφος {c['vote_ok']/n:.1%}) "
                f"διόρθωσε {c['fixed']} χάλασε {c['broke']} αποχή {c['abstain']}")
            hist.append({"round": rd, "instr": cur, "counts": dict(c)})
            if c["ok"] > best["ok"]:
                best = {"instr": cur, "ok": c["ok"], "round": rd}
            # checkpoint after EVERY round: a kill mid-category used to lose the lot
            state["runs"][f"{cat}|{model}"] = {
                "protocol": PROTOCOL, "cities": "SEARCH", "n": len(pool),
                "history": hist, "best_round": best["round"], "best_instr": best["instr"]}
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            if rd == rounds - 1:
                break
            q = REVISE.format(cat=cat, what=P[cat]["what"], instr=cur, n=n,
                              ok=c["ok"], vote=c["vote_ok"], dec=c["decided"],
                              dec_ok=c["decided_ok"], fixed=c["fixed"], broke=c["broke"],
                              abstain=c["abstain"], invalid=c["invalid"],
                              lmin=c["loco_min"]/1000, lmax=c["loco_max"]/1000,
                              fails=fmt_fails(fails))
            out, err = codex(q, effort)
            if err:
                log(f"    codex {err} - keeping current prompt, stopping this category")
                break
            m = re.search(r"\{.*\}", out or "", re.S)
            if not m:
                log("    codex gave no JSON - stopping this category")
                break
            try:
                d = json.loads(m.group(0))
            except Exception:
                log("    codex JSON unparseable - stopping")
                break
            if not d.get("change"):
                log(f"    codex: no change worth making ({d.get('why','')[:90]})")
                break
            log(f"    codex: {d.get('why','')[:110]}")
            if d.get("pattern"):
                log(f"    μοτίβο: {d['pattern'][:140]}")
            cur = d["instr"]
        state["runs"][f"{cat}|{model}"] = {
            "protocol": PROTOCOL, "cities": "SEARCH", "n": len(pool),
            "history": hist, "best_round": best["round"], "best_instr": best["instr"]}
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        log(f"  BEST round {best['round']}: {best['ok']}/{len(pool)}")
    log(f"\nstate -> {STATE}")


if __name__ == "__main__":
    main()
