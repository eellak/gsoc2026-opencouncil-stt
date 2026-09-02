# -*- coding: utf-8 -*-
"""Headless chooser: one category at a time, batched, scored against the oracle.

Islands are split into categories ALGORITHMICALLY (cats.py, no reference used), then
each category goes to the model with its own instruction. The model only ever picks
one of the offered paths or abstains; it can never write new text, so it cannot
invent a fourth option.

Scoring is IN-SAMPLE: `oracle` reads the reference. These numbers price the idea,
they do not confirm it.

Env: CAT, MODEL (sonnet|opus), N (default 120), BATCH (default 15),
     THINK (thinking tokens, default 4000), PROMPTS (prompts module, default prompts)
"""
import json, os, re, subprocess, sys, random
from collections import Counter
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

SHORT = ["scribe", "soniox", "ours"]
NM = {"scribe": "Σύστημα Α", "soniox": "Σύστημα Β", "ours": "Σύστημα Γ"}

SYSTEM = """Επιλέγεις μία διαδρομή μέσα σε ένα δίκτυο σύγχυσης ελληνικής αναγνώρισης ομιλίας.

Ο στόχος σου είναι η ΠΙΣΤΟΤΗΤΑ σε αυτό που ειπώθηκε. Δεν ξαναγράφεις, δεν διορθώνεις τη
γραμματική, δεν βελτιώνεις το ύφος, δεν μετατρέπεις τον προφορικό λόγο σε πρακτικά. Οι
επαναλήψεις, οι δισταγμοί και τα λάθη του ομιλητή μπορεί να είναι ακριβώς αυτό που ειπώθηκε.

Το κείμενο είναι πεζό, χωρίς τόνους και χωρίς στίξη. Μην προσθέτεις τίποτα από αυτά.

Διαλέγεις ΜΟΝΟ μία από τις προσφερόμενες διαδρομές. Η κενή διαδρομή σημαίνει ότι δεν
γράφεται τίποτα εκεί - δεν σημαίνει ότι ο ήχος ήταν σιωπηλός. Αν τα στοιχεία δεν
επαρκούν, γράψε ABSTAIN και θα κρατηθεί η υπάρχουσα ψήφος.

Η ταυτότητα του συστήματος είναι ένδειξη, όχι εξουσία: το τοπικό συμφραζόμενο υπερισχύει
ενός στατιστικού prior.

Απαντάς ΜΟΝΟ με JSON, χωρίς σχόλια και χωρίς markdown."""


def paths_of(row):
    """Candidate paths: each distinct whole-system span, the empty path, and R1+R2.

    Fable's finding: R1+R2 MIXES systems, so it reaches answers no whole-system path
    can. In THREE|COMPOSITE it lifts the candidate ceiling from 63.3% to 71.9%. It
    costs one extra line in the prompt and it is also the fallback when the model
    abstains, so abstention lands on the better baseline rather than on the vote.
    """
    out, seen = [], {}
    for i, s in enumerate(row["spans"]):
        t = tuple(s)
        if t in seen:
            seen[t].append(SHORT[i])
            continue
        seen[t] = [SHORT[i]]
        out.append(t)
    for extra in (tuple(row.get("r12") or ()), ()):
        if extra not in seen:
            seen[extra] = []
            out.append(extra)
    return [(list(t), seen[t]) for t in out]


def render(rows, instr):
    b = ["ΚΑΝΟΝΑΣ ΓΙΑ ΑΥΤΗ ΤΗΝ ΚΑΤΗΓΟΡΙΑ:", "", instr, "",
         "ΠΛΑΙΣΙΟ: συνεδρίαση ελληνικού δημοτικού συμβουλίου. Το λεξιλόγιο είναι συχνά "
         "τυπικό και διοικητικό, αλλά ο λόγος είναι προφορικός: κατακερματισμένος, "
         "επαναληπτικός, συχνά αντιγραμματικός.", "",
         f"Ακολουθούν {len(rows)} σημεία διαφωνίας. Για κάθε ένα διάλεξε μία διαδρομή.", ""]
    for n, r in enumerate(rows):
        b.append(f"### σημείο {n}")
        b.append(f"αριστερά: …{r['L']}")
        for k, (toks, sup) in enumerate(paths_of(r)):
            who = " + ".join(NM[s] for s in sup) if sup else "παράλειψη"
            txt = " ".join(toks) if toks else "(τίποτα)"
            b.append(f"  P{k}: «{txt}»   [{who}]")
        b.append(f"δεξιά: {r['R']}…")
        b.append("")
    b += ["Απάντησε με έναν πίνακα JSON, ένα αντικείμενο ανά σημείο, με τη σειρά:",
          '[{"i":0,"p":"P0","c":"high"}, {"i":1,"p":"ABSTAIN","c":"low"}, ...]',
          '"p" είναι το αναγνωριστικό της διαδρομής ή ABSTAIN. "c" είναι high, medium ή low.',
          "Καμία άλλη έξοδος."]
    return "\n".join(b)


def call(prompt, model, think):
    env = dict(os.environ, MAX_THINKING_TOKENS=str(think))
    p = subprocess.run(
        ["claude", "-p", prompt, "--model", model, "--output-format", "json",
         "--append-system-prompt", SYSTEM, "--allowed-tools", "",
         "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}})],
        capture_output=True, text=True, env=env, timeout=900)
    if p.returncode != 0:
        return None, p.stderr[-400:]
    try:
        d = json.loads(p.stdout)
        if isinstance(d, list):
            d = [x for x in d if x.get("type") == "result"][-1]
    except Exception as e:
        return None, f"{e}: {p.stdout[-300:]}"
    # A credit or rate limit comes back as a NORMAL result whose text is an error
    # message. Silently falling through to the fallback made a whole model
    # comparison read as "both models tied" when neither had answered at all.
    if d.get("is_error"):
        return None, f"is_error: {str(d.get('result'))[:200]}"
    r = d.get("result", "") or ""
    if "[" not in r:
        return None, f"no JSON in result: {r[:200]}"
    return r, None


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
            out[i] = (str(o.get("p", "")).strip().upper(), o.get("c", ""))
    return out


def main():
    cat = os.environ["CAT"]
    model = os.environ.get("MODEL", "sonnet")
    N = int(os.environ.get("N", "120"))
    B = int(os.environ.get("BATCH", "15"))
    think = int(os.environ.get("THINK", "4000"))
    P = __import__(os.environ.get("PROMPTS", "prompts_v2")).P

    rows = [r for r in json.loads((HERE / "cats.json").read_text())]
    def key(k):
        return k.replace("|COMPOSITE", "") if k.startswith(("TWO_", "SOLO_", "PAIRE_")) else k
    pool = [r for r in rows if key(r["cat"]) == cat]
    if not pool:
        raise SystemExit(f"no islands in category {cat}")
    random.Random(20260825).shuffle(pool)
    pool = pool[:N]
    instr = P[cat]["instr"]

    picks, errs = {}, []
    for s in range(0, len(pool), B):
        batch = pool[s:s + B]
        txt, err = call(render(batch, instr), model, think)
        if err:
            errs.append(err)
            print(f"  batch {s//B}: ERROR {err[:120]}", flush=True)
            continue
        got = parse(txt, len(batch))
        if not got:
            errs.append("unparseable")
            print(f"  batch {s//B}: unparseable -> {(txt or '')[:160]}", flush=True)
            continue
        for i, v in got.items():
            picks[s + i] = v
        print(f"  batch {s//B}: {len(got)}/{len(batch)}", flush=True)

    # ---- score
    c = Counter()
    detail = []
    for i, r in enumerate(pool):
        ps = paths_of(r)
        tgt = tuple(r["oracle"])
        vote = tuple(r["vote"])
        c["n"] += 1
        c["vote_ok"] += int(vote == tgt)
        c["r12_ok"] += int(tuple(r["r12"]) == tgt)
        fb = tuple(r["r12"])          # the frozen fallback, not the vote
        pick = picks.get(i)
        if not pick:
            c["missing"] += 1
            chosen = fb
            kind = "missing"
        elif pick[0] == "ABSTAIN":
            c["abstain"] += 1
            chosen = fb
            kind = "abstain"
        else:
            m = re.fullmatch(r"P(\d+)", pick[0])
            if not m or int(m.group(1)) >= len(ps):
                c["invalid"] += 1
                chosen = fb
                kind = "invalid"
            else:
                chosen = tuple(ps[int(m.group(1))][0])
                kind = "chose"
        c["llm_ok"] += int(chosen == tgt)
        if kind == "chose":
            c["decided"] += 1
            c["decided_ok"] += int(chosen == tgt)
            if vote != tgt and chosen == tgt:
                c["fixed"] += 1
            if vote == tgt and chosen != tgt:
                c["broke"] += 1
        detail.append({"i": i, "city": r["city"], "kind": kind,
                       "pick": pick[0] if pick else None,
                       "conf": pick[1] if pick else None,
                       "paths": [p[0] for p in ps], "sup": [p[1] for p in ps],
                       "vote": list(vote), "chosen": list(chosen), "oracle": list(tgt),
                       "L": r["L"], "R": r["R"]})

    n = c["n"]
    print(f"\n=== {cat} · {model} · think={think} · n={n} ===")
    print(f"  ψήφος σωστή     {c['vote_ok']/n:6.1%}")
    print(f"  R1+R2 σωστό     {c['r12_ok']/n:6.1%}")
    print(f"  LLM σωστό       {c['llm_ok']/n:6.1%}   <- με fallback στην ψήφο")
    if c["decided"]:
        print(f"  όπου αποφάσισε  {c['decided_ok']/c['decided']:6.1%}  (n={c['decided']})")
    print(f"  διόρθωσε {c['fixed']}   χάλασε {c['broke']}   "
          f"abstain {c['abstain']}   invalid {c['invalid']}   missing {c['missing']}")
    out = HERE / f"run_{cat.replace('|','_').replace('+','-')}_{model}.json"
    out.write_text(json.dumps({"cat": cat, "model": model, "think": think,
                               "counts": dict(c), "errors": errs, "detail": detail},
                              ensure_ascii=False, indent=1))
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
