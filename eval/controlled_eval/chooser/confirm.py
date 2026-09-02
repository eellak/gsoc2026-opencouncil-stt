# -*- coding: utf-8 -*-
"""The ONE confirmation run. Read the refusals before running it.

PROTOCOL (autoprompt-2026-08-25a, inherited from eval/controlled_eval/autoresearch.py):
  SEARCH   athens chalandri chania orestiada vrilissia zografou
  CONFIRM  argos samothraki sparta xylokastro argithea

This script touches the CONFIRM cities. It may be run ONCE per frozen policy. It
refuses to run twice against the same freeze, because "at most five" is not a
protection if result 1 decides who gets slot 2. To run it again you must write a
new freeze file with a new id, which is a deliberate act, not a retry.

WHAT IS FROZEN, and hashed into the freeze id: the per-category prompts, which
categories go to the LLM at all, the model, the thinking budget, the candidate
enumeration, the fallback, and the batch size. Change any of them and the previous
confirmation no longer describes what you are running.

Env: FREEZE (path to the freeze json), N (islands per category, 0 = all)
"""
import json, os, re, sys, hashlib
from collections import Counter
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_chooser as RC
import autoprompt as A

LOG = HERE / "confirm_log.json"


def freeze_id(fr):
    blob = json.dumps(fr, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def main():
    fr = json.loads(Path(os.environ["FREEZE"]).read_text())
    fid = freeze_id(fr)
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    if fid in log:
        raise SystemExit(
            f"REFUSED: freeze {fid} was already confirmed on {log[fid]['when']}.\n"
            f"A second run against the same freeze is not a retry, it is a second\n"
            f"look at the confirmation partition. Write a new freeze to run again.")

    model = fr["model"]
    think = fr.get("think", 4000)
    batch = fr.get("batch", 15)
    N = int(os.environ.get("N", "0"))
    rows = []
    for cat, pol in fr["policy"].items():
        pool = A.load_pool(cat, A.CONFIRM_CITIES, N or 10**6)
        if not pool:
            continue
        n = len(pool)
        f = lambda g: sum(1 for r in pool if g(r)) / n
        base = {"ψήφος": f(lambda r: r["ok"]),
                "R1+R2": f(lambda r: r["r12ok"]),
                "Scribe": f(lambda r: r["spans"][0] == r["oracle"]),
                "σβήσε": f(lambda r: not r["oracle"])}
        if pol["mode"] == "rule":
            got = base[pol["rule"]]
            print(f"{cat[:34]:36s} n={n:4d}  ΚΑΝΟΝΑΣ {pol['rule']:8s} {got:6.1%}", flush=True)
            rows.append({"cat": cat, "n": n, "mode": "rule", "rule": pol["rule"],
                         "score": got, "nulls": base})
            continue
        picks = {}
        for s in range(0, n, batch):
            b = pool[s:s + batch]
            txt, err = RC.call(RC.render(b, pol["instr"]), model, think)
            if err:
                print(f"  ERROR {err[:90]}", flush=True)
                continue
            for i, v in RC.parse(txt, len(b)).items():
                picks[s + i] = v
        c = Counter()
        for i, r in enumerate(pool):
            ps = RC.paths_of(r)
            fb = tuple(r["r12"])
            p = picks.get(i)
            if not p or p[0] == "ABSTAIN":
                chosen, kind = fb, ("abstain" if p else "missing")
            else:
                m = re.fullmatch(r"P(\d+)", p[0])
                if m and int(m.group(1)) < len(ps):
                    chosen, kind = tuple(ps[int(m.group(1))][0]), "chose"
                else:
                    chosen, kind = fb, "invalid"
            c[kind] += 1
            c["ok"] += int(chosen == tuple(r["oracle"]))
        got = c["ok"] / n
        bn = max(base, key=base.get)
        print(f"{cat[:34]:36s} n={n:4d}  LLM {got:6.1%}  καλύτερο null {base[bn]:6.1%} "
              f"({bn})  Δ {got-base[bn]:+.1%}", flush=True)
        rows.append({"cat": cat, "n": n, "mode": "llm", "score": got,
                     "nulls": base, "kinds": dict(c)})

    tot_n = sum(r["n"] for r in rows)
    llm = [r for r in rows if r["mode"] == "llm"]
    d = sum(r["n"] * (r["score"] - max(r["nulls"].values())) for r in llm)
    print(f"\nνησιά επιβεβαίωσης {tot_n} · LLM σε {sum(r['n'] for r in llm)}")
    if llm:
        print(f"σταθμισμένο Δ έναντι καλύτερου null, μόνο στις LLM κατηγορίες: "
              f"{d/sum(r['n'] for r in llm):+.1%}")
    import datetime
    log[fid] = {"when": datetime.datetime.now().isoformat(timespec="seconds"),
                "rows": rows, "freeze": fr}
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(f"\nfreeze {fid} καταγράφηκε -> {LOG.name}. Δεύτερο τρέξιμο θα απορριφθεί.")


if __name__ == "__main__":
    main()
