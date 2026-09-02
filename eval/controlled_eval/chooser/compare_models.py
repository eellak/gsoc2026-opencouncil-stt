# -*- coding: utf-8 -*-
"""Same prompt, same islands, different model. The only variable is the model.

BIAS TO DECLARE: the prompts were optimised with Sonnet in the loop, so they are
tuned to Sonnet's failure modes. A tie therefore favours Opus, and an Opus loss is
weak evidence. The clean version would re-run the search with Opus in the loop,
which costs the whole search again.

Reports the paired breakdown, not just two rates: how often they agree, and who is
right when they disagree. Two models at 60% that disagree on half their answers are
a different situation from two models at 60% that agree everywhere.

Env: MODELS (comma list, default "sonnet,opus"), THINK, BATCH, CATS
"""
import json, os, re, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_chooser as RC
import autoprompt as A

OUT = HERE / "model_compare.json"


def run(pool, instr, model, batch, think):
    picks = {}
    for s in range(0, len(pool), batch):
        b = pool[s:s + batch]
        txt, err = RC.call(RC.render(b, instr), model, think)
        if err:
            print(f"    {model} batch {s//batch}: ERROR {err[:80]}", flush=True)
            continue
        for i, v in RC.parse(txt, len(b)).items():
            picks[s + i] = v
    out = []
    for i, r in enumerate(pool):
        ps = RC.paths_of(r)
        fb = tuple(r["r12"])
        p = picks.get(i)
        if not p or p[0] == "ABSTAIN":
            out.append((fb, "abstain" if p else "missing"))
            continue
        m = re.fullmatch(r"P(\d+)", p[0])
        if m and int(m.group(1)) < len(ps):
            out.append((tuple(ps[int(m.group(1))][0]), "chose"))
        else:
            out.append((fb, "invalid"))
    return out


def main():
    models = os.environ.get("MODELS", "sonnet,opus").split(",")
    think = int(os.environ.get("THINK", "4000"))
    batch = int(os.environ.get("BATCH", "15"))
    st = json.loads((HERE / "autoprompt_state.json").read_text())
    only = os.environ.get("CATS", "")
    only = set(only.split(",")) if only else None

    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k, v in st["runs"].items():
        cat = k.rsplit("|", 1)[0]
        if only and cat not in only:
            continue
        pool = A.load_pool(cat, A.SEARCH_CITIES, v["n"])
        instr = v["best_instr"]
        print(f"\n===== {cat} · n={len(pool)} · frozen prompt from round {v['best_round']} =====",
              flush=True)
        got = {}
        for mdl in models:
            got[mdl] = run(pool, instr, mdl, batch, think)
            ok = sum(1 for (c, _), r in zip(got[mdl], pool) if c == tuple(r["oracle"]))
            dec = sum(1 for _, kd in got[mdl] if kd == "chose")
            print(f"  {mdl:8s} σωστά {ok/len(pool):5.1%}  αποφάσισε {dec}/{len(pool)}", flush=True)
        if len(models) == 2:
            a, b = models
            c = Counter()
            for i, r in enumerate(pool):
                t = tuple(r["oracle"])
                ca, cb = got[a][i][0], got[b][i][0]
                c["same" if ca == cb else "diff"] += 1
                if ca != cb:
                    c[f"{a}_right"] += int(ca == t)
                    c[f"{b}_right"] += int(cb == t)
                    c["both_wrong"] += int(ca != t and cb != t)
            print(f"  συμφωνούν {c['same']}/{len(pool)} · διαφωνούν {c['diff']}: "
                  f"{a} σωστό {c[a+'_right']}, {b} σωστό {c[b+'_right']}, "
                  f"και τα δύο λάθος {c['both_wrong']}", flush=True)
            res[cat] = {"n": len(pool), "paired": dict(c),
                        **{m: {"ok": sum(1 for (x, _), r in zip(got[m], pool)
                                         if x == tuple(r["oracle"])),
                               "decided": sum(1 for _, kd in got[m] if kd == "chose")}
                           for m in models}}
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
