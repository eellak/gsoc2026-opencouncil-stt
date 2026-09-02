# -*- coding: utf-8 -*-
"""Does batch size change the answer, and does a cheaper model break?

Two questions the cost discussion cannot answer by arithmetic:
  - bigger batches save the repeated system prompt, but do they cost accuracy or
    lose the numbering?
  - Sonnet and Opus differ by less than their own run-to-run variance here, so how
    far down the model ladder can this go before it breaks?

Same islands, same frozen prompt, everything else held. Sequential, so wall time is
comparable.
"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_chooser as RC
import autoprompt as A

CAT = os.environ.get("CAT", "PAIR_soniox+ours|ATOMIC")
st = json.loads((HERE / "autoprompt_state.json").read_text())
instr = [v for k, v in st["runs"].items() if k.rsplit("|", 1)[0] == CAT][0]["best_instr"]
pool = A.load_pool(CAT, A.SEARCH_CITIES, 60)
tgt = [tuple(r["oracle"]) for r in pool]

def run(model, batch, think=4000):
    t0 = time.time(); picks = {}; calls = 0; bad = 0
    for s in range(0, len(pool), batch):
        b = pool[s:s + batch]
        txt, err = RC.call(RC.render(b, instr), model, think)
        calls += 1
        if err:
            bad += 1; continue
        got = RC.parse(txt, len(b))
        if len(got) < len(b) * 0.9:
            bad += 1
        for i, v in got.items():
            picks[s + i] = v
    ok = miss = 0
    for i, r in enumerate(pool):
        ps = RC.paths_of(r); p = picks.get(i)
        if not p or p[0] == "ABSTAIN":
            chosen = tuple(r["r12"]); miss += (p is None)
        else:
            import re
            m = re.fullmatch(r"P(\d+)", p[0])
            chosen = (tuple(ps[int(m.group(1))][0])
                      if m and int(m.group(1)) < len(ps) else tuple(r["r12"]))
        ok += int(chosen == tgt[i])
    return {"ok": ok / len(pool), "calls": calls, "bad_batches": bad,
            "unanswered": miss, "sec": round(time.time() - t0, 1)}

print(f"κατηγορία {CAT} · {len(pool)} νησιά · παγωμένη προτροπή\n")
print(f"{'μοντέλο':9s} {'batch':>6s} {'σωστά':>7s} {'κλήσεις':>8s} {'αναπ.':>6s} {'κακά':>5s} {'δευτ.':>7s}")
for model, batch in [("sonnet", 15), ("sonnet", 60), ("haiku", 15), ("haiku", 60),
                     ("opus", 60)]:
    r = run(model, batch)
    print(f"{model:9s} {batch:6d} {r['ok']:6.1%} {r['calls']:8d} "
          f"{r['unanswered']:6d} {r['bad_batches']:5d} {r['sec']:7.1f}", flush=True)
