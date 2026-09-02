# -*- coding: utf-8 -*-
"""Every island the frozen policy empties, with context, for a human to judge.

The WER measurement cannot tell a deleted «εε» from a deleted surname. This dumps
the actual deletions so a person can. Confirmation cities only - the same 169
windows the WER number describes.
"""
import json, os, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "/home/harold/opencouncil-fine-tuning")
from eval.controlled_eval.atlas_substrate import load
import autoprompt as A
import reconstruct as RE

fr = json.loads(Path(os.environ["FREEZE"]).read_text())
pol = fr["policy"]
sub = load(Path(os.environ["REPORT"]))
wins = [w for w in sub["windows"] if w["city"] in A.CONFIRM_CITIES]

FILL = {"εε","εεε","ε","μμ","μμμ","αα","ααα"}
out, c = [], Counter()
for w in wins:
    isl, tail, wsel, cols = RE.window_parts(w)
    for i in isl:
        p = pol.get(i["cat"])
        # a rule-emptied island; LLM islands are judged separately
        if not p or p["mode"] != "rule":
            continue
        got = RE.RULES[p["rule"]](i)
        lost = [t for t in i["vote"] if t not in got]
        if not (i["vote"] and not got):
            continue                      # only whole-island deletions
        kind = ("δισταγμός" if all(t in FILL for t in i["vote"])
                else "επανάληψη" if all(t in (i["L"].split()[-3:] + i["R"].split()[:3])
                                        for t in i["vote"])
                else "λέξεις")
        c[kind] += 1; c["σύνολο"] += 1
        out.append({"city": w["city"], "meeting": w["meeting"], "win": w["id"],
                    "cat": i["cat"], "rule": p["rule"],
                    "kind": kind, "lost": i["vote"], "oracle": [],
                    "L": i["L"], "R": i["R"], "spans": i["spans"],
                    # position of the island inside the window, as a fraction, so a
                    # short clip can be cut near it. Columns are a crude proxy for
                    # time - good enough for a 16 s clip, useless for anything finer.
                    "pos": i["s"] / max(1, len(cols))})
# was the deletion right? oracle for that island
rows = json.loads((HERE / "cats.json").read_text())
key = lambda r: (r["city"], r["L"], r["R"])
orc = {}
for r in rows:
    orc.setdefault((r["city"], r["L"], r["R"]), r["oracle"])
right = 0
for o in out:
    tgt = orc.get((o["city"], o["L"], o["R"]))
    o["oracle"] = tgt if tgt is not None else None
    if tgt == []:
        right += 1
print(f"διαγραφές ολόκληρου νησιού: {c['σύνολο']}")
for k in ("λέξεις", "δισταγμός", "επανάληψη"):
    if c[k]: print(f"  {k:12s} {c[k]:5d}  {c[k]/c['σύνολο']:5.1%}")
known = [o for o in out if o["oracle"] is not None]
if known:
    ok = sum(1 for o in known if o["oracle"] == [])
    print(f"\nμε γνωστή απάντηση: {len(known)} · σωστή διαγραφή {ok} = {ok/len(known):.1%}")
Path("deletions.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",",":")))
print(f"-> deletions.json ({len(out)} εγγραφές)")
