"""Confusion networks per disagreement island, and the category each one falls into.

Category scheme is Codex job f45215fb's, adopted before any of it was counted:

  FOUR ORDERED SPECIAL ROUTES, tested in this order so every island lands in
  exactly one bucket:
     NUM   every candidate is equal once numbers are normalised
     FILL  candidates differ only by tokens from a frozen Greek filler lexicon
     DUP   candidates differ only by exact duplication of adjacent context
     LOW   equal only after BOTH filler removal and duplicate collapse

  THEN 13 ISLAND-LEVEL SIGNATURES, on the three systems' whole spans - not on one
  arbitrarily chosen column. Which PAIR agrees is part of the signature, because
  "scribe+soniox agree" is not the same evidence as "soniox+ours agree".
     PAIR_x    two identical non-empty spans, third different and non-empty  (3)
     PAIRE_x   two identical non-empty spans, third empty                    (3)
     SOLO_x    one non-empty span, two empty                                 (3)
     TWO_x     two different non-empty spans, third empty                    (3)
     THREE     three different non-empty spans                               (1)

  each split ATOMIC / COMPOSITE (composite = ragged token counts or split_merge).
"""
import os, sys, json, re, hashlib, unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,'/home/harold/opencouncil-fine-tuning')
from eval.controlled_eval.atlas_substrate import load, TRIO
from eval.controlled_eval.column_classes import column_class, split_merge_columns
from eval.controlled_eval.msa import oracle_select, vote_column
from eval.controlled_eval.error_atlas import islands

SHORT = ["scribe","soniox","ours"]
FILLERS = {"εε","εεε","εεεε","ε","μμ","μμμ","αα","ααα","ααμ","χμ","ε ε","ναι ναι"}
UNITS={"μηδεν":0,"ενα":1,"δυο":2,"τρια":3,"τρεις":3,"τεσσερα":4,"τεσσερις":4,"πεντε":5,
 "εξι":6,"εφτα":7,"επτα":7,"οχτω":8,"οκτω":8,"εννια":9,"εννεα":9,"δεκα":10,"εντεκα":11,
 "δωδεκα":12,"δεκατρια":13,"εικοσι":20,"τριαντα":30,"σαραντα":40,"πενηντα":50,
 "εξηντα":60,"εβδομηντα":70,"ογδοντα":80,"ενενηντα":90,"εκατο":100,"χιλια":1000}
def numkey(t):
    if re.fullmatch(r"\d+", t): return "#"+str(int(t))
    if t in UNITS: return "#"+str(UNITS[t])
    return t
def norm_num(span): return tuple(numkey(t) for t in span)
def strip_fill(span): return tuple(t for t in span if t not in FILLERS)
def strip_dup(span, ctx): return tuple(t for t in span if t not in ctx)

def signature(spans):
    ne=[i for i,s in enumerate(spans) if s]
    if len(ne)==1: return "SOLO_"+SHORT[ne[0]]
    if len(ne)==2:
        a,b=ne
        if spans[a]==spans[b]:
            miss=[i for i in range(3) if i not in ne][0]
            return "PAIRE_"+"+".join(SHORT[i] for i in ne)
        return "TWO_no_"+SHORT[[i for i in range(3) if i not in ne][0]]
    d=defaultdict(list)
    for i,s in enumerate(spans): d[s].append(i)
    if len(d)==2:
        pair=[v for v in d.values() if len(v)==2][0]
        return "PAIR_"+"+".join(SHORT[i] for i in pair)
    return "THREE"

def job(w):
    cols=[tuple(c) for c in w["cols"]]; ref=w["ref"]
    osel=oracle_select(cols,ref); wsel=[vote_column(c,w["pivot"])[0] for c in cols]
    sm=split_merge_columns(cols)
    out=[]
    for (s,e) in islands(cols):
        span=list(range(s,e))
        spans=[tuple(x for x in (cols[i][k] for i in span) if x is not None) for k in range(3)]
        ragged=len({len(x) for x in spans})>1
        composite = ragged or any(i in sm for i in span)
        # anchors
        L=[cols[i][0] for i in range(max(0,s-12),s) if column_class(cols[i])=="agree"]
        R=[cols[i][0] for i in range(e,min(len(cols),e+12)) if column_class(cols[i])=="agree"]
        ctx=set(L[-3:])|set(R[:3])
        uniq=set(spans)
        # --- ordered special routes
        route=None
        if len({norm_num(x) for x in spans})==1: route="NUM"
        elif len({strip_fill(x) for x in spans})==1: route="FILL"
        elif len({strip_dup(x,ctx) for x in spans})==1: route="DUP"
        elif len({strip_dup(strip_fill(x),ctx) for x in spans})==1: route="LOW"
        cat = route if route else (signature(spans)+("|COMPOSITE" if composite else "|ATOMIC"))
        target=tuple(t for t in (osel[i] for i in span) if t is not None)
        vote=tuple(t for t in (wsel[i] for i in span) if t is not None)
        r=[]
        for i in span:
            k=column_class(cols[i])
            if k=="unresolved_two": continue
            t=cols[i][0] if k=="unresolved_three" else wsel[i]
            if t is not None: r.append(t)
        out.append({"win":w["id"],"city":w["city"],"mt":w["meeting"],"st":s,"len":e-s,
            "cat":cat,"route":route or "","sig":signature(spans),
            "comp":int(composite),"ragged":int(ragged),
            "spans":[list(x) for x in spans],
            "vote":list(vote),"r12":r,"oracle":list(target),
            "ok":int(vote==target),"r12ok":int(tuple(r)==target),
            "L":" ".join(L[-8:]),"R":" ".join(R[:8])})
    return out

if __name__=="__main__":
    sub=load(Path(os.environ["REPORT"])); rows=[]
    with ProcessPoolExecutor(max_workers=14) as p:
        for r in p.map(job, sub["windows"], chunksize=4): rows+=r
    Path(os.environ["OUT"]).write_text(json.dumps(rows,ensure_ascii=False,separators=(",",":")))
    C=Counter(r["cat"] for r in rows)
    print(f"islands {len(rows)}  categories {len(C)}")
    print(f"{'category':34s} {'n':>6s} {'vote ok':>8s} {'R1+R2 ok':>9s}")
    for k,n in C.most_common():
        a=[r for r in rows if r["cat"]==k]
        print(f"{k:34s} {n:6d} {sum(x['ok'] for x in a)/n:7.1%} {sum(x['r12ok'] for x in a)/n:8.1%}")
