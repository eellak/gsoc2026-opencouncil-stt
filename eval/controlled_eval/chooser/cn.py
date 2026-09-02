"""Step 1 of Codex job f45215fb: does the right path even EXIST in the candidate set?

No LLM calls, no money. If the oracle's own path through an island is usually
absent from the top-K candidates we would show a model, prompt engineering cannot
rescue it and the whole confusion-network idea dies here.

SLOTS. An island is `atomic` when all three systems put the same number of tokens
in it and no column is split_merge: then one column is one slot. Otherwise it is
`composite` and the WHOLE island collapses to ONE slot whose arcs are each system's
entire substring. Collapsing is not cosmetic - it is what stops a path taking half
of scribe's one-token spelling and half of soniox's two-token spelling, which the
per-column vote cannot currently prevent.

TARGET. The oracle's own selection restricted to the island. That is a lattice path,
so containment is well defined. It is NOT semantic truth: the oracle minimises edit
distance to the reference along one fixed alignment and sometimes picks tokens a
human would call wrong. Read this as feasibility of the candidate set, nothing more.
"""
import os, sys, json, hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,'/home/harold/opencouncil-fine-tuning')
from eval.controlled_eval.atlas_substrate import load, TRIO
from eval.controlled_eval.column_classes import column_class, split_merge_columns
from eval.controlled_eval.msa import oracle_select, vote_column
from eval.controlled_eval.error_atlas import islands

CAP = 16
BEAM = 64

def slots_of(cols, span, sm):
    """Returns a list of slots; each slot is a list of (tuple_of_tokens, support_set)."""
    subs=[[cols[i][k] for i in span if cols[i][k] is not None] for k in range(3)]
    ragged = len({len(s) for s in subs})>1
    composite = ragged or any(i in sm for i in span)
    def arcs(cands):
        d={}
        for k,t in enumerate(cands):
            d.setdefault(tuple(t),set()).add(k)
        return sorted(d.items(), key=lambda kv:(-len(kv[1]), kv[0]))
    if composite:
        return [arcs([tuple(s) for s in subs])], True
    out=[]
    for i in span:
        c=cols[i]
        out.append(arcs([() if c[k] is None else (c[k],) for k in range(3)]))
    return out, False

def enumerate_paths(slots, always):
    """Deterministic beam over slots. `always` are paths forced into the set."""
    beam=[((), 0, 0, None)]   # tokens, support, switches, last_owner
    for sl in slots:
        nxt=[]
        for toks,sup,sw,last in beam:
            for t,owners in sl:
                own = min(owners)
                nxt.append((toks+t, sup+len(owners),
                            sw + (0 if last is None or own==last else 1), own))
        nxt.sort(key=lambda p:(-p[1], p[2], hashlib.md5(str(p[0]).encode()).hexdigest()))
        seen=set(); beam=[]
        for p in nxt:
            if p[0] in seen: continue
            seen.add(p[0]); beam.append(p)
            if len(beam)>=BEAM: break
    ranked=[p[0] for p in beam]
    out=[]
    for a in always:
        if a not in out: out.append(a)
    for p in ranked:
        if p not in out: out.append(p)
        if len(out)>=max(CAP,len(always)): break
    return out

def job(w):
    cols=[tuple(c) for c in w["cols"]]; ref=w["ref"]
    osel=oracle_select(cols,ref); wsel=[vote_column(c,w["pivot"])[0] for c in cols]
    sm=split_merge_columns(cols)
    res=[]
    for (s,e) in islands(cols):
        span=list(range(s,e))
        slots, composite = slots_of(cols,span,sm)
        target=tuple(t for t in (osel[i] for i in span) if t is not None)
        vote=tuple(t for t in (wsel[i] for i in span) if t is not None)
        # the frozen incumbent R1+R2 path
        r=[]
        for i in span:
            k=column_class(cols[i])
            if k=="unresolved_two": continue
            t=cols[i][0] if k=="unresolved_three" else wsel[i]
            if t is not None: r.append(t)
        sysp=[tuple(x for x in (cols[i][k] for i in span) if x is not None) for k in range(3)]
        always=[]
        for a in sysp+[vote,tuple(r)]:
            if a not in always: always.append(a)
        paths=enumerate_paths(slots, always)
        try: rank=paths.index(target)
        except ValueError: rank=-1
        res.append({"len":e-s,"composite":int(composite),"npaths":len(paths),
                    "rank":rank,"vote_ok":int(vote==target),
                    "in_sys":int(target in sysp),"nslots":len(slots),
                    "cls":column_class(cols[s])})
    return res

if __name__=="__main__":
    sub=load(Path(os.environ["REPORT"])); rows=[]
    with ProcessPoolExecutor(max_workers=14) as p:
        for r in p.map(job, sub["windows"], chunksize=4): rows+=r
    N=len(rows)
    def sh(f, sel=None):
        a=[r for r in rows if sel is None or sel(r)]
        return (sum(1 for r in a if f(r))/len(a) if a else 0.0, len(a))
    print(f"islands {N}")
    for K in (1,3,8,16):
        v,_=sh(lambda r,K=K: 0<=r["rank"]<K)
        print(f"  oracle path inside top-{K:2d}: {v:6.1%}")
    v,_=sh(lambda r: r["rank"]>=0); print(f"  inside the full candidate set: {v:6.1%}")
    v,_=sh(lambda r: r["in_sys"]);  print(f"  equal to some SYSTEM's own span: {v:6.1%}")
    v,_=sh(lambda r: r["vote_ok"]); print(f"  the vote already has it:         {v:6.1%}")
    print("\nby structure (top-16 containment):")
    for name,sel in (("atomic",lambda r:not r["composite"]),("composite",lambda r:r["composite"])):
        v,n=sh(lambda r:0<=r["rank"]<CAP, sel); print(f"  {name:10s} n={n:6d}  {v:6.1%}")
    print("\nby island length (top-16):")
    for L in (1,2,3,4):
        sel=(lambda r,L=L: r["len"]>=4) if L==4 else (lambda r,L=L: r["len"]==L)
        v,n=sh(lambda r:0<=r["rank"]<CAP, sel); print(f"  len {L}{'+' if L==4 else ' '} n={n:6d}  {v:6.1%}")
    print("\namong islands the vote gets WRONG (the recoverable set):")
    bad=lambda r: not r["vote_ok"]
    for K in (3,8,16):
        v,n=sh(lambda r,K=K:0<=r["rank"]<K, bad); print(f"  top-{K:2d}: {v:6.1%}  (n={n})")
    v,n=sh(lambda r:r["in_sys"], bad); print(f"  equal to some system's own span: {v:6.1%}")
    print("\ncandidate-set size:", dict(Counter(r["npaths"] for r in rows).most_common(6)))
