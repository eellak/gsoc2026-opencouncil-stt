"""Do the two convergent rules survive end-to-end rescoring, or only in columns?

Fable and Codex job 056cc3fd independently proposed the same two changes to the
hierarchical vote:
    R1  unresolved_two [x, y, eps]   -> emit nothing
    R2  unresolved_three [x, y, z]   -> take scribe
Both counted them in COLUMNS. Columns are not WER: replacing a column realigns
everything after it. This reconstructs the full text and rescores with the frozen
scorer, which is the only number that counts.

The rules were CHOSEN after looking at these 391 windows. They have no fitted
parameter, so leave-one-city-out changes nothing mechanically, but the CHOICE is
in-sample. Every number here is exploratory until a fresh cluster set is run.
"""
import os, sys, json, random
sys.path.insert(0,'/home/harold/opencouncil-fine-tuning')
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from eval.controlled_eval.atlas_substrate import load, TRIO
from eval.controlled_eval.column_classes import column_class
from eval.controlled_eval.msa import vote_column
from eval.controlled_eval.exp_fusion_deletions import sdi

ARMS=("W","R1","R2","R1+R2")

def job(w):
    cols=[tuple(c) for c in w["cols"]]; R=" ".join(w["ref"])
    base=[vote_column(c,w["pivot"])[0] for c in cols]
    kl=[column_class(c) for c in cols]
    out={}
    for arm in ARMS:
        sel=list(base)
        for i,k in enumerate(kl):
            if k=="unresolved_two" and arm in ("R1","R1+R2"): sel[i]=None
            if k=="unresolved_three" and arm in ("R2","R1+R2"): sel[i]=cols[i][0]
        out[arm]=sdi(R," ".join(t for t in sel if t is not None))
    return w["city"],w["meeting"],out

def wer(t): return (t[0]+t[1]+t[2])/t[3]
def add(a,b): return [a[i]+b[i] for i in range(4)]

if __name__=="__main__":
    sub=load(Path(os.environ["REPORT"]))
    rows=[]
    with ProcessPoolExecutor(max_workers=14) as p:
        for r in p.map(job, sub["windows"], chunksize=4): rows.append(r)

    def tot(arm, sel=None):
        t=[0,0,0,0]
        for c,m,o in rows:
            if sel and not sel(c,m): continue
            t=add(t,o[arm])
        return t

    print(f"{'arm':8s} {'WER':>8s} {'sub':>8s} {'del':>8s} {'ins':>8s}")
    for a in ARMS:
        t=tot(a); print(f"{a:8s} {wer(t):8.5f} {t[0]/t[3]:8.5f} {t[1]/t[3]:8.5f} {t[2]/t[3]:8.5f}")

    # paired cluster bootstrap vs W
    by=defaultdict(list)
    for r in rows: by[(r[0],r[1])].append(r)
    cl=sorted(by); rng=random.Random(20260825); NB=10000
    print()
    for a in ARMS[1:]:
        def d(sample):
            ta=[0,0,0,0]; tb=[0,0,0,0]
            for c,m,o in sample: ta=add(ta,o[a]); tb=add(tb,o["W"])
            return wer(ta)-wer(tb)
        pt=d(rows); bs=[]
        for _ in range(NB):
            s=[rng.choice(cl) for _ in cl]
            bs.append(d([r for k in s for r in by[k]]))
        bs.sort()
        # single-cluster domination
        per={k: sum(sum(o[a][:3])-sum(o["W"][:3]) for _,_,o in v) for k,v in by.items()}
        T=sum(abs(x) for x in per.values()); top=max(per.values(), key=abs)
        print(f"{a:8s} vs W: delta {pt:+.5f}  CI [{bs[250]:+.5f}, {bs[9750]:+.5f}]  "
              f"top-cluster share {abs(top)/T:.1%} of {len(per)} clusters")

    # leave-one-city-out: the rule has no fitted parameter, so this prices
    # heterogeneity, not overfitting
    cities=sorted({c for c,_,_ in rows})
    print(f"\nper-city delta of R1+R2 vs W ({len(cities)} cities):")
    ds=[]
    for city in cities:
        sel=lambda c,m,city=city: c==city
        d=wer(tot("R1+R2",sel))-wer(tot("W",sel))
        n=sum(1 for c,_,_ in rows if c==city); ds.append(d)
        print(f"  {city:16s} n={n:4d}  {d:+.5f}")
    print(f"  cities improved: {sum(1 for x in ds if x<0)}/{len(ds)}")
