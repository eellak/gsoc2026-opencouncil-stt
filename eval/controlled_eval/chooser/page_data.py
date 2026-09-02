"""Data layer for the disagreement explorer. Islands, not columns, are the unit.

Two galleries, kept apart on Codex's instruction (job 17d6055a):
  - `random`     simple random sample of ALL non-agree islands, fixed seed, known
                 inclusion probability. Estimates a typical island.
  - `strata`     predeclared strata for the rare interesting cases, sampled
                 uniformly WITHIN each stratum, with n/N and the oversampling
                 factor carried on every row so they are never pooled.
"""
import os, sys, json, random
sys.path.insert(0,'/home/harold/opencouncil-fine-tuning')
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from eval.controlled_eval.atlas_substrate import load, TRIO
from eval.controlled_eval.column_classes import column_class, column_flags, split_merge_columns
from eval.controlled_eval.msa import oracle_select, vote_column
from eval.controlled_eval.error_atlas import islands
from eval.controlled_eval.scoring import edist

def job(w):
    cols=[tuple(c) for c in w["cols"]]; ref=w["ref"]
    osel=oracle_select(cols,ref); wsel=[vote_column(c,w["pivot"])[0] for c in cols]
    sm=split_merge_columns(cols)
    out=[]
    for (s,e) in islands(cols):
        span=list(range(s,e))
        toks=lambda k:[cols[i][k] for i in span if cols[i][k] is not None]
        wt=[wsel[i] for i in span if wsel[i] is not None]
        ot=[osel[i] for i in span if osel[i] is not None]
        sysspans=[toks(k) for k in range(3)]
        # nearest agree anchors, the only context that is not itself disputed
        L=[cols[i][0] for i in range(max(0,s-7),s) if column_class(cols[i])=="agree"]
        R=[cols[i][0] for i in range(e,min(len(cols),e+7)) if column_class(cols[i])=="agree"]
        cls=[column_class(cols[i]) for i in span]
        fl=[column_flags(cols[i]) for i in span]
        d=[]
        for a in range(3):
            for b in range(a+1,3):
                d.append(edist(list(" ".join(sysspans[a])), list(" ".join(sysspans[b]))))
        out.append({
            "win":w["id"],"city":w["city"],"meeting":w["meeting"],"start":s,"len":e-s,
            "sys":{TRIO[k]:sysspans[k] for k in range(3)},
            "vote":wt,"oracle":ot,
            "classes":cls,
            "ragged": len({len(x) for x in sysspans})>1,
            "split_merge": any(i in sm for i in span),
            "max_chardist": max([f["max_char_dist"] for f in fl] or [0]),
            "any_strict_hom": any(f["strict_homophone"] for f in fl),
            "any_partial_hom": any(f["partial_homophone"] for f in fl),
            "pair_chardist": d,
            "vote_eq_oracle": wt==ot,
            "left":" ".join(L[-5:]),"right":" ".join(R[:5]),
        })
    return out

if __name__=="__main__":
    sub=load(Path(os.environ["REPORT"])); OUT=Path(os.environ["OUTDIR"])
    allisl=[]
    with ProcessPoolExecutor(max_workers=14) as p:
        for r in p.map(job, sub["windows"], chunksize=4): allisl+=r
    N=len(allisl)
    rng=random.Random(20260825)

    # ---- gallery 1: simple random, known inclusion probability
    k=400
    rand=rng.sample(allisl,k)
    for r in rand: r["_frame"]={"mode":"random","N":N,"n":k,"p":k/N}

    # ---- gallery 2: predeclared strata
    STRATA={
      "near_char_unresolved": lambda r: r["max_chardist"]<=2 and not r["any_strict_hom"] and any(c=="unresolved_three" for c in r["classes"]),
      "homophone":            lambda r: r["any_strict_hom"],
      "split_merge":          lambda r: r["split_merge"],
      "ragged_long":          lambda r: r["ragged"] and r["len"]>=4,
      "vote_wrong_len1":      lambda r: r["len"]==1 and not r["vote_eq_oracle"],
    }
    strat=[]
    for name,f in STRATA.items():
        pool=[r for r in allisl if f(r)]
        take=min(120,len(pool))
        for r in rng.sample(pool,take):
            r=dict(r); r["_frame"]={"mode":"stratum","stratum":name,"N":len(pool),
                                    "n":take,"p":take/len(pool),
                                    "oversampling":(take/len(pool))/(k/N)}
            strat.append(r)

    census=Counter()
    for r in allisl:
        census["n"]+=1
        census[f"len_{min(r['len'],4)}"]+=1
        if r["ragged"]: census["ragged"]+=1
        if r["split_merge"]: census["split_merge"]+=1
        if r["any_strict_hom"]: census["homophone"]+=1
        if r["max_chardist"]<=2: census["chardist_le2"]+=1
        if r["vote_eq_oracle"]: census["vote_eq_oracle"]+=1
    (OUT/"page_islands.json").write_text(json.dumps(
        {"meta":sub["meta"],"census":dict(census),"trio":TRIO,
         "random":rand,"strata":strat}, ensure_ascii=False))
    print("islands", N, "| census", dict(census))
    print("random", len(rand), "strata", len(strat))
