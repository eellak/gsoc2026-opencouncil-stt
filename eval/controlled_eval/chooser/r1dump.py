"""Every column R1 would delete, with context, so a human can judge it."""
import os,sys,json,random
sys.path.insert(0,'/home/harold/opencouncil-fine-tuning')
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from eval.controlled_eval.atlas_substrate import load, TRIO
from eval.controlled_eval.column_classes import column_class
from eval.controlled_eval.msa import oracle_select, vote_column

def job(w):
    cols=[tuple(c) for c in w["cols"]]; ref=w["ref"]
    osel=oracle_select(cols,ref); wsel=[vote_column(c,w["pivot"])[0] for c in cols]
    out=[]
    for i,c in enumerate(cols):
        if column_class(c)!="unresolved_two": continue
        L=[wsel[j] for j in range(max(0,i-8),i) if wsel[j] is not None]
        R=[wsel[j] for j in range(i+1,min(len(cols),i+9)) if wsel[j] is not None]
        out.append({"w":w["id"],"c":w["city"],
                    "s":[c[0],c[1],c[2]],
                    "v":wsel[i],"o":osel[i],
                    "agree":int(osel[i] is None),   # did the oracle also delete?
                    "L":" ".join(L),"R":" ".join(R)})
    return out

if __name__=="__main__":
    sub=load(Path(os.environ["REPORT"])); rows=[]
    with ProcessPoolExecutor(max_workers=14) as p:
        for r in p.map(job, sub["windows"], chunksize=4): rows+=r
    rng=random.Random(20260825); rng.shuffle(rows)
    kept=[r for r in rows if r["v"] is not None]     # R1 changes only these
    print(f"unresolved_two columns: {len(rows)}")
    print(f"  the vote currently WRITES a word: {len(kept)}  <- R1 deletes exactly these")
    print(f"  oracle also deletes: {sum(r['agree'] for r in kept)} = {sum(r['agree'] for r in kept)/len(kept):.1%}")
    c=Counter()
    FILL={"εε","εεε","ε","μμ","μμμ","αα","ααα","ε.","εε."}
    for r in kept:
        t=r["v"]
        c["filler" if t in FILL else ("dup" if t in (r["L"].split()[-2:]+r["R"].split()[:2]) else "word")]+=1
    print("  what the vote writes there:", dict(c))
    Path(os.environ["OUT"]).write_text(json.dumps(
        {"n_all":len(rows),"n_deleted":len(kept),
         "oracle_agrees":sum(r['agree'] for r in kept),
         "rows":kept}, ensure_ascii=False, separators=(",",":")))
    print("wrote", os.environ["OUT"])
