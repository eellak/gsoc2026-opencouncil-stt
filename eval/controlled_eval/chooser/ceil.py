import os,sys
sys.path.insert(0,'/home/harold/opencouncil-fine-tuning')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from eval.controlled_eval.atlas_substrate import load
from eval.controlled_eval.error_atlas import lattice_best
from eval.controlled_eval.exp_fusion_deletions import sdi

SETS=(("trio",(0,1,2)),("drop_ours",(0,1)),("drop_soniox",(0,2)),
      ("drop_scribe",(1,2)),("scribe_only",(0,)),("soniox_only",(1,)),("ours_only",(2,)))

def job(w):
    cols=[tuple(c) for c in w["cols"]]; ref=w["ref"]; R=" ".join(ref)
    out={}
    for name,keep in SETS:
        cands=[]
        for c in cols:
            seen,cl=set(),[]
            if any(c[k] is None for k in keep): cl.append(None)
            for k in keep:
                if c[k] is not None and c[k] not in seen: seen.add(c[k]); cl.append(c[k])
            cands.append(cl or [None])
        toks=[t for t in lattice_best(cands,ref) if t is not None]
        out[name]=sdi(R," ".join(toks))
    return out

if __name__=="__main__":
    sub=load(Path(os.environ["REPORT"]))
    tot={}
    with ProcessPoolExecutor(max_workers=14) as ex:
        for r in ex.map(job, sub["windows"], chunksize=4):
            for k,v in r.items():
                t=tot.setdefault(k,[0,0,0,0])
                for i in range(4): t[i]+=v[i]
    print("Ceiling inside the SAME fixed alignment, candidate set restricted:")
    base=(lambda t:(t[0]+t[1]+t[2])/t[3])(tot["trio"])
    for k,_ in SETS:
        S,D,I,R=tot[k]; wer=(S+D+I)/R
        print(f"  {k:14s} ceiling {wer:.4f}   cost of removing the rest {wer-base:+.4f}")
