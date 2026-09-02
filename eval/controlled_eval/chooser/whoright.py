import os,sys,json,random
sys.path.insert(0,'/home/harold/opencouncil-fine-tuning')
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from eval.controlled_eval.atlas_substrate import load, TRIO
from eval.controlled_eval.column_classes import column_class, column_flags, split_merge_columns
from eval.controlled_eval.msa import oracle_select, vote_column
from eval.controlled_eval.greek_phonetics import phon
from eval.controlled_eval.scoring import edist

NON_AGREE=("exact_2_of_3","two_present_same","singleton","unresolved_two","unresolved_three")

def job(w):
    cols=[tuple(c) for c in w["cols"]]; ref=w["ref"]
    osel=oracle_select(cols,ref)
    wsel=[vote_column(c,w["pivot"])[0] for c in cols]
    sm=split_merge_columns(cols)
    cnt=Counter(); ex=[]
    for i,c in enumerate(cols):
        k=column_class(c)
        if k not in NON_AGREE: continue
        o=osel[i]
        cnt[f"{k}|n"]+=1
        if o is None: cnt[f"{k}|oracle_eps"]+=1
        for s in range(3):
            if c[s]==o: cnt[f"{k}|right:{TRIO[s][:6]}"]+=1
        if wsel[i]==o: cnt[f"{k}|W_right"]+=1
        else:
            cnt[f"{k}|W_wrong"]+=1
            for s in range(3):
                if c[s]==o and o is not None: cnt[f"{k}|W_wrong_but_right:{TRIO[s][:6]}"]+=1
        # ---- example dump, only where W is wrong (that is the recoverable set)
        if wsel[i]!=o:
            f=column_flags(c)
            lo=max(0,i-6); hi=min(len(cols),i+7)
            ctx=[t for t in (wsel[j] for j in range(lo,i)) if t is not None]
            ctx2=[t for t in (wsel[j] for j in range(i+1,hi)) if t is not None]
            ex.append({"win":w["id"],"city":w["city"],"col":i,"klass":k,
                       "scribe":c[0],"soniox":c[1],"ours":c[2],
                       "W":wsel[i],"oracle":o,
                       "strict_hom":f["strict_homophone"],"loose_hom":f["loose_homophone"],
                       "partial_hom":f["partial_homophone"],"chardist":f["max_char_dist"],
                       "split_merge":i in sm,
                       "left":" ".join(ctx),"right":" ".join(ctx2)})
    return cnt, ex

if __name__=="__main__":
    sub=load(Path(os.environ["REPORT"]))
    tot=Counter(); allex=[]
    with ProcessPoolExecutor(max_workers=14) as p:
        for c,e in p.map(job, sub["windows"], chunksize=4):
            tot.update(c); allex+=e
    print(f"{'class':18s} {'n':>7s} {'oracle=eps':>10s} {'scribe':>8s} {'soniox':>8s} {'ours':>8s} {'W right':>8s}")
    for k in NON_AGREE:
        n=tot[f"{k}|n"]
        if not n: continue
        print(f"{k:18s} {n:7d} {tot[f'{k}|oracle_eps']/n:9.1%} "
              f"{tot[f'{k}|right:scribe']/n:7.1%} {tot[f'{k}|right:soniox']/n:7.1%} "
              f"{tot[f'{k}|right:oc-cle']/n:7.1%} {tot[f'{k}|W_right']/n:7.1%}")
    print(f"\nWhere the VOTE is wrong, who had the oracle token (share of W-wrong columns):")
    print(f"{'class':18s} {'W wrong':>8s} {'scribe':>8s} {'soniox':>8s} {'ours':>8s} {'nobody':>8s}")
    for k in NON_AGREE:
        n=tot[f"{k}|W_wrong"]
        if not n: continue
        a,b,c_=(tot[f'{k}|W_wrong_but_right:scribe'],tot[f'{k}|W_wrong_but_right:soniox'],
                tot[f'{k}|W_wrong_but_right:oc-cle'])
        nobody=n-len([x for x in ()])  # computed below properly
        print(f"{k:18s} {n:8d} {a/n:7.1%} {b/n:7.1%} {c_/n:7.1%}")
    out=Path(os.environ["OUTDIR"])/"disagreements.jsonl"
    random.Random(20260825).shuffle(allex)
    out.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in allex))
    print(f"\nwrote {len(allex)} recoverable-disagreement examples -> {out}")
