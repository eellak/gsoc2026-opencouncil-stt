# -*- coding: utf-8 -*-
"""Regenerate THREE|COMPOSITE islands with wide context, for the compose experiment."""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "/home/harold/opencouncil-fine-tuning")
from concurrent.futures import ProcessPoolExecutor
from eval.controlled_eval.atlas_substrate import load
from eval.controlled_eval.column_classes import column_class, split_merge_columns
from eval.controlled_eval.msa import oracle_select, vote_column
from eval.controlled_eval.error_atlas import islands
import cats as C

WIDE = 30          # agreed tokens each side, up from 8

def job(w):
    cols=[tuple(c) for c in w["cols"]]; ref=w["ref"]
    osel=oracle_select(cols,ref); wsel=[vote_column(c,w["pivot"])[0] for c in cols]
    sm=split_merge_columns(cols); out=[]
    for (s,e) in islands(cols):
        span=list(range(s,e))
        spans=[tuple(x for x in (cols[i][k] for i in span) if x is not None) for k in range(3)]
        ragged=len({len(x) for x in spans})>1
        if not (ragged or any(i in sm for i in span)): continue
        if C.signature(spans)!="THREE": continue
        L=[cols[i][0] for i in range(max(0,s-60),s) if column_class(cols[i])=="agree"]
        R=[cols[i][0] for i in range(e,min(len(cols),e+60)) if column_class(cols[i])=="agree"]
        r12=[]
        for i in span:
            k=column_class(cols[i])
            if k=="unresolved_two": continue
            t=cols[i][0] if k=="unresolved_three" else wsel[i]
            if t is not None: r12.append(t)
        out.append({"win":w["id"],"city":w["city"],"st":s,"len":e-s,
                    "spans":[list(x) for x in spans],
                    "vote":[wsel[i] for i in span if wsel[i] is not None],
                    "r12":r12,
                    "oracle":[t for t in (osel[i] for i in span) if t is not None],
                    "L":" ".join(L[-WIDE:]),"R":" ".join(R[:WIDE])})
    return out

if __name__=="__main__":
    sub=load(Path(os.environ["REPORT"])); rows=[]
    with ProcessPoolExecutor(max_workers=14) as p:
        for r in p.map(job, sub["windows"], chunksize=4): rows+=r
    Path("three_composite_wide.json").write_text(json.dumps(rows,ensure_ascii=False))
    import statistics as st
    print(f"{len(rows)} νησιά THREE|COMPOSITE")
    print(f"context: αριστερά διάμεσος {st.median(len(r['L'].split()) for r in rows):.0f} λέξεις, "
          f"δεξιά {st.median(len(r['R'].split()) for r in rows):.0f}")
    # can a composition from available tokens even reach the answer?
    reach=0
    for r in rows:
        avail=set()
        for s in r['spans']: avail|=set(s)
        if set(r['oracle'])<=avail: reach+=1
    print(f"η σωστή απάντηση χρησιμοποιεί ΜΟΝΟ διαθέσιμες λέξεις: {reach}/{len(rows)} = {reach/len(rows):.1%}")
    exact=sum(1 for r in rows if tuple(r['oracle']) in [tuple(s) for s in r['spans']])
    print(f"η σωστή απάντηση είναι ΑΚΡΙΒΩΣ ενός συστήματος:        {exact}/{len(rows)} = {exact/len(rows):.1%}")
