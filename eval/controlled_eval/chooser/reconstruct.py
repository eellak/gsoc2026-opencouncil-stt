# -*- coding: utf-8 -*-
"""Rebuild every confirmation window under the frozen policy and score it as WER.

Island accuracy is not WER. An island is not a word, and replacing one column
realigns everything after it, so +6.8 points of island accuracy does not convert
into any particular WER number. This is the measurement that does.

WHY THE LLM RUNS AGAIN. `confirm.py` stored counts, not per-island decisions - my
omission. The policy is unchanged and hashed; nothing about it can move based on
what the first pass showed. What this pass costs is a fresh sample of the same
frozen policy, which is honest, and a second set of calls, which is waste caused by
the earlier omission. It is recorded as a reconstruction pass, never as a second
confirmation of a new hypothesis.

Env: FREEZE, WORKERS
"""
import json, os, re, sys, random
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/harold/opencouncil-fine-tuning")
from eval.controlled_eval.atlas_substrate import load, TRIO      # noqa
from eval.controlled_eval.column_classes import (column_class,   # noqa
                                                 split_merge_columns)
from eval.controlled_eval.msa import vote_column                 # noqa
from eval.controlled_eval.error_atlas import islands             # noqa
from eval.controlled_eval.exp_fusion_deletions import sdi        # noqa
import run_chooser as RC
import autoprompt as A
import cats as C

N_BOOT = int(os.environ.get("N_BOOT", "10000"))


def classify(cols, span, sm, ctx):
    spans = [tuple(x for x in (cols[i][k] for i in span) if x is not None)
             for k in range(3)]
    ragged = len({len(x) for x in spans}) > 1
    composite = ragged or any(i in sm for i in span)
    if len({C.norm_num(x) for x in spans}) == 1:
        return "NUM", spans
    if len({C.strip_fill(x) for x in spans}) == 1:
        return "FILL", spans
    if len({C.strip_dup(x, ctx) for x in spans}) == 1:
        return "DUP", spans
    if len({C.strip_dup(C.strip_fill(x), ctx) for x in spans}) == 1:
        return "LOW", spans
    sig = C.signature(spans)
    if sig.startswith(("TWO_", "SOLO_", "PAIRE_")):
        return sig, spans
    return sig + ("|COMPOSITE" if composite else "|ATOMIC"), spans


def window_parts(w):
    """Returns (agree_prefixes, islands) enough to rebuild any policy's text."""
    cols = [tuple(c) for c in w["cols"]]
    wsel = [vote_column(c, w["pivot"])[0] for c in cols]
    sm = split_merge_columns(cols)
    isl, out, prev = islands(cols), [], 0
    for (s, e) in isl:
        span = list(range(s, e))
        L = [cols[i][0] for i in range(max(0, s - 12), s)
             if column_class(cols[i]) == "agree"]
        R = [cols[i][0] for i in range(e, min(len(cols), e + 12))
             if column_class(cols[i]) == "agree"]
        cat, spans = classify(cols, span, sm, set(L[-3:]) | set(R[:3]))
        vote = [wsel[i] for i in span if wsel[i] is not None]
        r12 = []
        for i in span:
            k = column_class(cols[i])
            if k == "unresolved_two":
                continue
            t = cols[i][0] if k == "unresolved_three" else wsel[i]
            if t is not None:
                r12.append(t)
        out.append({"s": s, "e": e, "cat": cat, "spans": [list(x) for x in spans],
                    "vote": vote, "r12": r12,
                    "L": " ".join(L[-8:]), "R": " ".join(R[:8]),
                    "before": [cols[i][0] for i in range(prev, s)
                               if column_class(cols[i]) == "agree"]})
        prev = e
    tail = [cols[i][0] for i in range(prev, len(cols))
            if column_class(cols[i]) == "agree"]
    return out, tail, wsel, cols


RULES = {"ψήφος": lambda i: i["vote"], "R1+R2": lambda i: i["r12"],
         "Scribe": lambda i: i["spans"][0], "σβήσε": lambda i: []}


def main():
    fr = json.loads(Path(os.environ["FREEZE"]).read_text())
    pol = fr["policy"]
    sub = load(Path(os.environ["REPORT"]))
    WHICH = os.environ.get("CITIES", "CONFIRM")
    if WHICH == "ALL":
        wins = list(sub["windows"])
    elif WHICH == "SEARCH":
        wins = [w for w in sub["windows"] if w["city"] in A.SEARCH_CITIES]
    else:
        wins = [w for w in sub["windows"] if w["city"] in A.CONFIRM_CITIES]
    print(f"{len(wins)} παράθυρα ({WHICH})", flush=True)

    parsed = {w["id"]: window_parts(w) for w in wins}

    # --- collect the LLM islands and decide them in batches
    need = []
    # SKIP_LLM=1 verifies the deterministic arms (W, R1+R2, singles) without
    # spending LLM calls. The `policy` arm is then NOT reproduced: its two llm
    # categories fall back to r12 and its number must be ignored.
    skip_llm = os.environ.get("SKIP_LLM") == "1"
    for w in ([] if skip_llm else wins):
        for k, i in enumerate(parsed[w["id"]][0]):
            if pol.get(i["cat"], {}).get("mode") == "llm":
                need.append((w["id"], k, i))
    print(f"{len(need)} νησιά πάνε στο LLM", flush=True)
    decided = {}
    bycat = defaultdict(list)
    for wid, k, i in need:
        bycat[i["cat"]].append((wid, k, i))
    for cat, items in bycat.items():
        instr = pol[cat]["instr"]
        B = fr.get("batch", 15)
        for s in range(0, len(items), B):
            chunk = items[s:s + B]
            rows = [{"L": i["L"], "R": i["R"], "spans": i["spans"], "r12": i["r12"]}
                    for _, _, i in chunk]
            txt, err = RC.call(RC.render(rows, instr), fr["model"], fr.get("think", 4000))
            if err:
                print(f"  ERROR {err[:80]}", flush=True)
                continue
            for j, v in RC.parse(txt, len(chunk)).items():
                wid, k, i = chunk[j]
                ps = RC.paths_of({"spans": i["spans"], "r12": i["r12"]})
                m = re.fullmatch(r"P(\d+)", v[0])
                decided[(wid, k)] = (list(ps[int(m.group(1))][0])
                                     if m and int(m.group(1)) < len(ps) else i["r12"])
        print(f"  {cat}: {len(items)} νησιά", flush=True)

    # --- rebuild each window under every arm
    def build(w, arm):
        isl, tail, wsel, cols = parsed[w["id"]]
        out = []
        for k, i in enumerate(isl):
            out += i["before"]
            if arm == "policy":
                p = pol.get(i["cat"])
                if p and p["mode"] == "llm":
                    out += decided.get((w["id"], k), i["r12"])
                elif p:
                    out += RULES[p["rule"]](i)
                else:
                    out += i["vote"]
            elif arm == "W":
                out += i["vote"]
            elif arm == "R1+R2":
                out += i["r12"]
        return out + tail

    per = defaultdict(dict)
    for w in wins:
        R = " ".join(w["ref"])
        for arm in ("policy", "W", "R1+R2"):
            per[arm][w["id"]] = sdi(R, " ".join(build(w, arm)))
        for k, p in enumerate(TRIO):
            per[f"single:{p}"][w["id"]] = sdi(R, " ".join(w["hyps"][k]))

    def tot(a, ids):
        t = [0, 0, 0, 0]
        for i in ids:
            for j in range(4):
                t[j] += per[a][i][j]
        return t

    def wer(t):
        return (t[0] + t[1] + t[2]) / t[3]

    ids = [w["id"] for w in wins]
    SRC = {w["id"]: ("SEARCH" if w["city"] in A.SEARCH_CITIES else "CONFIRM")
           for w in wins}
    print(f"\n{'arm':26s} {'WER':>8s} {'sub':>8s} {'del':>8s} {'ins':>8s}")
    for a in ("W", "R1+R2", "policy") + tuple(f"single:{p}" for p in TRIO):
        t = tot(a, ids)
        print(f"{a:26s} {wer(t):8.5f} {t[0]/t[3]:8.5f} {t[1]/t[3]:8.5f} {t[2]/t[3]:8.5f}")

    # the split matters: prompts were chosen on SEARCH, so a pooled number over all
    # 391 windows is 57% contaminated and is NOT a held-out result.
    if len({SRC[i] for i in ids}) > 1:
        print()
        for part in ("SEARCH", "CONFIRM"):
            sel = [i for i in ids if SRC[i] == part]
            print(f"  {part:8s} n={len(sel):4d}  policy {wer(tot('policy', sel)):.5f}  "
                  f"R1+R2 {wer(tot('R1+R2', sel)):.5f}  W {wer(tot('W', sel)):.5f}")
    by = defaultdict(list)
    for w in wins:
        by[(w["city"], w["meeting"])].append(w["id"])
    cl = sorted(by)
    rng = random.Random(20260825)
    print()
    for a, b in (("policy", "W"), ("policy", "R1+R2")):
        def d(sample):
            return wer(tot(a, sample)) - wer(tot(b, sample))
        pt = d(ids)
        bs = sorted(d([i for c in [rng.choice(cl) for _ in cl] for i in by[c]])
                    for _ in range(N_BOOT))
        dom = {}
        for c in cl:
            dom[c] = sum(sum(per[a][i][:3]) - sum(per[b][i][:3]) for i in by[c])
        T = sum(abs(v) for v in dom.values()) or 1
        print(f"{a} vs {b}: Δ {pt:+.5f}  CI [{bs[int(.025*N_BOOT)]:+.5f}, "
              f"{bs[int(.975*N_BOOT)]:+.5f}]  μεγαλύτερο cluster "
              f"{max(abs(v) for v in dom.values())/T:.1%} από {len(cl)}")
    Path(f"wer_result_{os.environ.get('CITIES','CONFIRM')}.json").write_text(json.dumps(
        {"arms": {a: tot(a, ids) for a in per}, "n_windows": len(wins),
         "n_clusters": len(cl), "freeze": fr.get("declared")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
