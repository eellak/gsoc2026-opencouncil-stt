"""Score any hypothesis against the human-from-audio reference (48 windows).

The reference lives in `~/.cache/oc-public/reference_answers.json`; the split key
(`dev` vs `locked`) lives in `~/oc-reference-audit/KEY_DO_NOT_OPEN_UNTIL_DONE.json`.
Only the 32 `dev` windows are scored unless `--split locked` is passed explicitly,
which is the one thing this script will refuse to do quietly.

Why fitting alignment and not plain WER: a 20 s window cuts mid-utterance, and any
transcript built from whole utterances carries speech from outside the clip. Charging
that as insertions produced a WER of 0.52 and a 1.35x word ratio on the first attempt.
Leading and trailing hypothesis material is therefore free; everything inside the
matched span is scored normally.

Usage:
    score_audio_faithful.py HYP.json [HYP2.json ...] [--split dev] [--json OUT]

Each HYP.json is `{"rclip_001.wav": "text", ...}`. With two or more, the script also
reports the paired, meeting-clustered bootstrap interval on the WER difference
against the first file, which is the comparison the mixture-ratio arms need.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from eval.controlled_eval.scoring import wtoks

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
KEY = Path(os.environ.get("REF_KEY", Path.home()
                          / "oc-reference-audit/KEY_DO_NOT_OPEN_UNTIL_DONE.json"))
ANSWERS = SC / "reference_answers.json"

# The listener wrote `[?]` where the audio was not intelligible. It is a marker, not a
# word, and must not be scored as one on either side.
UNCLEAR = "[?]"


def fitting_align(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) with free hyp prefix and suffix.

    D[i][j] = cheapest edit of ref[:i] against any hyp[a:j]. Row 0 is all zeros, which
    is what makes the leading skip free; the answer is min over j of the last row,
    which makes the trailing skip free.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, 0
    # op codes: 0 match, 1 sub, 2 del (ref word unmatched), 3 ins (hyp word unmatched)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
        bp[i][0] = 2
    for i in range(1, n + 1):
        Di, Dp = D[i], D[i - 1]
        bpi = bp[i]
        ri = ref[i - 1]
        for j in range(1, m + 1):
            sub = Dp[j - 1] + (ri != hyp[j - 1])
            dele = Dp[j] + 1
            ins = Di[j - 1] + 1
            best = sub
            op = 0 if ri == hyp[j - 1] else 1
            if dele < best:
                best, op = dele, 2
            if ins < best:
                best, op = ins, 3
            Di[j] = best
            bpi[j] = op
    # Among equally cheap endings, consume as MUCH hypothesis as possible. The first
    # version took the earliest column, which turned every tied substitution into a
    # deletion: ref ["a"] against hyp ["b"] scored as one deletion, because ending at
    # column 0 ties at cost 1 and came first. Total WER is identical either way, but
    # the deletion rate is a decision guardrail here — it is the number that catches a
    # model learning to omit — so the biased split was not harmless.
    j_end = max(range(m + 1), key=lambda j: (-D[n][j], j))
    s = d = ins = 0
    i, j = n, j_end
    while i > 0:
        op = bp[i][j]
        if op == 0:
            i, j = i - 1, j - 1
        elif op == 1:
            s += 1
            i, j = i - 1, j - 1
        elif op == 2:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def ref_tokens(text: str) -> list[str]:
    return [t for t in wtoks(text.replace(UNCLEAR, " ")) if t]


def score(answers: dict, items: list[dict], hyp: dict) -> dict:
    """Per-window (errors, ref_words, s, d, i), keyed by clip, plus corpus totals."""
    per = {}
    for it in items:
        clip = it["clip"]
        ref = ref_tokens(answers.get(clip, ""))
        if not ref:
            continue
        s, d, ins = fitting_align(ref, wtoks(hyp.get(clip, "")))
        per[clip] = {"meeting": f"{it['city_id']}/{it['meeting_id']}",
                     "errors": s + d + ins, "ref_words": len(ref),
                     "sub": s, "del": d, "ins": ins}
    tot = lambda k: sum(v[k] for v in per.values())
    n = tot("ref_words")
    return {"per_clip": per, "n_clips": len(per), "ref_words": n,
            "wer": tot("errors") / n if n else float("nan"),
            "del_rate": tot("del") / n if n else float("nan"),
            "sub_rate": tot("sub") / n if n else float("nan"),
            "ins_rate": tot("ins") / n if n else float("nan")}


def paired_bootstrap(a: dict, b: dict, n_boot=10000, seed=7, lo_q=0.05, hi_q=0.95):
    """CI on WER(a) - WER(b), resampling meetings. Windows in one meeting share a
    speaker and a room, so the meeting is the independent unit, not the window."""
    clips = sorted(set(a["per_clip"]) & set(b["per_clip"]))
    by_mtg: dict[str, list[str]] = {}
    for c in clips:
        by_mtg.setdefault(a["per_clip"][c]["meeting"], []).append(c)
    keys = sorted(by_mtg)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        ea = na = eb = nb = 0
        for k in pick:
            for c in by_mtg[k]:
                pa, pb = a["per_clip"][c], b["per_clip"][c]
                ea += pa["errors"]; na += pa["ref_words"]
                eb += pb["errors"]; nb += pb["ref_words"]
        if na and nb:
            diffs.append(ea / na - eb / nb)
    diffs.sort()
    q = lambda p: diffs[min(len(diffs) - 1, int(p * len(diffs)))]
    return {"delta": a["wer"] - b["wer"], "lo": q(lo_q), "hi": q(hi_q),
            "n_meetings": len(keys), "n_clips": len(clips)}


def paired_seed_contrast(pairs: list[tuple[dict, dict]], n_boot=10000, seed=7,
                         lo_q=0.05, hi_q=0.95):
    """The preregistered confirmatory estimator: mean of the per-seed C − A gaps.

    Each bootstrap replicate resamples meetings ONCE and applies that same resample to
    every seed pair, then averages the three differences. Resampling independently per
    pair would treat the seeds as independent evaluations and understate the shared
    evaluation-set uncertainty; comparing only one pair would ignore training variance,
    which is the thing three seeds were bought to measure."""
    common = sorted(set.intersection(*[set(a["per_clip"]) & set(b["per_clip"])
                                       for a, b in pairs]))
    by_mtg: dict[str, list[str]] = {}
    for c in common:
        by_mtg.setdefault(pairs[0][0]["per_clip"][c]["meeting"], []).append(c)
    keys = sorted(by_mtg)

    def gap(a, b, clips):
        ea = na = eb = nb = 0
        for c in clips:
            pa, pb = a["per_clip"][c], b["per_clip"][c]
            ea += pa["errors"]; na += pa["ref_words"]
            eb += pb["errors"]; nb += pb["ref_words"]
        return (ea / na - eb / nb) if na and nb else None

    point = [gap(t, ctl, common) for ctl, t in pairs]
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        clips = [c for k in pick for c in by_mtg[k]]
        gs = [gap(t, ctl, clips) for ctl, t in pairs]
        if all(g is not None for g in gs):
            draws.append(sum(gs) / len(gs))
    draws.sort()
    q = lambda p: draws[min(len(draws) - 1, int(p * len(draws)))]
    return {"per_seed_delta": [round(p, 5) for p in point],
            "mean_delta": round(sum(point) / len(point), 5),
            "lo": round(q(lo_q), 5), "hi": round(q(hi_q), 5),
            "n_meetings": len(keys), "n_clips": len(common), "n_pairs": len(pairs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hyp", nargs="*", help="JSON files mapping clip -> text")
    ap.add_argument("--split", default="dev", choices=["dev", "locked", "all"])
    ap.add_argument("--json", default="", help="write the full result here")
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="CONTROL:TREATMENT tags, e.g. A_s13:C_s13. Runs the "
                         "preregistered paired-seed contrast instead of comparing "
                         "everything against the first file.")
    ap.add_argument("--dir", default=".", help="where the hypothesis JSONs live")
    args = ap.parse_args()
    if not args.hyp and not args.pairs:
        ap.error("give hypothesis files, or --pairs")

    if args.split != "dev":
        print(f"!! scoring the {args.split} split — the locked windows are the final "
              f"test and every use of them spends them")

    answers = json.loads(ANSWERS.read_text())
    key = json.loads(KEY.read_text())
    items = [it for it in key if args.split == "all" or it["split"] == args.split]

    if args.pairs:
        d = Path(args.dir)
        loaded: dict[str, dict] = {}
        pairs = []
        for spec in args.pairs:
            ctl, trt = spec.split(":", 1)
            for tag in (ctl, trt):
                if tag not in loaded:
                    loaded[tag] = score(answers, items,
                                        json.loads((d / f"{tag}.json").read_text()))
                    r = loaded[tag]
                    print(f"{tag:12s} WER {r['wer']:.4f}  del {r['del_rate']:.4f}  "
                          f"sub {r['sub_rate']:.4f}  ins {r['ins_rate']:.4f}")
            pairs.append((loaded[ctl], loaded[trt]))
        res = paired_seed_contrast(pairs)
        print(f"\npaired {len(pairs)}-seed contrast (treatment − control), "
              f"{res['n_meetings']} meetings, {res['n_clips']} clips")
        print(f"  per seed : {res['per_seed_delta']}")
        print(f"  mean     : {res['mean_delta']:+.4f}  "
              f"90% [{res['lo']:+.4f}, {res['hi']:+.4f}]")
        # The guardrail is read on the interval, not the point estimate.
        dels = paired_seed_contrast(
            [({"per_clip": {k: {**v, "errors": v["del"]} for k, v in a["per_clip"].items()}},
              {"per_clip": {k: {**v, "errors": v["del"]} for k, v in b["per_clip"].items()}})
             for a, b in pairs])
        print(f"  deletions: {dels['mean_delta']:+.4f}  "
              f"90% [{dels['lo']:+.4f}, {dels['hi']:+.4f}]  "
              f"(gate: upper bound must stay below +0.005)")
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"split": args.split, "arms": {k: {x: v[x] for x in v if x != "per_clip"}
                                               for k, v in loaded.items()},
                 "wer_contrast": res, "deletion_contrast": dels},
                ensure_ascii=False, indent=1))
            print(f"\nwrote {args.json}")
        return

    results = {}
    for p in args.hyp:
        name = Path(p).stem
        results[name] = score(answers, items, json.loads(Path(p).read_text()))
        r = results[name]
        print(f"{name:28s} WER {r['wer']:.4f}  (sub {r['sub_rate']:.4f} "
              f"del {r['del_rate']:.4f} ins {r['ins_rate']:.4f})  "
              f"{r['n_clips']} clips, {r['ref_words']} words")

    names = list(results)
    comparisons = {}
    if len(names) > 1:
        base = names[0]
        print(f"\npaired, clustered by meeting, 90% interval, vs {base}:")
        for n in names[1:]:
            cmp_ = paired_bootstrap(results[n], results[base])
            comparisons[f"{n}_vs_{base}"] = cmp_
            print(f"  {n:26s} delta {cmp_['delta']:+.4f}  "
                  f"[{cmp_['lo']:+.4f}, {cmp_['hi']:+.4f}]  "
                  f"({cmp_['n_meetings']} meetings)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"split": args.split, "results": results, "comparisons": comparisons},
            ensure_ascii=False, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
