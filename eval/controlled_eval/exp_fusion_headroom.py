#!/usr/bin/env python3
"""Is there anything to gain by combining ASR systems, before we pay an LLM to try?

The project's own model loses to base whisper on the benchmark, and the LLM post-editor
turned out to be a coin flip once its over-editing tax was measured. Both of those are
single-system results. This asks a different question: do the systems fail on the SAME
windows, or on different ones? If they fail together there is nothing to combine and the
fusion idea dies here, for free.

Three quantities, in increasing order of how much they promise:

  best single        the provider to beat.
  ORACLE selection   per window, take the hypothesis with the lowest WER, using the
                     reference. Not a system. It is the ceiling for any method that
                     picks whole windows, and if it sits at the best single, stop.
  CONSENSUS          per window, keep the hypothesis the others agree with most. No
                     reference, no LLM, no training. This is what a deployable selector
                     looks like, and it is the control any LLM fusion has to beat before
                     its gain can be attributed to the LLM.

Everything is scored with eval/controlled_eval/scoring.py over the benchmark's
human-corrected references, so the numbers here are directly comparable with the LLM
arms in exp_fusion.py and NOT identical to the benchmark app's published leaderboard
(which trims window-boundary words by cross-provider consensus).

Writes results_fusion_headroom.json (tracked; aggregates only, no transcript text).

Env: SC (cache dir for the report) N_BOOT (10000) SUBSET_MAX (3)
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.controlled_eval import bench_data as B                      # noqa: E402
from eval.controlled_eval.scoring import (cluster_bootstrap, edist,    # noqa: E402
                                          head2head, wer, wtoks)

OUT = Path(__file__).with_name("results_fusion_headroom.json")
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
SUBSET_MAX = int(os.environ.get("SUBSET_MAX", "3"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_counts(items, providers):
    """(errors, ref_words) for every provider on every window, computed once.

    Every arm below is a *selection* among these same hypotheses, so the whole
    experiment needs exactly len(items) x len(providers) edit distances.
    """
    out = []
    for it in items:
        ref = wtoks(it["ref"])
        out.append({p: (edist(ref, wtoks(it["hyp"][p])), len(ref)) for p in providers})
    return out


def arm_counts(counts, pick):
    """counts for a selector: pick(i) -> provider id."""
    return [counts[i][pick(i)] for i in range(len(counts))]


def summarize(items, counts, providers):
    """Single systems, oracle ceilings and the consensus selector, on one item set."""
    single = {p: wer([c[p] for c in counts]) for p in providers}
    best_p = min(single, key=single.get)

    def oracle(combo):
        return arm_counts(counts, lambda i: min(combo, key=lambda p: counts[i][p][0]))

    subsets = {}
    for k in range(2, min(SUBSET_MAX, len(providers)) + 1):
        for combo in itertools.combinations(providers, k):
            o = wer(oracle(combo))
            b = min(single[p] for p in combo)
            subsets["+".join(combo)] = {"oracle": o, "best_single": b, "headroom": b - o}
    allc = wer(oracle(tuple(providers)))
    return {"single": single, "best_provider": best_p, "best_single": single[best_p],
            "oracle_all": allc, "headroom_all": single[best_p] - allc,
            "subsets": subsets}


def consensus_arm(items, counts, combo):
    return arm_counts(counts, lambda i: B.consensus_pick(items[i], combo))


def evaluate(items, counts, combo, baseline_provider, label):
    """Consensus over `combo` against the single best provider, with a clustered CI."""
    a = consensus_arm(items, counts, combo)
    b = [c[baseline_provider] for c in counts]
    o = arm_counts(counts, lambda i: min(combo, key=lambda p: counts[i][p][0]))
    clusters = [it["cluster"] for it in items]
    ci = cluster_bootstrap(a, b, clusters, n_boot=N_BOOT)
    captured = ((wer(b) - wer(a)) / (wer(b) - wer(o))) if wer(b) > wer(o) else None
    res = {"combo": list(combo), "baseline": baseline_provider,
           "wer_consensus": wer(a), "wer_baseline": wer(b), "wer_oracle": wer(o),
           "headroom_captured": captured, "ci_vs_baseline": ci,
           "head2head": head2head(a, b), "n_items": len(items),
           "n_clusters": len(set(clusters))}
    log(f"{label}: baseline {wer(b):.4f} -> consensus {wer(a):.4f} "
        f"(oracle {wer(o):.4f}) | delta {ci['delta']:+.4f} "
        f"CI[{ci['ci95'][0]:+.4f},{ci['ci95'][1]:+.4f}] | "
        f"captured {captured:.0%}" if captured else label)
    return res


def split_half(items, counts, providers, baseline):
    """Pick the provider subset on one half of the meetings, score it on the other.

    Choosing the best-performing subset and then quoting its score is selection on the
    test set: with 21 pairs and 35 triples on 147 meetings, some subset wins by luck.
    Splitting by meeting (never by window, since windows from one meeting share a room
    and a speaker) turns the choice into a prediction that can fail.
    """
    meetings = sorted({it["cluster"] for it in items})
    half = set(meetings[::2])
    idx_a = [i for i, it in enumerate(items) if it["cluster"] in half]
    idx_b = [i for i, it in enumerate(items) if it["cluster"] not in half]
    out = {}
    for name, fit, test in (("A_picks_B", idx_a, idx_b), ("B_picks_A", idx_b, idx_a)):
        cand = [c for k in range(3, min(SUBSET_MAX, len(providers)) + 1)
                for c in itertools.combinations(providers, k)]
        if not cand:
            continue
        def score(idx, combo):
            return wer([counts[i][B.consensus_pick(items[i], combo)] for i in idx])
        best = min(cand, key=lambda c: score(fit, c))
        out[name] = {"chosen_on_fit_half": list(best),
                     "wer_on_fit": score(fit, best),
                     "wer_on_test": score(test, best),
                     "baseline_on_test": wer([counts[i][baseline] for i in test]),
                     "n_test_items": len(test)}
        log(f"split-half {name}: chose {'+'.join(best)} | test {out[name]['wer_on_test']:.4f} "
            f"vs baseline {out[name]['baseline_on_test']:.4f}")
    return out


def main():
    report = B.load_report()
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    log(f"{len(items)} windows with all {len(providers)} providers, "
        f"{len({it['cluster'] for it in items})} meetings")

    log("scoring every provider on every window (this is the only expensive part)")
    counts = build_counts(items, providers)

    results = {"run_id": B.RUN_ID, "providers": providers, "n_items": len(items),
               "scorer": "eval/controlled_eval/scoring.py (not the benchmark app's)",
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    results["all"] = summarize(items, counts, providers)
    base = results["all"]["best_provider"]
    log(f"best single provider: {base} at {results['all']['best_single']:.4f} | "
        f"oracle over all {len(providers)}: {results['all']['oracle_all']:.4f}")

    # Ranked headroom, so the report can show which pairings are worth anything.
    ranked = sorted(results["all"]["subsets"].items(), key=lambda kv: kv[1]["oracle"])
    log("top oracle subsets: " + ", ".join(
        f"{k} {v['oracle']:.4f}" for k, v in ranked[:5]))

    # The trio the exploratory pass picked, plus a no-own-model control: if the gain
    # only exists when our own (contaminated) model votes, that is worth knowing.
    trios = {
        "scribe+soniox+ours": ("scribe-v2-clean", "soniox", "oc-minipc-finetune"),
        "scribe+soniox+gladia": ("scribe-v2-clean", "soniox", "gladia-prod"),
        "scribe+soniox+whisper": ("scribe-v2-clean", "soniox",
                                  "hf-openai-whisper-large-v3"),
    }
    results["consensus"] = {}
    for name, combo in trios.items():
        if any(p not in providers for p in combo):
            continue
        results["consensus"][name] = evaluate(items, counts, combo, base, name)

    # Contamination split. 105 of 203 benchmark meetings are in the fine-tune's
    # training data, so any arm our own model votes in has to survive being restricted
    # to the meetings it has never seen.
    results["by_contamination"] = {}
    for label, keep in (("clean", False), ("in_training", True)):
        idx = [i for i, it in enumerate(items) if it["in_training"] is keep]
        if len(idx) < 20:
            continue
        sub_items = [items[i] for i in idx]
        sub_counts = [counts[i] for i in idx]
        entry = {"n_items": len(idx),
                 "single": {p: wer([c[p] for c in sub_counts]) for p in providers}}
        for name, combo in trios.items():
            if all(p in providers for p in combo):
                entry[name] = evaluate(sub_items, sub_counts, combo, base,
                                       f"  [{label}] {name}")
        results["by_contamination"][label] = entry

    results["split_half"] = split_half(items, counts, providers, base)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
