"""Read a benchmark report.json honestly.

The benchmark app publishes a pooled score per provider. That number alone has
already misled this project twice: once by hiding that a single window carried a
headline delta, once by treating a provider that omits hard passages as better.
This script prints the pooled score the way it must be computed -- sum of edits
over sum of reference words -- next to the things that decide whether it means
anything: the deletion rate with its denominator, the training-disjoint strata
separately, and a leave-one-window-out check on every pairwise delta.

The report contains reference and hypothesis text, so its input path must stay
under ~/.cache/oc-public/. Nothing printed here contains transcript text.

    python3 eval/bench_report_analysis.py <report.json> <provider> [baseline ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

# The only cities in the benchmark with no meeting in the training set. Every
# other stratum is contaminated and must not be read as model quality.
TRAINING_DISJOINT = ("city:argos", "city:orestiada")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def provider_row(report: dict, instance_id: str) -> dict:
    for p in report["providers"]:
        if p["instanceId"] == instance_id:
            return p
    raise SystemExit(f"provider {instance_id} not in report")


def per_item(report: dict, instance_id: str, metric: str) -> dict:
    """itemId -> (numerator, denominator, sub, del, ins), scored items only."""
    out = {}
    for item in report["items"]:
        pp = item["perProvider"].get(instance_id)
        if not pp or pp.get("status") != "ok":
            continue
        m = pp.get("perMetric", {}).get(metric)
        if not m:
            continue
        d = m.get("details") or {}
        out[item["itemId"]] = (
            m["numerator"], m["denominator"],
            d.get("sub", 0), d.get("del", 0), d.get("ins", 0),
        )
    return out


def pooled(rows: dict) -> tuple[float, int, int]:
    num = sum(v[0] for v in rows.values())
    den = sum(v[1] for v in rows.values())
    return (num / den if den else float("nan")), num, den


def rates(rows: dict) -> dict:
    den = sum(v[1] for v in rows.values()) or 1
    return {
        "sub": sum(v[2] for v in rows.values()) / den,
        "del": sum(v[3] for v in rows.values()) / den,
        "ins": sum(v[4] for v in rows.values()) / den,
        "ref_words": den,
    }


def strata(report: dict, instance_id: str, metric: str) -> dict:
    row = provider_row(report, instance_id)
    return {k: v for k, v in row["metrics"][metric].items() if k.startswith("city:")}


def paired_delta(a: dict, b: dict) -> dict:
    """a - b, pooled over the items BOTH scored, plus domination checks."""
    common = sorted(set(a) & set(b))
    if not common:
        return {"items": 0}
    an = sum(a[i][0] for i in common)
    bn = sum(b[i][0] for i in common)
    den = sum(a[i][1] for i in common)
    assert den == sum(b[i][1] for i in common), "reference words differ across providers"
    delta = (an - bn) / den

    # Which single window supplies the largest share of the net edit difference,
    # and does removing any one window reverse the sign?
    contrib = {i: (a[i][0] - b[i][0]) for i in common}
    net = an - bn
    largest = max(contrib.items(), key=lambda kv: abs(kv[1])) if contrib else (None, 0)
    share = abs(largest[1]) / abs(net) if net else float("inf")
    reversals = 0
    for i in common:
        loo_num = net - contrib[i]
        loo_den = den - a[i][1]
        if loo_den and (loo_num / loo_den) * delta < 0:
            reversals += 1

    better = sum(1 for i in common if contrib[i] < 0)
    tied = sum(1 for i in common if contrib[i] == 0)
    worse = sum(1 for i in common if contrib[i] > 0)
    return {
        "items": len(common), "delta": delta, "ref_words": den,
        "better": better, "tied": tied, "worse": worse,
        "largest_window_share": share, "loo_sign_reversals": reversals,
    }


def identical_to_scribe(report: dict, instance_id: str) -> list[str]:
    """Items where the provider's hypothesis is byte-identical to Scribe's.

    The fusion route answers `{"text": …}` and nothing else, so a window that fell
    back to exact Scribe is invisible in the benchmark: it reports `ok`. It is not
    invisible here, because "exact Scribe" means exact. This over-counts only if a
    fused window happens to reproduce Scribe word for word, which the trace count
    is there to cross-check.
    """
    out = []
    for item in report["items"]:
        mine = item["perProvider"].get(instance_id) or {}
        scribe = item["perProvider"].get("scribe") or {}
        if mine.get("status") != "ok" or scribe.get("status") != "ok":
            continue
        if mine.get("hypothesisText") and mine["hypothesisText"] == scribe.get("hypothesisText"):
            out.append(item["itemId"])
    return out


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    path, target, baselines = sys.argv[1], sys.argv[2], sys.argv[3:]
    report = load(path)

    print(f"run: {report['manifest']['id']}")
    print(f"items in sample: {len(report['items'])}")

    if any(p["instanceId"] == "scribe" for p in report["providers"]) and target != "scribe":
        same = identical_to_scribe(report, target)
        print(f"\nwindows where {target} is byte-identical to scribe: {len(same)}"
              f" — cross-check against the trace count of non-fused outcomes")
        if same:
            print("  " + ", ".join(same[:10]) + (" …" if len(same) > 10 else ""))

    excluded = set(identical_to_scribe(report, target)) if target != "scribe" else set()

    for metric in ("wer", "wer-nofillers"):
        print(f"\n=== metric: {metric} ===")
        rows = {pid: per_item(report, pid, metric) for pid in [target, *baselines]}
        for pid in [target, *baselines]:
            row = provider_row(report, pid)
            score, num, den = pooled(rows[pid])
            r = rates(rows[pid])
            print(f"{pid:34s} pooled {score:.5f}  ({num}/{den})  "
                  f"scored {len(rows[pid])}  ok {row['ok']} err {row['errors']}  "
                  f"sub {r['sub']:.5f} del {r['del']:.5f} ins {r['ins']:.5f}")

        print("\n-- training-disjoint strata (the only ones that are model quality) --")
        for pid in [target, *baselines]:
            st = strata(report, pid, metric)
            cells = [f"{k.split(':')[1]} {st[k]['score']:.5f} (n={st[k]['items']})"
                     for k in TRAINING_DISJOINT if k in st]
            print(f"{pid:34s} " + "  ".join(cells))

        for base in baselines:
            d = paired_delta(rows[target], rows[base])
            if not d.get("items"):
                continue
            print(f"\n-- {target} minus {base}, paired on {d['items']} items --")
            print(f"   delta {d['delta']:+.5f} over {d['ref_words']} reference words")
            print(f"   better {d['better']} / tied {d['tied']} / worse {d['worse']}")
            print(f"   largest single window = {d['largest_window_share']:.1%} of the net edit delta")
            print(f"   leave-one-out sign reversals: {d['loo_sign_reversals']}")

        if excluded:
            # A window that fell back is a Scribe row wearing the fusion's name. It
            # belongs in the headline number -- it is what the system did -- but the
            # question "how good is the fusion when it fuses" needs it dropped from
            # every provider at once, or the sets stop being paired.
            keep = {pid: {k: v for k, v in rows[pid].items() if k not in excluded}
                    for pid in [target, *baselines]}
            print(f"\n-- secondary, {len(excluded)} non-fused window(s) dropped from every provider --")
            for pid in [target, *baselines]:
                score, num, den = pooled(keep[pid])
                print(f"   {pid:31s} pooled {score:.5f}  ({num}/{den})  items {len(keep[pid])}")


if __name__ == "__main__":
    main()
