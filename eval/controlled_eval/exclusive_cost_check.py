#!/usr/bin/env python3
"""Cost projection for the frozen ~$20 stop rule.

Execution deviation from the preregistration, recorded here rather than hidden:
the spec has `exclusive_diar_run.py` project cost from the `quantity` field of the
first ten completed jobs, but `GET /v1/jobs/{id}` does not return `quantity` — only
the `GET /v2/jobs` listing does. The rule is therefore evaluated here instead, from
the listing, over the jobs actually billed. The threshold, the arithmetic and the
consequence are unchanged.

Usage: python eval/controlled_eval/exclusive_cost_check.py [n_planned]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import API, api_key, curl, log  # noqa: E402

EUR_PER_HOUR = 0.112
USD_PER_EUR = 1.10
CAP_USD = 20.0


def main():
    n_planned = int(sys.argv[1]) if len(sys.argv) > 1 else 285
    key = api_key()
    cursor, rows = None, []
    while True:
        url = f"{API.replace('/v1', '/v2')}/jobs?limit=100" + (
            f"&cursor={cursor}" if cursor else "")
        r = curl([url, "-H", f"Authorization: Bearer {key}"])
        rows += r.get("items", [])
        cursor = r.get("nextCursor")
        if not cursor or len(rows) >= 2000:
            break

    billed = [x for x in rows if x.get("quantity")]
    secs = [x["quantity"] for x in billed]
    mean = sum(secs) / len(secs) if secs else 0
    proj_usd = (mean * n_planned / 3600) * EUR_PER_HOUR * USD_PER_EUR
    spent_usd = (sum(secs) / 3600) * EUR_PER_HOUR * USD_PER_EUR

    log(f"{len(billed)} billed jobs visible, mean {mean:.0f} s/job")
    log(f"spent so far (all jobs on this key, all experiments): "
        f"{sum(secs) / 3600:.2f} h  ~${spent_usd:.2f}")
    log(f"projection for {n_planned} jobs: {mean * n_planned / 3600:.2f} h  "
        f"~${proj_usd:.2f}  (cap ${CAP_USD:.0f})")
    log("RULE: " + ("under cap, proceed" if proj_usd <= CAP_USD
                    else "OVER CAP -> comparator subset fallback"))
    print(json.dumps({"n_billed_visible": len(billed), "mean_sec": round(mean, 1),
                      "projected_usd": round(proj_usd, 2),
                      "over_cap": proj_usd > CAP_USD}))


if __name__ == "__main__":
    main()
