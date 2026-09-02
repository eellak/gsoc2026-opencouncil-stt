"""Private, deterministic diagnosis of the dense-screen insertion regression.

The output contains window ids and token strings, so it belongs under the private
cache, never in git.  Alignment uses the exact tie-break from ``exp_same_stack.sdi``.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from eval.controlled_eval.eval_freeze import ftoks


def align(ref: list[str], hyp: list[str]) -> list[dict]:
    n, m = len(ref), len(hyp)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        distance[i][0] = i
    for j in range(m + 1):
        distance[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )

    operations = []
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and distance[i][j] ==
                distance[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])):
            operations.append({
                "op": "match" if ref[i - 1] == hyp[j - 1] else "sub",
                "ref_index": i - 1,
                "hyp_index": j - 1,
                "ref": ref[i - 1],
                "hyp": hyp[j - 1],
            })
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            operations.append({"op": "del", "ref_index": i - 1,
                               "ref": ref[i - 1]})
            i -= 1
        else:
            operations.append({"op": "ins", "hyp_index": j - 1,
                               "hyp": hyp[j - 1]})
            j -= 1
    operations.reverse()
    return operations


def window_analysis(ref: list[str], segments: list[str]) -> dict:
    hyp = [token for segment in segments for token in ftoks(segment)]
    operations = align(ref, hyp)
    counts = collections.Counter(op["op"] for op in operations)
    inserted = []
    for op in operations:
        if op["op"] != "ins":
            continue
        index = op["hyp_index"]
        token = op["hyp"]
        inserted.append({
            "token": token,
            "hyp_index": index,
            "repeated_neighbor": ((index > 0 and hyp[index - 1] == token) or
                                  (index + 1 < len(hyp) and hyp[index + 1] == token)),
            "occurs_in_reference": token in ref,
            "numeric": token.isdigit(),
        })
    return {
        "sub": counts["sub"],
        "del": counts["del"],
        "ins": counts["ins"],
        "errors": counts["sub"] + counts["del"] + counts["ins"],
        "insertions": inserted,
    }


def build_report(validation: Path, hypotheses: Path) -> dict:
    rows = json.loads(validation.read_text())["eval_windows"]
    by_id = {row["window_id"]: row for row in rows}
    seeds = []
    token_added = collections.Counter()
    token_removed = collections.Counter()
    added_categories = collections.Counter()

    for seed in (13, 29, 47):
        states = {}
        for arm in ("A", "B"):
            decoded = json.loads((hypotheses / f"{arm}_s{seed}.json").read_text())
            states[arm] = {
                key: window_analysis(by_id[key]["ref_tokens"], value["segments"])
                for key, value in decoded.items()
            }

        windows = []
        for key in sorted(by_id):
            a, b = states["A"][key], states["B"][key]
            a_tokens = collections.Counter(event["token"] for event in a["insertions"])
            b_tokens = collections.Counter(event["token"] for event in b["insertions"])
            added, removed = b_tokens - a_tokens, a_tokens - b_tokens
            token_added.update(added)
            token_removed.update(removed)
            for event in b["insertions"]:
                if added[event["token"]] <= 0:
                    continue
                added[event["token"]] -= 1
                added_categories["repeated_neighbor" if event["repeated_neighbor"]
                                 else "not_repeated_neighbor"] += 1
                added_categories["occurs_in_reference" if event["occurs_in_reference"]
                                 else "absent_from_reference"] += 1
                added_categories["numeric" if event["numeric"] else "non_numeric"] += 1
            windows.append({
                "window_id": key,
                "city": by_id[key]["city"],
                "meeting_id": by_id[key]["meeting_id"],
                "A": {field: a[field] for field in ("sub", "del", "ins", "errors")},
                "B": {field: b[field] for field in ("sub", "del", "ins", "errors")},
                "wer_gain_errors": a["errors"] - b["errors"],
                "insertion_delta": b["ins"] - a["ins"],
            })

        net_gain = sum(row["wer_gain_errors"] for row in windows)
        net_insertions = sum(row["insertion_delta"] for row in windows)
        ranked_gain = sorted(windows, key=lambda row: row["wer_gain_errors"], reverse=True)
        ranked_insertions = sorted(windows, key=lambda row: row["insertion_delta"], reverse=True)
        seeds.append({
            "seed": seed,
            "net_wer_gain_errors": net_gain,
            "net_extra_insertions": net_insertions,
            "windows_with_more_insertions": sum(row["insertion_delta"] > 0 for row in windows),
            "windows_with_fewer_insertions": sum(row["insertion_delta"] < 0 for row in windows),
            "top_wer_gain_windows": ranked_gain[:10],
            "top_insertion_windows": ranked_insertions[:10],
            "all_windows": windows,
        })

    return {
        "schema": "dense-insertion-private-v1",
        "seeds": seeds,
        "totals": {
            "net_extra_insertions": sum(seed["net_extra_insertions"] for seed in seeds),
            "added_insertion_tokens": sum(token_added.values()),
            "removed_insertion_tokens": sum(token_removed.values()),
            "added_categories": dict(added_categories),
            "top_added_tokens": token_added.most_common(30),
            "top_removed_tokens": token_removed.most_common(30),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.validation, args.hypotheses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "net_extra_insertions": report["totals"]["net_extra_insertions"],
        "added_categories": report["totals"]["added_categories"],
        "per_seed": [{key: seed[key] for key in (
            "seed", "net_wer_gain_errors", "net_extra_insertions",
            "windows_with_more_insertions", "windows_with_fewer_insertions")}
            for seed in report["seeds"]],
    }, indent=1))


if __name__ == "__main__":
    main()
