# -*- coding: utf-8 -*-
"""Step 1 of `exp-2026-09-16-deletion-recovery`: how much is there to recover?

Scribe v2 with `no_verbatim: true` scores 0.09294 on the 391 benchmark windows and
deletes 0.02118 of the reference; the same vendor in verbatim mode deletes 0.00955.
The gap is words the clean mode drops at the source. This screen counts them, and
prices the ceiling of putting them back, before anyone pays for a listening test.

The donor is the SAME vendor in the SAME request shape, so no cross-vendor
alignment is involved: the two token streams differ only by what the mode
suppressed. That is the whole reason this is cheaper than patching from Soniox or
from our adapter.

WHAT THIS CANNOT DECIDE. Every number here is agreement-with-OpenCouncil, and the
reference was produced by the clean-read pipeline, so a correctly restored word can
still score as an insertion. A gain here is real evidence; a loss here is expected
and decides nothing. Fidelity is decided on the gold set, not here.

Counts, rates and window ids only. No transcript text is printed or written --
the hypotheses this reads are PII.

    .venv-eval/bin/python -m eval.deletion_recovery.screen [report.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.controlled_eval.eval_freeze import FILLER_RE            # noqa: E402
from eval.controlled_eval.scoring import wtoks                    # noqa: E402

REPORT = (Path.home() /
          ".cache/oc-public/bench_2026-09-15-clean-scribe-baseline-vs-fusion.json")
BASE_ID = "scribe-v2-clean"      # the shipped read, and the base transcript
DONOR_ID = "scribe"              # the same model in verbatim mode, the donor
OUT = Path(__file__).resolve().parents[2] / "eval/results_deletion_recovery_screen.json"

# The training-disjoint cities. Every other stratum is contaminated and is not
# read as model quality -- CLAUDE.md, and the benchmark runbook.
DISJOINT = ("argos", "orestiada")


def is_filler(token: str) -> bool:
    """The frozen hesitation rule of eval_freeze, imported rather than retyped."""
    return bool(FILLER_RE.match(token))


def _backtrace(base: list[str], donor: list[str]):
    """Levenshtein ops turning `base` into `donor`, as (op, base_i, donor_j).

    Ties are resolved match -> substitute -> insert -> delete, fixed here so the
    op string cannot drift between runs. Windows are a few hundred tokens, so the
    full matrix is cheap and a banded approximation would only add a parameter
    nobody froze.
    """
    n, m = len(base), len(donor)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1,                                  # base-only
                          d[i][j - 1] + 1,                                  # donor-only
                          d[i - 1][j - 1] + (base[i - 1] != donor[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and base[i - 1] == donor[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(("match", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1)); i -= 1; j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ops.append(("donor_only", None, j - 1)); j -= 1
        else:
            ops.append(("base_only", i - 1, None)); i -= 1
    ops.reverse()
    return ops


def only_in_donor(base: list[str], donor: list[str]) -> list[str]:
    """Tokens the donor emitted in a slot where the base emitted nothing.

    A substitution is deliberately NOT a candidate: both arms spoke in that slot,
    so the disagreement is about which word is right, not about a missing word,
    and re-inserting there would double the slot.
    """
    return [donor[j] for op, _i, j in _backtrace(base, donor) if op == "donor_only"]


def _blind_stream(base: list[str], donor: list[str]) -> tuple[list[str], list[int]]:
    """The base with every non-filler donor addition put back, and where they landed."""
    out: list[str] = []
    added: list[int] = []
    for op, i, j in _backtrace(base, donor):
        if op in ("match", "sub", "base_only"):
            out.append(base[i])
            continue
        tok = donor[j]
        if is_filler(tok):
            continue
        added.append(len(out))
        out.append(tok)
    return out, added


def restore(base: list[str], donor: list[str], oracle_ref: list[str] | None = None):
    """Put the donor's non-filler additions back into the base stream, in place.

    With `oracle_ref`, keep only the additions that land on a reference word AT
    THAT POSITION -- the UPPER BOUND of any selection rule, never a policy.

    Position matters, and the first version of this got it wrong: it kept any
    addition whose token appeared anywhere in the reference's unspent multiset.
    That credits a restoration which makes the transcript worse -- reference
    "α α β", base "α β", donor adding a trailing "α" scores as supported while
    raising the edit distance from 1 to 2. The support rate it produced was an
    overstatement, and it was published before this was caught.
    """
    stream, added = _blind_stream(base, donor)
    if oracle_ref is None:
        return stream, len(added)
    matched = set()
    for op, _i, j in _backtrace(oracle_ref, stream):
        if op == "match":
            matched.add(j)
    drop = {idx for idx in added if idx not in matched}
    out = [t for k, t in enumerate(stream) if k not in drop]
    return out, len(added) - len(drop)


def ops_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of `hyp` against `ref`."""
    s = de = ins = 0
    for op, _i, _j in _backtrace(ref, hyp):
        if op == "sub":
            s += 1
        elif op == "base_only":
            de += 1
        elif op == "donor_only":
            ins += 1
    return s, de, ins


def screen(report_path: Path = REPORT) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_city: dict[str, list[int]] = {}
    tot = {"windows": 0, "ref_words": 0, "candidates": 0, "fillers": 0,
           "non_fillers": 0, "restored_blind": 0, "restored_oracle": 0}
    edits = {"clean": [0, 0, 0], "blind": [0, 0, 0], "oracle": [0, 0, 0]}

    for item in report["items"]:
        pp = item["perProvider"]
        base_p, donor_p = pp.get(BASE_ID, {}), pp.get(DONOR_ID, {})
        if base_p.get("status") != "ok" or donor_p.get("status") != "ok":
            continue
        ref = wtoks(item.get("referenceText") or "")
        if not ref:
            continue
        base = wtoks(base_p.get("hypothesisText") or "")
        donor = wtoks(donor_p.get("hypothesisText") or "")

        cand = only_in_donor(base, donor)
        fillers = [t for t in cand if is_filler(t)]
        blind, kept_blind = restore(base, donor)
        oracle, kept_oracle = restore(base, donor, oracle_ref=ref)

        tot["windows"] += 1
        tot["ref_words"] += len(ref)
        tot["candidates"] += len(cand)
        tot["fillers"] += len(fillers)
        tot["non_fillers"] += len(cand) - len(fillers)
        tot["restored_blind"] += kept_blind
        tot["restored_oracle"] += kept_oracle
        for name, hyp in (("clean", base), ("blind", blind), ("oracle", oracle)):
            s, d, i = ops_counts(ref, hyp)
            edits[name][0] += s
            edits[name][1] += d
            edits[name][2] += i

        city = item.get("cityId") or "?"
        row = per_city.setdefault(city, [0, 0, 0, 0])     # clean_e, blind_e, ref, n
        row[0] += sum(ops_counts(ref, base))
        row[1] += sum(ops_counts(ref, blind))
        row[2] += len(ref)
        row[3] += 1

    n = tot["ref_words"] or 1
    result = {
        "protocol": "deletion-recovery-screen-2026-09-16",
        "experiment": "exp-2026-09-16-deletion-recovery",
        "report": report_path.name,
        "base": BASE_ID, "donor": DONOR_ID,
        "totals": tot,
        "wer": {name: sum(v) / n for name, v in edits.items()},
        "edits": {name: {"sub": v[0] / n, "del": v[1] / n, "ins": v[2] / n}
                  for name, v in edits.items()},
        "per_city": {c: {"clean": v[0] / v[2], "blind": v[1] / v[2],
                         "delta": (v[1] - v[0]) / v[2], "windows": v[3],
                         "ref_words": v[2], "training_disjoint": c in DISJOINT}
                     for c, v in sorted(per_city.items())},
    }
    return result


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT
    if not path.exists():
        print(f"no report at {path}")
        return 2
    r = screen(path)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    t, w, e = r["totals"], r["wer"], r["edits"]
    print(f"windows {t['windows']}  reference words {t['ref_words']}")
    print(f"donor-only tokens {t['candidates']}  "
          f"fillers {t['fillers']} ({t['fillers'] / max(1, t['candidates']):.1%})  "
          f"non-fillers {t['non_fillers']}")
    print(f"restored blind {t['restored_blind']}  restored oracle {t['restored_oracle']}")
    print()
    for name in ("clean", "blind", "oracle"):
        print(f"{name:7s} WER {w[name]:.5f}  "
              f"sub {e[name]['sub']:.5f} del {e[name]['del']:.5f} ins {e[name]['ins']:.5f}")
    print()
    for c, v in r["per_city"].items():
        mark = " *training-disjoint" if v["training_disjoint"] else ""
        print(f"{c:14s} clean {v['clean']:.5f}  blind {v['blind']:.5f}  "
              f"{v['delta']:+.5f}  n={v['windows']}{mark}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
