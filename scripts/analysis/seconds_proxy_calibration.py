"""How wrong is the uncovered-seconds proxy? (issue #23, validity check)

The coverage audit has no word timestamps -- Soniox was run text-only and re-running
it would cost money. It converts uncovered tokens to seconds with a uniform rate:

    seconds = uncovered_tokens * clip_duration / n_clip_tokens

That rate is wrong whenever a clip carries silence or a change of speech rate. This
script measures how wrong, using the one place where both a token stream and real
word timestamps exist for free: the cached whisper-turbo word-level transcription of
the benchmark windows.

Method. Draw pseudo-clips from the windows with the same duration distribution as
the deletion-hard rows themselves. Inside each pseudo-clip, take a contiguous run of
k tokens, estimate its duration with the proxy, and compare against the real span
from the timestamps. Report the error distribution and, at the audit's 1.0 s gate,
the false-positive and false-negative rates.

    SC=~/.cache/oc-public .venv-eval/bin/python \
        scripts/analysis/seconds_proxy_calibration.py
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.scoring import wtoks  # noqa: E402

CACHE = Path(os.environ.get("SC", str(Path.home() / ".cache/oc-public")))
TURBO = CACHE / "whisper_turbo"
COVERAGE = CACHE / "deletion-hard-audit/coverage.json"
OUT = CACHE / "deletion-hard-audit/seconds-proxy-calibration.json"

FILLER_RE = re.compile(r"^(ε{2,}|μ{2,}|α{2,}|ο{3,}|χμ+|εμ+|μχ+|χ{2,})$")
SEED = 20260816
N_DRAWS = 20000
GATE = 1.0


def ftoks(text: str) -> list[str]:
    return [t for t in wtoks(text) if not FILLER_RE.match(t)]


def stream(pk: dict) -> list[tuple[float, float]]:
    """(start, end) per token, in window time."""
    out = []
    for w in pk["wordLevelTranscription"]:
        tt = ftoks(w["text"])
        if not tt:
            continue
        a, b = float(w["start"]), float(w["end"])
        span = (b - a) / len(tt)
        for n in range(len(tt)):
            out.append((a + n * span, a + (n + 1) * span))
    return out


def main() -> None:
    rng = random.Random(SEED)
    windows = []
    for p in sorted(TURBO.glob("win_*.json")):
        try:
            pk = json.loads(p.read_text())["output"]
        except Exception:
            continue
        toks = stream(pk)
        if len(toks) > 20:
            windows.append(toks)
    if not windows:
        raise SystemExit(f"no whisper-turbo windows under {TURBO}")

    # Duration distribution of the rows the proxy is actually applied to. An
    # unweighted grid would over-sample the long clips, and the proxy's error is
    # duration-dependent, so the calibration has to match the audited bucket.
    cov = json.loads(COVERAGE.read_text())
    q = cov["deletion_hard"]["all_measured"]["uncovered_sec"]  # only for provenance
    dur_grid = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    bins = cov["gate_rate_by_duration"]
    dur_w = []
    for d in dur_grid:
        w = 0
        for key, b in bins.items():
            lo, hi = (float(x) for x in key.strip("[)").split(","))
            if lo <= d < hi:
                # spread each bin's mass over the grid points inside it
                n_pts = sum(1 for g in dur_grid if lo <= g < hi)
                w = b["n_witnessed"] / max(1, n_pts)
                break
        dur_w.append(w)
    if not any(dur_w):
        raise SystemExit("no duration weights derived from the coverage file")

    rows = []
    for _ in range(N_DRAWS):
        toks = rng.choice(windows)
        D = rng.choices(dur_grid, weights=dur_w, k=1)[0]
        t0 = rng.uniform(toks[0][0], max(toks[0][0], toks[-1][1] - D))
        inside = [t for t in toks if t[0] >= t0 and t[1] <= t0 + D]
        if len(inside) < 2:
            continue
        n = len(inside)
        k = rng.randint(1, min(n, 12))
        i = rng.randrange(0, n - k + 1)
        true_s = inside[i + k - 1][1] - inside[i][0]
        est_s = k * D / n
        rows.append((true_s, est_s, k, n, D))

    if not rows:
        raise SystemExit("every draw was rejected -- no pseudo-clip held two tokens")
    err = [e - t for t, e, *_ in rows]
    abs_err = sorted(abs(x) for x in err)
    ratio = sorted(e / t for t, e, *_ in rows if t > 0)

    def q_(xs, p):
        return round(xs[min(len(xs) - 1, int(p * len(xs)))], 3)

    tp = sum(1 for t, e, *_ in rows if e > GATE and t > GATE)
    fp = sum(1 for t, e, *_ in rows if e > GATE and t <= GATE)
    fn = sum(1 for t, e, *_ in rows if e <= GATE and t > GATE)
    tn = sum(1 for t, e, *_ in rows if e <= GATE and t <= GATE)

    out = {
        "generated_for": "github issue #23 -- validity of the uncovered-seconds proxy",
        "source": str(TURBO),
        "windows_used": len(windows),
        "draws": len(rows),
        "seed": SEED,
        "gate_s": GATE,
        "coverage_file_checked": str(COVERAGE),
        "coverage_uncovered_sec_median": q["median"],
        "abs_error_s": {"median": q_(abs_err, 0.5), "p75": q_(abs_err, 0.75),
                        "p90": q_(abs_err, 0.90), "p95": q_(abs_err, 0.95)},
        "est_over_true_ratio": {"p10": q_(ratio, 0.10), "median": q_(ratio, 0.50),
                                "p90": q_(ratio, 0.90),
                                "mean": round(statistics.fmean(ratio), 3)},
        "at_the_gate": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "flagged_share_est": round((tp + fp) / len(rows), 4),
            "flagged_share_true": round((tp + fn) / len(rows), 4),
        },
        "near_gate_stratum": {
            "definition": "draws whose ESTIMATE lands in [1.0, 3.0] s -- the band the "
                          "audit's flagged rows actually sit in (median flagged row: "
                          "5 uncovered tokens, 20 clip tokens, 8.1 s clip -> ~2.0 s)",
            **(lambda sel: {
                "n": len(sel),
                "share_truly_over_gate": round(
                    sum(1 for t, e, *_ in sel if t > GATE) / len(sel), 4) if sel else None,
            })([r for r in rows if 1.0 < r[1] <= 3.0]),
        },
        "reading": "The proxy is biased upward exactly where a clip carries silence "
                   "(it spreads the clip's whole duration over its spoken tokens). "
                   "precision below 1 means the audit's flagged share is an OVER-"
                   "estimate of rows with a real >1.0 s gap; the ratio of "
                   "flagged_share_est to flagged_share_true is the correction "
                   "factor to apply to the headline number.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
