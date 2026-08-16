#!/usr/bin/env python3
"""A reusable bench for post-fusion ideas: substrate, cross-fitting, gates.

Every fusion experiment on this project has so far rebuilt the same four things by
hand — load the benchmark, drop the sealed windows, align the trio, score with the
frozen scorer — and then invented its own cross-validation. This module is the
deliverable half of wayfinder #24: an idea is a small object, and `evaluate` returns
the out-of-fold WER, the two rate gates, the clustered CI, the leave-one-out sweeps
and the oracle recovery, all under one frozen protocol.

SUBSTRATE. 247 two-minute windows, 144 meetings, 10 cities of the
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju` benchmark run, with the 6
sealed temporal-holdout windows of `eval-freeze-2026-08` removed by the same explicit
filter `exp_fusion_deletions.py` carries. The alignment is the exact 3-way
sum-of-pairs MSA of `msa.py` and the baseline text is W, the hierarchical per-column
vote of `exp-2026-08-16-composition-over-selection`. Alignments are cached under $SC
(they contain verbatim council speech; they never go in git).

CROSS-FITTING. An idea may look at the reference text of the training fold and at
nothing else. The default fold is the CITY, which is the coarsest and most honest
block available here: a threshold fitted on nine cities is applied to the tenth and
only those out-of-fold outputs are scored. Meeting folds are available for ideas that
need more training mass, and they are weaker — meetings of one city share a roster,
a room and a vocabulary.

WHAT CROSS-FITTING DOES NOT BUY. The class definitions, the candidate restrictions
and the arm list itself were chosen by a human who has already seen four passes over
these same 247 windows. Leave-one-city-out prices *parameter* overfitting only. An
idea with no fitted parameter gets a `fold_note` saying its out-of-fold number is
identical to its in-fold one by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import bench_data as B                        # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import rates, sdi        # noqa: E402
from eval.controlled_eval.msa import align3, compose, oracle_columns    # noqa: E402
from eval.controlled_eval.scoring import (cluster_bootstrap,            # noqa: E402
                                          head2head, wtoks)

RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
TRIO = ["scribe-v2-clean", "soniox", "oc-runpod-fixed-2026-08-10"]
BAND_FLOOR = 40                       # identical to exp_composition.py
N_WINDOWS = 247
N_SEALED_INSIDE = 6
N_BOOT = int(os.environ.get("N_BOOT", "10000"))


def log(m):
    print(m, flush=True)


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


@dataclass
class Window:
    """One two-minute window, aligned and voted."""
    item_id: str
    city: str
    meeting: str
    ref: list[str]
    hyps: list[list[str]]                 # in TRIO order
    pivot: int                            # index into TRIO of the whole-window vote
    cols: list[tuple]                     # MSA columns, entries None for epsilon
    decisions: list[dict]                 # per column: {col, token, reason}
    w_tokens: list[str]                   # the W arm's output
    v_tokens: list[str]                   # the whole-window vote's output
    in_training: bool

    @property
    def col_tokens(self):
        """Per column, the distinct non-epsilon candidates in (a, b, c) order."""
        out = []
        for col in self.cols:
            seen, cl = set(), []
            for e in col:
                if e is not None and e not in seen:
                    seen.add(e)
                    cl.append(e)
            out.append(cl)
        return out


@dataclass
class Substrate:
    windows: list[Window]
    meta: dict = field(default_factory=dict)

    def by_city(self) -> dict[str, list[Window]]:
        out: dict[str, list[Window]] = {}
        for w in self.windows:
            out.setdefault(w.city, []).append(w)
        return out


# ------------------------------------------------------------------ alignment
def _band(a, b, c) -> int:
    return max(BAND_FLOOR, max(len(a), len(b), len(c)) - min(len(a), len(b), len(c)) + 20)


def _align_one(payload):
    wid, toks, pivot = payload
    a, b, c = toks
    cols = align3(a, b, c, band=_band(a, b, c))
    w_tokens, decisions = compose(cols, pivot=pivot)
    return wid, cols, w_tokens, decisions


def _cache_path() -> Path:
    key = hashlib.sha256(
        (RUN_ID + "|" + ",".join(TRIO) + f"|band{BAND_FLOOR}|"
         + hashlib.sha256((ROOT / "eval/controlled_eval/msa.py").read_bytes()).hexdigest()
         ).encode()).hexdigest()[:16]
    return sc() / "fusion_lab" / f"align_{key}.json"


def load_substrate(workers: int | None = None, limit: int | None = None) -> Substrate:
    report = B.load_report(RUN_ID)
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    before = len(items)
    items = [it for it in items if it["item_id"] not in sealed]
    assert before - len(items) == N_SEALED_INSIDE, \
        f"expected {N_SEALED_INSIDE} sealed windows inside this run, removed {before - len(items)}"
    assert len(items) == N_WINDOWS, f"expected {N_WINDOWS} windows, got {len(items)}"
    if limit:
        items = items[:limit]

    toks = {it["item_id"]: [wtoks(it["hyp"][p]) for p in TRIO] for it in items}
    v_pick = {it["item_id"]: B.consensus_pick(it, TRIO) for it in items}
    pivot = {w: TRIO.index(p) for w, p in v_pick.items()}

    cache = _cache_path()
    got: dict[str, dict] = {}
    if cache.exists():
        got = json.loads(cache.read_text())
    todo = [(it["item_id"], toks[it["item_id"]], pivot[it["item_id"]])
            for it in items if it["item_id"] not in got]
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        n = workers or int(os.environ.get("WORKERS", str(min(8, os.cpu_count() or 4))))
        log(f"aligning {len(todo)} windows on {n} workers (cache {cache})")
        with ProcessPoolExecutor(max_workers=n) as ex:
            for k, (wid, cols, wt, dec) in enumerate(
                    ex.map(_align_one, todo, chunksize=1), 1):
                got[wid] = {"cols": cols, "w_tokens": wt, "decisions": dec}
                if k % 25 == 0:
                    log(f"  {k}/{len(todo)}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(got, ensure_ascii=False))

    windows = []
    for it in items:
        wid = it["item_id"]
        g = got[wid]
        windows.append(Window(
            item_id=wid, city=it["city_id"], meeting=it["meeting_id"],
            ref=wtoks(it["ref"]), hyps=toks[wid], pivot=pivot[wid],
            cols=[tuple(c) for c in g["cols"]], decisions=g["decisions"],
            w_tokens=list(g["w_tokens"]),
            v_tokens=list(toks[wid][pivot[wid]]),
            in_training=it["in_training"]))
    return Substrate(windows, meta={
        "run_id": RUN_ID, "trio": TRIO, "n_windows": len(windows),
        "n_meetings": len({w.meeting for w in windows}),
        "n_cities": len({w.city for w in windows}),
        "ref_tokens": sum(len(w.ref) for w in windows),
        "align_cache": str(cache),
    })


# --------------------------------------------------------------------- ideas
class Idea:
    """An idea rewrites W's token stream for one window, using fold-fitted params.

    `fit` sees only training-fold windows and may read their reference text.
    `apply` sees one held-out window and the params, never a reference.
    """
    name = "identity"
    fitted = False                # False => LOCO is vacuous, and evaluate says so

    def fit(self, train: list[Window]):
        return None

    def apply(self, w: Window, params) -> list[str]:
        return list(w.w_tokens)


# ------------------------------------------------------------------ evaluate
def _counts(rows, idx=None):
    if idx is None:
        return [(r[0] + r[1] + r[2], r[3]) for r in rows]
    return [(r[idx], r[3]) for r in rows]


def _loo(rows_a, rows_b, keys):
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    ca, cb = _counts(rows_a), _counts(rows_b)
    allidx = list(range(len(keys)))

    def delta(idx):
        den = sum(ca[i][1] for i in idx)
        if den == 0:
            return None
        return (sum(ca[i][0] for i in idx) - sum(cb[i][0] for i in idx)) / den

    full = delta(allidx)
    ds = []
    for k, idx in groups.items():
        s = set(idx)
        d = delta([i for i in allidx if i not in s])
        if d is not None:
            ds.append((d, str(k)))
    ds.sort()
    if not ds:
        return {"full": full, "n_groups": len(groups), "sign_flips": 0}
    return {"full": full, "min": ds[0][0], "min_group": ds[0][1],
            "max": ds[-1][0], "max_group": ds[-1][1], "n_groups": len(ds),
            "sign_flips": sum(1 for d, _ in ds if (d > 0) != (full > 0))}


def _per_city(sub: Substrate, rows_a, rows_b) -> dict:
    by: dict[str, list[int]] = {}
    for i, w in enumerate(sub.windows):
        by.setdefault(w.city, []).append(i)
    out = {}
    for city, idx in sorted(by.items()):
        den = sum(rows_a[i][3] for i in idx)
        da = sum(sum(rows_a[i][:3]) for i in idx)
        db = sum(sum(rows_b[i][:3]) for i in idx)
        out[city] = {"n_windows": len(idx), "ref_tokens": den,
                     "wer_arm": da / den if den else None,
                     "wer_W": db / den if den else None,
                     "delta": (da - db) / den if den else None}
    deltas = [v["delta"] for v in out.values() if v["delta"] is not None]
    return {"per_city": out,
            "cities_better": sum(1 for d in deltas if d < 0),
            "cities_worse": sum(1 for d in deltas if d > 0),
            "cities_unchanged": sum(1 for d in deltas if d == 0),
            "city_unweighted_mean_delta": sum(deltas) / len(deltas) if deltas else None}


def evaluate(idea: Idea, sub: Substrate, fold: str = "city",
             n_boot: int = N_BOOT, extra: dict | None = None) -> dict:
    """Out-of-fold evaluation of one idea against W (and V), under the frozen gates.

    Returns out_of_fold WER/del/ins/sub, both rate gates, the paired
    meeting-clustered bootstrap CI, leave-one-out over window/meeting/city, and the
    share of the alignment-conditional column oracle recovered.
    """
    keyf = {"city": lambda w: w.city, "meeting": lambda w: w.meeting}[fold]
    groups: dict = {}
    for w in sub.windows:
        groups.setdefault(keyf(w), []).append(w)

    out_tokens: dict[str, list[str]] = {}
    fold_params = {}
    for k, held in groups.items():
        train = [w for w in sub.windows if keyf(w) != k]
        params = idea.fit(train)
        fold_params[str(k)] = params if isinstance(params, (int, float, str, type(None))) \
            else "<complex>"
        for w in held:
            out_tokens[w.item_id] = idea.apply(w, params)

    rows_arm, rows_w, rows_v, rows_oracle = [], [], [], []
    for w in sub.windows:
        rows_arm.append(sdi(" ".join(w.ref), " ".join(out_tokens[w.item_id])))
        rows_w.append(sdi(" ".join(w.ref), " ".join(w.w_tokens)))
        rows_v.append(sdi(" ".join(w.ref), " ".join(w.v_tokens)))
        rows_oracle.append(sdi(" ".join(w.ref),
                               " ".join(oracle_columns(w.cols, w.ref))))

    clusters = [w.meeting for w in sub.windows]
    arm, base, vv, orc = (rates(rows_arm), rates(rows_w), rates(rows_v),
                          rates(rows_oracle))

    def contrast(a_rows, b_rows):
        out = {}
        for metric, idx in (("wer", None), ("sub_rate", 0), ("del_rate", 1),
                            ("ins_rate", 2)):
            out[metric] = cluster_bootstrap(_counts(a_rows, idx), _counts(b_rows, idx),
                                            clusters, n_boot=n_boot)
        out["head2head_wer"] = head2head(_counts(a_rows), _counts(b_rows))
        return out

    vs_w = contrast(rows_arm, rows_w)
    changed = [w.item_id for w in sub.windows
               if out_tokens[w.item_id] != w.w_tokens]
    res = {
        "idea": idea.name,
        "fold": fold,
        "fitted": idea.fitted,
        "fold_note": ("out-of-fold == in-fold by construction: this idea fits no "
                      "parameter" if not idea.fitted else
                      f"parameters fitted leave-one-{fold}-out over "
                      f"{len(groups)} folds; only held-out outputs scored"),
        "n_windows": len(sub.windows),
        "n_folds": len(groups),
        "windows_changed_vs_W": len(changed),
        "out_of_fold": arm,
        "baseline_W": base,
        "baseline_V": vv,
        "oracle_column": orc,
        "vs_W": vs_w,
        "vs_V": contrast(rows_arm, rows_v),
        "gates": {
            "del_rate_gate": {"W": base["del_rate"], "arm": arm["del_rate"],
                              "pass": arm["del_rate"] <= base["del_rate"]},
            "ins_rate_gate": {"W": base["ins_rate"], "arm": arm["ins_rate"],
                              "pass": arm["ins_rate"] <= base["ins_rate"]},
            "wer_ci_excludes_zero": vs_w["wer"]["excludes_zero"],
            "wer_improves": arm["wer"] < base["wer"],
        },
        "loo_vs_W": {
            "window": _loo(rows_arm, rows_w, [w.item_id for w in sub.windows]),
            "meeting": _loo(rows_arm, rows_w, clusters),
            "city": _loo(rows_arm, rows_w, [w.city for w in sub.windows]),
        },
        "oracle_recovery_vs_W": ((base["wer"] - arm["wer"]) / (base["wer"] - orc["wer"])
                                 if base["wer"] != orc["wer"] else None),
        # Codex job 55293f6b: with ten cities a city-level bootstrap is not
        # reassuring, so the per-city deltas are reported raw beside the
        # meeting-clustered CI, together with the city-unweighted mean.
        "per_city": _per_city(sub, rows_arm, rows_w),
        "fold_params": fold_params,
    }
    res["gates"]["loo_sign_stable"] = all(
        res["loo_vs_W"][k]["sign_flips"] == 0 for k in ("window", "meeting", "city"))
    res["gates"]["all_pass"] = bool(
        res["gates"]["del_rate_gate"]["pass"] and res["gates"]["ins_rate_gate"]["pass"]
        and res["gates"]["wer_ci_excludes_zero"] and res["gates"]["wer_improves"]
        and res["gates"]["loo_sign_stable"])
    if extra:
        res["extra"] = extra
    return res


def summary_line(res: dict) -> str:
    a = res["out_of_fold"]
    g = res["gates"]
    return (f"{res['idea']:22s} wer={a['wer']:.4f} del={a['del_rate']:.4f} "
            f"ins={a['ins_rate']:.4f} sub={a['sub_rate']:.4f} "
            f"dWER={res['vs_W']['wer']['delta']:+.5f} "
            f"{res['vs_W']['wer']['ci95']} gates={'PASS' if g['all_pass'] else 'FAIL'}")


if __name__ == "__main__":
    s = load_substrate()
    log(json.dumps(s.meta, indent=1))
    r = evaluate(Idea(), s, n_boot=200)
    log(summary_line(r))
