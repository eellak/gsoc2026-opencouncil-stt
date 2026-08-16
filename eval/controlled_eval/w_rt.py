#!/usr/bin/env python3
"""W-rt: the 247-window fusion substrate with the FREE realtime Soniox arm.

Why this module exists, in one paragraph. The cached Soniox text of the benchmark run
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju` came from the paid
`stt-async-v5` and carries no per-word confidence — the client threw it away. The only
free path is `stt-rt-v4`, a DIFFERENT MODEL, so its text differs. Soniox is one of W's
three voters, so re-running it changes W itself. Attaching new confidences to the old
text is therefore forbidden: this module builds a PARALLEL substrate, W-rt, in which
the other two systems are byte-identical from cache and only the Soniox arm is
replaced. Every number measured on it is measured against the W-rt baseline, never
against the frozen old-W numbers.

Nothing frozen is modified. New cache root
(`$SC/composition-rt-2026-08/`), no edit to `fusion_lab.py`, `msa.py`,
`column_classes.py` or `scoring.py`. The 6 sealed temporal-holdout windows of
`eval-freeze-2026-08` are removed by the same explicit filter `fusion_lab` carries and
are never transcribed, aligned or scored here.

Preregistration: `docs/specs/2026-08-16-w-rt-confidence-prereg.md`.

CONFIDENCE MAPPING. The Soniox hypothesis handed to the alignment is DERIVED FROM THE
CACHED FINAL TOKENS, not from the client's `text` field, so the token stream and the
confidence stream are the same object by construction and no post-hoc matching is
needed. Words are built by the production algorithm (`soniox_confidence_probe.
group_words`: finals only, subtokens exploded to runes carrying their token's
confidence, whitespace-delimited words, score = MIN confidence over LEXICAL runes
only) and mapped onto the scorer's normalized token space by `word_units`. Confidence
travels BY OCCURRENCE INDEX, never by token string, so repeated adjacent words are
never confused. A confidence that is missing, NaN, infinite or outside [0, 1] makes
that occurrence INELIGIBLE for every arm while leaving the baseline text unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import bench_data as B                       # noqa: E402
from eval.controlled_eval import fusion_lab as F                       # noqa: E402
from eval.controlled_eval.msa import align3, compose                   # noqa: E402
from eval.controlled_eval.scoring import wtoks                         # noqa: E402
from eval.soniox_confidence_probe import group_words, word_units       # noqa: E402

SONIOX_IDX = F.TRIO.index("soniox")
CACHE_ROOT_NAME = "composition-rt-2026-08"
RT_MODEL = "stt-rt-v4"


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def rt_root() -> Path:
    return sc() / CACHE_ROOT_NAME


def token_dir() -> Path:
    return rt_root() / "soniox-tokens"


def valid_conf(c) -> bool:
    """Frozen validity rule: a real number in [0, 1]. Everything else is ineligible."""
    if c is None or isinstance(c, bool):
        return False
    try:
        v = float(c)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and 0.0 <= v <= 1.0


def rt_arm(payload: dict) -> dict:
    """One window's Soniox-rt arm: normalized tokens plus a parallel confidence list.

    Returns {"tokens": [...], "conf": [...], "conf_min": [...], "conf_mean": [...],
             "stats": {...}}. `conf` is `conf_min_lex`, the production aggregate and
    the one every arm uses; the other two are sensitivity only. An entry is None when
    the confidence failed `valid_conf`.
    """
    words, wstats = group_words(payload.get("tokens", []))
    units, dropped, split = word_units(words)
    toks = [u["tok"] for u in units]

    def pick(u, key):
        v = u.get(key)
        return float(v) if valid_conf(v) else None

    conf = [pick(u, "conf_min_lex") for u in units]
    stats = dict(wstats)
    stats.update({"n_words": len(words), "n_units": len(units),
                  "words_dropped_by_normalization": dropped,
                  "words_split_into_several_tokens": split,
                  "units_with_invalid_confidence": sum(1 for c in conf if c is None),
                  "residual_nonfinal_tokens":
                      len(payload.get("residual_nonfinal_tokens") or [])})
    return {"tokens": toks, "conf": conf,
            "conf_min": [pick(u, "conf_min") for u in units],
            "conf_mean": [pick(u, "conf_mean") for u in units],
            "stats": stats}


def soniox_column_index(cols) -> list[int | None]:
    """For each MSA column, the index into the Soniox token stream, or None.

    Walking the columns in order and consuming one Soniox occurrence per non-epsilon
    entry is exact: the alignment emits the Soniox sequence in order. Matching tokens
    back by string would mis-attribute repeated words, which is the defect CodeRabbit
    found in the first version of the ceiling arms.
    """
    out: list[int | None] = []
    n = 0
    for col in cols:
        if col[SONIOX_IDX] is None:
            out.append(None)
        else:
            out.append(n)
            n += 1
    return out


# ------------------------------------------------------------------- substrate
def _align_one(payload):
    wid, toks, pivot = payload
    a, b, c = toks
    cols = align3(a, b, c, band=F._band(a, b, c))
    w_tokens, decisions = compose(cols, pivot=pivot)
    return wid, cols, w_tokens, decisions


def _cache_path(fingerprint: str) -> Path:
    key = hashlib.sha256(
        (F.RUN_ID + "|" + ",".join(F.TRIO) + f"|band{F.BAND_FLOOR}|rt|"
         + hashlib.sha256((ROOT / "eval/controlled_eval/msa.py").read_bytes()).hexdigest()
         + "|" + fingerprint).encode()).hexdigest()[:16]
    return rt_root() / f"align_{key}.json"


def load_substrate_rt(workers: int | None = None, strict: bool = True):
    """Build (and cache) the W-rt substrate.

    Returns (Substrate, conf) where `conf` maps item_id -> the per-occurrence
    confidence lists of that window's Soniox-rt arm.

    `strict=True` implements the preregistered stop rule: the primary analysis needs
    247/247 windows transcribed, and a missing token file aborts. `strict=False` is the
    explicitly exploratory reduced-set path and reports what is missing.
    """
    report = B.load_report(F.RUN_ID)
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    before = len(items)
    items = [it for it in items if it["item_id"] not in sealed]
    assert before - len(items) == F.N_SEALED_INSIDE, \
        f"expected {F.N_SEALED_INSIDE} sealed windows removed, removed {before - len(items)}"
    assert len(items) == F.N_WINDOWS, f"expected {F.N_WINDOWS} windows, got {len(items)}"

    missing = [it["item_id"] for it in items
               if not (token_dir() / f"{it['item_id']}.json").exists()]
    if missing and strict:
        raise SystemExit(
            f"{len(missing)}/{len(items)} windows have no Soniox-rt tokens; the "
            f"preregistered stop rule aborts the primary analysis. "
            f"First missing: {missing[:5]}")
    if missing:
        items = [it for it in items if it["item_id"] not in set(missing)]

    arms, models, hashes = {}, set(), {}
    for it in items:
        p = token_dir() / f"{it['item_id']}.json"
        raw = p.read_bytes()
        hashes[it["item_id"]] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        models.add(payload.get("model"))
        arms[it["item_id"]] = rt_arm(payload)
    if models != {RT_MODEL}:
        raise SystemExit(f"expected every token file from {RT_MODEL}, got {models}")

    toks = {}
    for it in items:
        wid = it["item_id"]
        t = [wtoks(it["hyp"][p]) for p in F.TRIO]
        t[SONIOX_IDX] = list(arms[wid]["tokens"])
        toks[wid] = t

    # the pivot is recomputed from the NEW trio: the old consensus pick was made
    # against stt-async-v5 text and is not a fact about W-rt.
    pivot = {}
    for it in items:
        wid = it["item_id"]
        hyp = {p: (" ".join(toks[wid][i]) if i == SONIOX_IDX else it["hyp"][p])
               for i, p in enumerate(F.TRIO)}
        pivot[wid] = F.TRIO.index(B.consensus_pick({"hyp": hyp}, F.TRIO))

    fingerprint = hashlib.sha256(
        "".join(f"{k}:{hashes[k]}" for k in sorted(hashes)).encode()).hexdigest()
    cache = _cache_path(fingerprint)
    got: dict[str, dict] = {}
    if cache.exists():
        got = json.loads(cache.read_text())
    todo = [(it["item_id"], toks[it["item_id"]], pivot[it["item_id"]])
            for it in items if it["item_id"] not in got]
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        n = workers or int(os.environ.get("WORKERS", str(min(8, os.cpu_count() or 4))))
        F.log(f"aligning {len(todo)} W-rt windows on {n} workers (cache {cache})")
        with ProcessPoolExecutor(max_workers=n) as ex:
            for k, (wid, cols, wt, dec) in enumerate(
                    ex.map(_align_one, todo, chunksize=1), 1):
                got[wid] = {"cols": cols, "w_tokens": wt, "decisions": dec}
                if k % 25 == 0:
                    F.log(f"  {k}/{len(todo)}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(got, ensure_ascii=False))

    windows, conf = [], {}
    for it in items:
        wid = it["item_id"]
        g = got[wid]
        cols = [tuple(c) for c in g["cols"]]
        n_son = sum(1 for c in cols if c[SONIOX_IDX] is not None)
        assert n_son == len(toks[wid][SONIOX_IDX]), \
            f"{wid}: alignment consumed {n_son} soniox tokens, stream has " \
            f"{len(toks[wid][SONIOX_IDX])}"
        windows.append(F.Window(
            item_id=wid, city=it["city_id"], meeting=it["meeting_id"],
            ref=wtoks(it["ref"]), hyps=toks[wid], pivot=pivot[wid],
            cols=cols, decisions=g["decisions"], w_tokens=list(g["w_tokens"]),
            v_tokens=list(toks[wid][pivot[wid]]), in_training=it["in_training"]))
        conf[wid] = {k: arms[wid][k] for k in ("conf", "conf_min", "conf_mean")}
        conf[wid]["stats"] = arms[wid]["stats"]

    sub = F.Substrate(windows, meta={
        "substrate": "W-rt",
        "run_id": F.RUN_ID, "trio": F.TRIO, "soniox_model": RT_MODEL,
        "n_windows": len(windows),
        "n_meetings": len({w.meeting for w in windows}),
        "n_cities": len({w.city for w in windows}),
        "ref_tokens": sum(len(w.ref) for w in windows),
        "n_missing_token_files": len(missing),
        "missing_item_ids": missing,
        "strict": strict,
        "align_cache": str(cache),
        "token_manifest_sha256": fingerprint,
    })
    return sub, conf


def manifest(sub, conf) -> dict:
    """The frozen data-lineage manifest: IDs, hashes and asserted totals."""
    rows = []
    for w in sub.windows:
        wav = sc() / "bench_windows" / f"{w.item_id}.wav"
        rows.append({
            "item_id": w.item_id, "city": w.city, "meeting": w.meeting,
            "ref_tokens": len(w.ref),
            "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest() if wav.exists()
            else None,
            "tokens_sha256": hashlib.sha256(
                (token_dir() / f"{w.item_id}.json").read_bytes()).hexdigest(),
            "n_soniox_units": len(conf[w.item_id]["conf"]),
        })
    return {"meta": sub.meta, "windows": rows}


if __name__ == "__main__":
    s, c = load_substrate_rt(strict=os.environ.get("STRICT", "1") == "1")
    print(json.dumps(s.meta, indent=1, ensure_ascii=False))
