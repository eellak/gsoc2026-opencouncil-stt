#!/usr/bin/env python3
"""Align the post-June held-out trio once and cache it, for the error atlas.

`exp_composition_postjune.py` recomputes the 3-way MSA every run and keeps only
aggregates. The atlas needs the columns themselves, repeatedly, so this builds them
once and caches under $SC. The cache holds verbatim council speech: it never goes in
git, exactly like `fusion_lab`'s alignment cache.

The window filter is COPIED from exp_composition_postjune.main() unchanged, so the
atlas describes the same 391 windows the published composition numbers describe.
Relaxing it would silently change the substrate.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.msa import align3, compose, oracle_columns   # noqa: E402
from eval.controlled_eval.scoring import wtoks                         # noqa: E402

OURS_NEW = "oc-cleanpack-cont-s47-b"
OURS_OLD = "oc-adapter-fixed-restage-2026-08-22"
TRIO = ["scribe", "soniox", OURS_NEW]
NEED_EXTRA = {OURS_OLD, "hf-openai-whisper-large-v3", "gpt-4o-transcribe",
              "gladia-prod"}


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def log(m):
    print(m, flush=True)


def _band(toks):
    lo, hi = min(len(t) for t in toks), max(len(t) for t in toks)
    return max(40, hi - lo + 20)


def _align_one(payload):
    wid, toks, pivot, ref = payload
    cols = align3(*toks, band=_band(toks))
    w_tokens, _ = compose(cols, pivot)
    return wid, cols, w_tokens, oracle_columns(cols, ref)


def _consensus_pivot(texts_w):
    """Index into TRIO of the hypothesis closest to the other two. Never sees ref."""
    from eval.controlled_eval.exp_fusion_deletions import sdi
    best, best_score = 0, None
    for pi, p in enumerate(TRIO):
        score = 0.0
        for q in TRIO:
            if q == p:
                continue
            s, d, i, r = sdi(texts_w[q], texts_w[p])
            score += (s + d + i) / max(1, r)
        if best_score is None or score < best_score:
            best, best_score = pi, score
    return best


def load(report: Path, workers: int | None = None) -> dict:
    """Returns {meta, windows: [{id, city, meeting, ref, hyps, cols, w, oracle}]}."""
    rh = hashlib.sha256(report.read_bytes()).hexdigest()[:16]
    key = hashlib.sha256(f"{rh}|{'|'.join(TRIO)}|atlas1".encode()).hexdigest()[:16]
    cache = sc() / "fusion_lab" / f"atlas_{key}.json"
    if cache.is_file():
        log(f"cache hit {cache}")
        return json.loads(cache.read_text())

    d = json.loads(report.read_text())
    providers = [p["instanceId"] for p in d["manifest"]["providers"]]
    need = set(TRIO) | NEED_EXTRA

    items, dropped = [], 0
    for it in d["items"]:
        pp = it.get("perProvider") or {}
        usable = {p for p in providers if p in pp and pp[p].get("status") == "ok"}
        if not need.issubset(usable):
            dropped += 1
            continue
        items.append({
            "id": it["itemId"],
            "city": it["cityId"],
            "meeting": it["meetingId"],
            "ref": wtoks(it["referenceText"]),
            "hyps": [wtoks(pp[p]["hypothesisText"]) for p in TRIO],
            "texts": {p: pp[p]["hypothesisText"] for p in TRIO},
        })
    log(f"{len(items)} windows ({dropped} dropped for a missing provider)")

    clusters = {(i["city"], i["meeting"]) for i in items}
    bare = len({m for _, m in clusters})
    log(f"clustering on (city, meeting): {len(clusters)}; meeting id alone: {bare}")
    assert len(items) < 100 or len(clusters) > bare, "check the clustering key"

    payloads = [(i["id"], i["hyps"], _consensus_pivot(i["texts"]), i["ref"])
                for i in items]
    workers = workers or int(os.environ.get("WORKERS",
                                            str(min(12, os.cpu_count() or 4))))
    log(f"aligning {len(payloads)} windows on {workers} workers")
    A = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (wid, cols, w, orc) in enumerate(
                ex.map(_align_one, payloads, chunksize=1), 1):
            A[wid] = {"cols": cols, "w": w, "oracle": orc}
            if n % 50 == 0:
                log(f"  {n}/{len(payloads)}")

    out = {
        "meta": {"run": d["manifest"]["id"], "report_sha256_16": rh, "trio": TRIO,
                 "n_windows": len(items), "n_clusters": len(clusters),
                 "n_cities": len({i["city"] for i in items})},
        "windows": [{"id": i["id"], "city": i["city"], "meeting": i["meeting"],
                     "ref": i["ref"], "hyps": i["hyps"],
                     "pivot": p[2], **A[i["id"]]}
                    for i, p in zip(items, payloads)],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    log(f"cached {cache} ({cache.stat().st_size/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    r = Path(os.environ["REPORT"])
    s = load(r)
    print(json.dumps(s["meta"], indent=1))
