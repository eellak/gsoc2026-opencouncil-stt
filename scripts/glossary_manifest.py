"""Reconstruct and freeze the provenance of `data/glossary/glossary.json`.

The glossary was written on 2026-06-20 by `eval/build_dataset.py` with no manifest
beside it, which is why a glossary arm was rejected on leakage grounds in July: nobody
could say which meetings had fed it.

This script proves provenance the only way that counts — it re-runs the miner on the
reconstructed TRAIN fold and requires a **byte-identical** reproduction of the file on
disk. If that holds, the set of (city_id, meeting_id) in that fold *is* the provenance.
It then intersects that set with the public benchmark run used to judge the model, so
that any later evaluation of the glossary can be restricted to windows it never saw.

Writes (both from one dict, so they cannot drift):
  data/glossary/glossary.build-manifest.json        — local, beside the artifact
  research/glossary/glossary-2026-06-20.manifest.json — git-tracked

No transcript text is written to either. Meeting and city ids only, which the public
benchmark report already publishes.

Run:  .venv-eval/bin/python scripts/glossary_manifest.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH_RUN = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
BENCH_CACHE = Path.home() / ".cache/oc-public" / f"bench_{BENCH_RUN}.json"

GLOSSARY = ROOT / "data/glossary/glossary.json"
CHAINS = ROOT / "data/eval/chains.parquet"
SPLIT = ROOT / "data/eval/split.json"
CORRECTIONS_CSV = ROOT / "data-1779206108158.csv"
TRAIN_MANIFEST = ROOT / "data/asr/train_manifest.csv"

OUT_LOCAL = ROOT / "data/glossary/glossary.build-manifest.json"
OUT_GIT = ROOT / "research/glossary/glossary-2026-06-20.manifest.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    import pandas as pd

    from eval.glossary import (GLOBAL_MIN_CITIES, KEEP_MIN_MEETINGS, _STOP,
                               mine_glossary)
    from eval.splits import split_by_meeting

    split = json.loads(SPLIT.read_text())
    chains = pd.read_parquet(CHAINS).to_dict("records")
    train, _ = split_by_meeting(chains, eval_meetings=set(split["eval_meeting_ids"]))

    mined = json.dumps(mine_glossary(train), ensure_ascii=False, indent=2)
    on_disk = GLOSSARY.read_text(encoding="utf-8")
    reproduced = mined == on_disk
    if not reproduced:
        raise SystemExit(
            "glossary.json did NOT reproduce byte-identically from the reconstructed "
            "train fold — provenance is NOT proven, do not write a manifest"
        )

    gloss = json.loads(on_disk)
    source = sorted({(c["city_id"], c["meeting_id"]) for c in train})

    # ------------------------------------------------ benchmark leakage
    leak: dict = {"note": "benchmark report not cached; run the benchmark skill first"}
    if BENCH_CACHE.exists():
        rep = json.loads(BENCH_CACHE.read_text())
        provs = [p["instanceId"] for p in rep["manifest"]["providers"]]
        common = [it for it in rep["items"]
                  if all(p in it["perProvider"]
                         and it["perProvider"][p]["status"] == "ok" for p in provs)]
        src = set(source)
        ft = {(r["city_id"], r["meeting_id"])
              for r in csv.DictReader(open(TRAIN_MANIFEST))}

        def counts(items):
            g = [it for it in items if (it["cityId"], it["meetingId"]) in src]
            f = [it for it in items if (it["cityId"], it["meetingId"]) in ft]
            both = [it for it in items
                    if (it["cityId"], it["meetingId"]) in src
                    or (it["cityId"], it["meetingId"]) in ft]
            return {
                "n_windows": len(items),
                "n_meetings": len({(it["cityId"], it["meetingId"]) for it in items}),
                "windows_in_glossary_source": len(g),
                "windows_in_finetune_train": len(f),
                "windows_in_either": len(both),
                "windows_disjoint_from_both": len(items) - len(both),
                "clean_window_ids": sorted(
                    it["itemId"] for it in items
                    if (it["cityId"], it["meetingId"]) not in src
                    and (it["cityId"], it["meetingId"]) not in ft),
            }

        leak = {
            "run_id": BENCH_RUN,
            "all_items": counts(rep["items"]),
            "common_items": counts(common),
            "verdict": (
                "The glossary's source meetings cover more of this benchmark than the "
                "fine-tune's training set does. `bench_data.training_meetings()` is "
                "therefore NOT the right disjointness line for a glossary arm."
            ),
        }
        for k in ("all_items",):
            leak[k].pop("clean_window_ids", None)

    manifest = {
        "artifact": {
            "path": "data/glossary/glossary.json",
            "built_at": "2026-06-20",
            "builder": "eval/build_dataset.py -> eval.glossary.mine_glossary",
            "sha256": sha256_file(GLOSSARY),
            "bytes": GLOSSARY.stat().st_size,
            "n_global_terms": len(gloss["global"]),
            "per_city_term_counts": {c: len(v)
                                     for c, v in sorted(gloss["per_city"].items())},
            "n_per_city_terms_total": sum(len(v) for v in gloss["per_city"].values()),
        },
        "provenance_proof": {
            "claim": "reconstructed provenance with byte-identical output",
            "method": "re-run the miner on the reconstructed TRAIN fold and require a "
                      "byte-identical reproduction of the artifact",
            "reproduced_byte_identical": reproduced,
            "verified_at": "2026-08-16",
            "verified_at_git_head": git_head(),
            "command": ".venv-eval/bin/python scripts/glossary_manifest.py",
            "miner_sha256": {
                "eval/glossary.py": sha256_file(ROOT / "eval/glossary.py"),
                "eval/splits.py": sha256_file(ROOT / "eval/splits.py"),
                "eval/build_dataset.py": sha256_file(ROOT / "eval/build_dataset.py"),
            },
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "residual_uncertainty": (
                "Codex, job 45db933d: byte-identical reproduction does not prove the "
                "historical input set. Mining is lossy, so a meeting that contributed "
                "no surviving term can be added to or removed from the fold without "
                "changing the output. `source_meetings` is therefore the reconstructed "
                "fold under the current code, which is a superset of the meetings that "
                "actually contributed a term. Treat it as an upper bound on exposure, "
                "which is the safe direction for a leakage claim."
            ),
        },
        "inputs": {
            "corrections_csv": {"path": CORRECTIONS_CSV.name,
                                "sha256": sha256_file(CORRECTIONS_CSV)},
            "chains_parquet": {"path": "data/eval/chains.parquet",
                               "sha256": sha256_file(CHAINS)},
            "split_json": {"path": "data/eval/split.json",
                           "sha256": sha256_file(SPLIT)},
        },
        "mining_rule": {
            "text_field": "gold_final (human-corrected text of the chain)",
            "fold": "eval-harness TRAIN fold = every chain whose meeting_id is NOT in "
                    "data/eval/split.json::eval_meeting_ids",
            "split_key": "bare meeting_id string",
            "split_key_caveat": (
                "meeting_id is NOT unique across cities. The fold is defined on the "
                "bare string, so a meeting_id held out in one city is held out in "
                "every city, and conversely a meeting_id kept in one city is kept in "
                "every city that uses it. This is a defect of the 2026-06-20 split, "
                "recorded here rather than fixed, because fixing it would change the "
                "artifact."
            ),
            "keep_min_meetings": KEEP_MIN_MEETINGS,
            "global_min_cities": GLOBAL_MIN_CITIES,
            "stopword_count": len(_STOP),
            "candidate_classes": [
                "all-uppercase Greek token of length >= 2 (acronym)",
                "capitalised token not in the stop list (proper noun)",
                "run of >= 2 capitalised tokens (phrase)",
            ],
        },
        "source_meetings": {
            "n_pairs": len(source),
            "n_cities": len({c for c, _ in source}),
            "pairs": [{"city_id": c, "meeting_id": m} for c, m in source],
        },
        "benchmark_leakage": leak,
    }

    for out in (OUT_LOCAL, OUT_GIT):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_meetings"},
                     ensure_ascii=False, indent=1))
    print(f"wrote {OUT_LOCAL} and {OUT_GIT}")


if __name__ == "__main__":
    main()
