# -*- coding: utf-8 -*-
"""Turn the recorded live provider responses into a replay bundle.

Why this exists: the live route test proves the path works on ONE window, and it
costs three vendor calls each time it runs. The first live attempt fell back with
`timing_invariant` and the second did not, which means the interesting question
is not "does it work" but "how often does it fall back" -- and that question
needs many windows, not another payment.

The bundle is `<audio_sha256>.<provider>.json` holding the provider-NATIVE
response, so the route runs its own normalizers over real vendor output. It is
the same fake-provider mechanism the 391-window gate uses; only the recordings
are different.

The bundle holds transcript text. It goes under ~/.cache/oc-public, never git.

Usage: .venv-eval/bin/python -m eval.live_fusion.build_replay_bundle [--stack local|gpu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

CACHE = Path(os.environ.get("LIVE_FUSION_CACHE")
             or Path.home() / ".cache/oc-public/live-fusion-2026-09")
AUDIO_DIR = Path.home() / ".cache/oc-public/bench_windows"
OUT = Path(os.environ.get("LIVE_REPLAY_DIR")
           or Path.home() / ".cache/oc-public/live-replay-2026-09")

SCHEMA_REV = {"scribe": "scribe-words/1", "soniox": "soniox-tokens/1", "ours": "ocasr-words/1"}
MODEL = {"scribe": "scribe_v2", "soniox": "stt-async-v5", "ours": "oc-asr"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["local", "gpu"], default="gpu",
                    help="which recording of `ours` to put in the bundle")
    args = ap.parse_args()
    ours_dir = "ours_gpu" if args.stack == "gpu" else "ours"

    OUT.mkdir(parents=True, exist_ok=True)
    by_window: dict[str, dict[str, Path]] = {}
    for provider, directory in (("scribe", "scribe"), ("soniox", "soniox"), ("ours", ours_dir)):
        source = CACHE / directory
        if not source.exists():
            print(f"no recordings in {source}")
            return 2
        for file in source.glob("*.json"):
            window_id, params_sha, _ = file.name.rsplit(".", 2)
            by_window.setdefault(window_id, {})[provider] = file

    complete = {w: p for w, p in by_window.items() if len(p) == 3}
    written = 0
    index = {}
    for window_id, providers in sorted(complete.items()):
        wav = AUDIO_DIR / f"{window_id}.wav"
        if not wav.exists():
            continue
        audio_sha = sha256_file(wav)
        index[window_id] = audio_sha
        for provider, file in providers.items():
            params_sha = file.name.rsplit(".", 2)[1]
            entry = {
                "identity": {"model": MODEL[provider], "paramsSha": params_sha[:16],
                             "schemaRev": SCHEMA_REV[provider]},
                "raw": json.loads(file.read_text(encoding="utf-8")),
            }
            (OUT / f"{audio_sha}.{provider}.json").write_text(
                json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            written += 1

    (OUT / "INDEX.json").write_text(json.dumps({
        "stack": args.stack, "n_windows": len(index), "windows": index,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(index)} windows, {written} entries -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
