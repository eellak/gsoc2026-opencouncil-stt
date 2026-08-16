#!/usr/bin/env python3
"""Re-transcribe the 247 benchmark windows through the FREE Soniox realtime path,
keeping the per-token confidence the cached paid `stt-async-v5` text does not have.

This does NOT touch anything frozen. The cached `soniox` provider text of run
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju` is left exactly as it is; this
run produces a PARALLEL substrate (W-rt, see
`docs/specs/2026-08-16-w-rt-confidence-prereg.md`) under a new cache root.

Model `stt-rt-v4`, one realtime WebSocket session per ~140 s window, fed at ~1x,
`endpoint_detection=False`, `--lang el`, silence trimming OFF. N sessions run in
parallel; each is its own subprocess of `soniox-tools/file_transcribe.py --json`, so
each re-reads the temp key from `~/.cache/soniox-dictate/temp_key.json` at handshake
time rather than caching one key in memory for the whole batch.

Resumable: a window whose token file already exists is skipped, so a mid-run failure
costs only the windows that failed.

Raw tokens (verbatim council speech) are cached under
  ~/.cache/oc-public/composition-rt-2026-08/soniox-tokens/<item_id>.json
and never enter git.

Usage:
    python3 scripts/run_soniox_rt_bench.py [--jobs 12] [--limit N] [--list-only]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SONIOX_DIR = Path(os.environ.get("SONIOX_TOOLS_DIR",
                                 Path.home() / "projects" / "soniox-tools"))
PY = Path(os.environ.get("SONIOX_TOOLS_PYTHON", SONIOX_DIR / ".venv/bin/python"))


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def out_dir() -> Path:
    return sc() / "composition-rt-2026-08" / "soniox-tokens"


def window_ids() -> list[str]:
    """The exact 247 windows `fusion_lab.load_substrate` scores, same filter."""
    from eval.controlled_eval import bench_data as B
    from eval.controlled_eval import fusion_lab as F

    report = B.load_report(F.RUN_ID)
    items = B.common_items(report, B.provider_ids(report))
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    before = len(items)
    items = [it for it in items if it["item_id"] not in sealed]
    assert before - len(items) == F.N_SEALED_INSIDE, \
        f"expected {F.N_SEALED_INSIDE} sealed windows removed, removed {before - len(items)}"
    assert len(items) == F.N_WINDOWS, f"expected {F.N_WINDOWS} windows, got {len(items)}"
    return [it["item_id"] for it in items]


def transcribe(wav: Path, timeout: int) -> dict:
    r = subprocess.run(
        [str(PY), "file_transcribe.py", str(wav), "--lang", "el", "--json",
         "--realtime"],
        capture_output=True, text=True, timeout=timeout, cwd=str(SONIOX_DIR),
        errors="replace")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"rc={r.returncode} stderr={r.stderr[-300:]!r}")
    return json.loads(r.stdout)


_print_lock = threading.Lock()


def log(m: str) -> None:
    with _print_lock:
        print(m, flush=True)


def run_one(wid: str, attempts: int, timeout: int) -> tuple[str, bool, str]:
    dst = out_dir() / f"{wid}.json"
    if dst.exists():
        return wid, True, "cached"
    wav = sc() / "bench_windows" / f"{wid}.wav"
    if not wav.exists():
        return wid, False, "no audio"
    last = ""
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            d = transcribe(wav, timeout)
        except Exception as e:                                    # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            log(f"retry {wid} ({attempt}/{attempts}): {last[:180]}")
            time.sleep(5 * attempt)                               # linear back-off
            continue
        tmp = dst.with_suffix(".part")
        tmp.write_text(json.dumps(d, ensure_ascii=False))
        tmp.replace(dst)
        return wid, True, (f"{len(d['tokens'])} tok {time.time() - t0:.0f}s"
                           f" attempt {attempt}")
    return wid, False, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12,
                    help="parallel realtime sessions (soniox-core's working ceiling "
                         "is 18; back off on errors)")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-session subprocess timeout in seconds")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    if not PY.exists():
        sys.exit(f"no soniox-tools interpreter at {PY} — set SONIOX_TOOLS_DIR "
                 f"or SONIOX_TOOLS_PYTHON")
    ids = window_ids()
    if args.limit:
        ids = ids[:args.limit]
    out_dir().mkdir(parents=True, exist_ok=True)
    todo = [w for w in ids if not (out_dir() / f"{w}.json").exists()]
    log(f"{len(ids)} windows, {len(todo)} to transcribe, jobs={args.jobs}")
    if args.list_only:
        return

    t0 = time.time()
    failed: list[tuple[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_one, w, args.attempts, args.timeout) for w in todo]
        for f in futs:
            wid, ok, msg = f.result()
            done += 1
            if ok:
                log(f"[{done}/{len(todo)}] ok {wid} {msg}")
            else:
                failed.append((wid, msg))
                log(f"[{done}/{len(todo)}] FAIL {wid} {msg[:200]}")

    have = sum(1 for w in ids if (out_dir() / f"{w}.json").exists())
    log(f"done in {time.time() - t0:.0f}s. {have}/{len(ids)} windows cached, "
        f"{len(failed)} failed this pass")
    (out_dir().parent / "run_log.json").write_text(json.dumps({
        "jobs": args.jobs, "attempts": args.attempts, "n_windows": len(ids),
        "n_cached": have, "failed": failed,
        "wall_s": round(time.time() - t0, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ensure_ascii=False, indent=1))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
