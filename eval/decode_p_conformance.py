"""Does the SERVED arm P decoder produce what the measured arm P produced?

This is a conformance check, not an efficacy estimate. The efficacy numbers are
already in `~/.cache/oc-public/chunking-decode-2026-08/results-eval.json` over all
39 frozen windows and are not recomputed here.

Three windows, chosen as a minimum coverage set and pinned by ID:

* `win_orestiada_mar23_2026_6383163` -- every boundary at a reported silence, and
  the largest single-window gain P has (WER 0.1688 -> 0.0942);
* `win_orestiada_dec11_2025_600052` -- two FORCED boundaries, the case where P's
  mechanism has nothing to cut at;
* `win_argos_apr24_2026_1015557`    -- a window where P is WORSE than the control
  (0.1646 -> 0.2222), included so the check is not cherry-picked.

For each window it decodes the audio twice on this machine: once through
`serve.decode_p` (the served policy) and once as arm A, the control -- one call
with the frozen CONTROL config over the whole window, which is what
faster-whisper's own blind 30 s chunking does. Both use the experiment's
per-window seed, so a difference is attributable to the segmentation.

    SC=~/.cache/oc-public .venv-eval/bin/python -m eval.decode_p_conformance
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval import chunking_decode as CD          # noqa: E402
from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from serve import decode_p as DP                # noqa: E402
import decode_ablation as DA                    # noqa: E402

WINDOWS = [
    "win_orestiada_mar23_2026_6383163",
    "win_orestiada_dec11_2025_600052",
    "win_argos_apr24_2026_1015557",
]
P_CACHE = DA.sc() / "chunking-decode-2026-08" / "eval-P.json"
A_CACHE = DA.sc() / "decode-ablation" / "eval-A.json"


def _rates(counts: tuple[int, int, int, int]) -> dict:
    s, d, i, n = counts
    return {"wer": (s + d + i) / n, "sub": s / n, "del": d / n, "ins": i / n,
            "ref_tokens": n}


def main() -> None:
    import ctranslate2

    rows = {row["window_id"]: row for row in CD.eval_rows()}
    cached_p = json.loads(P_CACHE.read_text())["windows"]
    cached_a = json.loads(A_CACHE.read_text())["windows"]

    decoder = DP.DecodeP.load()                 # refuses a model that is not ct2-fixed
    print(f"policy_sha256 {DP.policy_sha256()}")
    print(f"model         {CD.MODEL_ARTIFACT_ID} {CD.verify_model()}\n")

    out = []
    for window_id in WINDOWS:
        row = rows[window_id]
        wav = DA.sc() / "bench_windows" / f"{window_id}.wav"
        reference = ftoks(DA.reference_text(window_id))
        seed = DA.seed_for("A", window_id)
        audio = CD._load_audio(wav)

        started = time.monotonic()
        served = decoder.transcribe(audio, seed=seed)
        p_wall = time.monotonic() - started

        started = time.monotonic()
        ctranslate2.set_random_seed(seed)
        control_text, _, _ = CD._decode_one(decoder.model, audio, dict(CD.CONTROL))
        a_wall = time.monotonic() - started

        record = {
            "window_id": window_id,
            "audio_seconds": round(served.audio_seconds, 2),
            "seed": seed,
            "P": _rates((*sdi(reference, ftoks(served.text)), len(reference))),
            "A": _rates((*sdi(reference, ftoks(control_text)), len(reference))),
            "P_cached": _rates((*sdi(reference, ftoks(cached_p[window_id]["text"])),
                                len(reference))),
            "A_cached": _rates((*sdi(reference, ftoks(cached_a[window_id]["text"])),
                                len(reference))),
            "P_text_matches_cache": served.text.strip() ==
                cached_p[window_id]["text"].strip(),
            "A_text_matches_cache": control_text.strip() ==
                cached_a[window_id]["text"].strip(),
            "n_pieces": served.counters["n_pieces"],
            "boundaries": [{"at": b.at, "kind": b.kind,
                            "silence_seconds": b.silence_seconds}
                           for b in served.boundaries],
            "wall_seconds": {"P": round(p_wall, 1), "A": round(a_wall, 1)},
        }
        out.append(record)
        print(f"{window_id}")
        print(f"  pieces {record['n_pieces']:2d}  "
              f"boundaries {[b['kind'] for b in record['boundaries']]}")
        print(f"  served  P WER {record['P']['wer']:.4f}   "
              f"A WER {record['A']['wer']:.4f}   "
              f"delta {record['P']['wer'] - record['A']['wer']:+.4f}")
        print(f"  cached  P WER {record['P_cached']['wer']:.4f}   "
              f"A WER {record['A_cached']['wer']:.4f}")
        print(f"  byte-identical to cache: P={record['P_text_matches_cache']} "
              f"A={record['A_text_matches_cache']}")
        print(f"  wall  P {p_wall:.0f}s  A {a_wall:.0f}s\n")

    destination = DA.sc() / "decode-p-conformance-2026-08.json"
    destination.write_text(json.dumps({
        "policy_sha256": DP.policy_sha256(),
        "model_artifact_id": CD.MODEL_ARTIFACT_ID,
        "environment": CD.environment(),
        "windows": out,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
