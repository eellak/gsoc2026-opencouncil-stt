#!/usr/bin/env python3
"""Score an identification arm on the four numbers that must be read together.

Astra's instruction, and the reason for each:

1. **meeting-wide B-cubed F** — clustering quality, unchanged from `score_arms`.
2. **correct names over ALL turns** — not over *named* turns. Accuracy among named
   turns rises for free when a system abstains on the hard ones, so quoting it
   alone rewards silence.
3. **wrong names on unenrolled speakers** — the specific damage a zero threshold
   causes: speech from someone with no voiceprint handed an enrolled person's name.
4. **abstention rate** — the price paid for 2 and 3.

A threshold that improves 2 while raising 4 has bought nothing; a threshold that
improves 3 while holding 2 has bought something real.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CACHE = Path(os.environ.get("OC_CACHE", Path.home() / ".cache/oc-public"))
RUN_DIR = CACHE / "diarization-clustering-2026-09"

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from score_arms import assign, bcubed, reference_turns  # noqa: E402


def naming_report(turns: list[dict], segments: list[dict], enrolled: set[str]) -> dict:
    """Every turn is counted exactly once, in exactly one bucket.

    **A label is a name only if it is one of the enrolled ids we sent.** Anything
    else — `SPEAKER_05` from a plain diarize call, or a namespaced `unmatched:`
    id — is no claim at all. Counting those as wrong names would score arms that
    never attempt naming (a0, a3, c3) as if they misnamed everyone, which is both
    false and flattering to the arms that do attempt it.
    """
    n = len(turns)
    correct = wrong_enrolled = wrong_on_unenrolled = no_claim = 0
    for t in turns:
        h = assign(t, segments)
        if h is None or h not in enrolled:
            no_claim += 1
            continue
        if h == t["identity"]:
            correct += 1
        elif t["identity"] not in enrolled:
            # a name was given to speech belonging to someone never enrolled
            wrong_on_unenrolled += 1
        else:
            wrong_enrolled += 1
    assert correct + wrong_enrolled + wrong_on_unenrolled + no_claim == n
    pairs = [(t["identity"], assign(t, segments), 1.0) for t in turns]
    return {
        "turns": n,
        "correct_over_all_turns": round(correct / max(n, 1), 4),
        "wrong_name_on_unenrolled": round(wrong_on_unenrolled / max(n, 1), 4),
        "wrong_name_on_enrolled": round(wrong_enrolled / max(n, 1), 4),
        "no_name_claimed": round(no_claim / max(n, 1), 4),
        "bcubed": bcubed(pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", required=True, help="city__meeting")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    a = ap.parse_args()

    run = a.run_dir
    k = a.meeting
    snap = json.loads((run / "reference-snapshot" / f"{k}.json").read_text())
    # All reviewed turns, resolved or not: a turn on an unresolved tag is still
    # speech that can be misnamed, and excluding it would hide exactly the damage.
    turns = reference_turns(snap)
    enrolled = {v["personId"] for v in
                json.loads((run / "voiceprints" / f"{k}.json").read_text())}

    print(f"{k}: {len(turns)} reviewed turns, {len(enrolled)} enrolled voiceprints\n")
    hdr = f'{"arm":>10} {"B3 F":>7} {"correct/all":>12} {"wrong@unenr":>12} ' \
          f'{"wrong@enr":>10} {"no name":>8}'
    print(hdr)
    print("-" * len(hdr))
    for arm in a.arms:
        p = run / k / f"diarization_{arm}.json"
        if not p.exists():
            print(f'{arm:>10} {"(not run)":>7}')
            continue
        d = json.loads(p.read_text())
        segs = d.get("exclusiveDiarization") or d["diarization"]
        r = naming_report(turns, segs, enrolled)
        print(f'{arm:>10} {r["bcubed"]["f1"]:7.4f} {r["correct_over_all_turns"]:12.4f} '
              f'{r["wrong_name_on_unenrolled"]:12.4f} {r["wrong_name_on_enrolled"]:10.4f} '
              f'{r["no_name_claimed"]:8.4f}')
    print("\nRead together. Accuracy among NAMED turns is deliberately not shown:")
    print("abstaining on hard turns raises it without improving anything.")
    print("'no name' = the arm made no naming claim; arms without voiceprints")
    print("(a0, a3, c3) emit SPEAKER_xx and sit at 1.0000 by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
