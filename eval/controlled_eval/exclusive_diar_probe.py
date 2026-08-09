#!/usr/bin/env python3
"""Phase 0: what does pyannoteAI `exclusive: true` actually return.

Three arm-C synthetic mixtures (known interjector interval by construction), each
submitted twice: baseline `{url, model}` and `{url, model, exclusive: true}`.
Answers, for the preregistration:

  1. the response schema of `exclusiveDiarization`;
  2. whether the regular `diarization` key is byte-identical between the two
     calls (if it is, Phase 1 needs one call per item, not two);
  3. how the known overlap event is resolved — interjector absorbed into the main
     speaker, main speaker's segment cut with an interjector segment inserted, or
     something else;
  4. whether the exclusive timeline fragments the main speaker outside the event.

Raw output goes to ~/.cache/oc-overlap/exclusive_probe.json (never git: it is
diarization only, no text, but the convention is that raw API output stays out).

Usage: python eval/controlled_eval/exclusive_diar_probe.py [n_items]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, api_key, log, run_one  # noqa: E402

MIX = SC / "mixtures"
MANIFEST = SC / "synth_overlap_manifest.json"
OUT = SC / "exclusive_probe.json"
ARM = "C"


def dominant_in(turns, t0: float, t1: float) -> list[tuple[str, float]]:
    """Speaker-time inside [t0, t1], descending."""
    per = collections.Counter()
    for t in turns:
        s, e = max(t0, float(t["start"])), min(t1, float(t["end"]))
        if e > s:
            per[t["speaker"]] += e - s
    return per.most_common()


def main_speaker(turns) -> str | None:
    per = collections.Counter()
    for t in turns:
        per[t["speaker"]] += float(t["end"]) - float(t["start"])
    return per.most_common(1)[0][0] if per else None


def seg_counts(turns) -> dict:
    per = collections.Counter(t["speaker"] for t in turns)
    return dict(per)


def describe(item, base_turns, excl_turns, reg_from_excl):
    ev0 = float(item["event_start_sec"])
    ev1 = ev0 + float(item["event_dur_sec"])
    dur = float(item["window_dur_sec"])
    ms = main_speaker(base_turns)
    out = {
        "item_id": item["item_id"],
        "event": [round(ev0, 2), round(ev1, 2)],
        "window_dur": round(dur, 2),
        "main_speaker_baseline": ms,
        "n_turns": {"baseline": len(base_turns), "exclusive": len(excl_turns),
                    "regular_from_exclusive_call": len(reg_from_excl)},
        "segments_per_speaker": {"baseline": seg_counts(base_turns),
                                 "exclusive": seg_counts(excl_turns)},
        "in_event": {"baseline": [(s, round(v, 3)) for s, v in dominant_in(base_turns, ev0, ev1)],
                     "exclusive": [(s, round(v, 3)) for s, v in dominant_in(excl_turns, ev0, ev1)]},
        "regular_identical_across_calls": base_turns == reg_from_excl,
    }
    # fragmentation of the main speaker OUTSIDE the event (±1 s guard)
    def outside(turns):
        return [t for t in turns
                if float(t["end"]) < ev0 - 1.0 or float(t["start"]) > ev1 + 1.0]
    if ms:
        b = [t for t in outside(base_turns) if t["speaker"] == ms]
        e = [t for t in outside(excl_turns) if t["speaker"] == ms]
        out["main_speaker_segments_outside_event"] = {"baseline": len(b), "exclusive": len(e)}
    # local view: turns touching the event ±2 s
    def near(turns):
        return [{"speaker": t["speaker"], "start": round(float(t["start"]), 3),
                 "end": round(float(t["end"]), 3)}
                for t in turns
                if float(t["end"]) > ev0 - 2.0 and float(t["start"]) < ev1 + 2.0]
    out["near_event"] = {"baseline": near(base_turns), "exclusive": near(excl_turns)}
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    key = api_key()
    man = json.loads(MANIFEST.read_text())
    items = [it for it in man["items"] if (MIX / f"{it['item_id']}__{ARM}.wav").exists()]
    items = items[:n]
    log(f"probing {len(items)} arm-{ARM} mixtures")

    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    for it in items:
        iid = it["item_id"]
        if iid in store and "error" not in store[iid]:
            log(f"  {iid}: cached")
            continue
        path = MIX / f"{iid}__{ARM}.wav"
        try:
            # sequential, one media object per job: two jobs submitted back to
            # back against the same media is exactly the pattern that hung.
            rb = run_one(path, key, f"excl_probe_base_{iid}__{ARM}", exclusive=False)
            re_ = run_one(path, key, f"excl_probe_excl_{iid}__{ARM}", exclusive=True)
        except RuntimeError as e:
            store[iid] = {"error": str(e)[:400]}
            log(f"  {iid}: {e}")
        else:
            store[iid] = {"baseline": rb, "exclusive": re_}
            log(f"  {iid}: {rb.get('status')} / {re_.get('status')}")
        OUT.write_text(json.dumps(store, ensure_ascii=False))

    # ---------------------------------------------------------------- report
    by_id = {it["item_id"]: it for it in man["items"]}
    log("\n=== output keys ===")
    for iid, v in store.items():
        if "error" in v:
            continue
        log(f"{iid}: baseline output keys = {sorted((v['baseline'].get('output') or {}).keys())}")
        log(f"{iid}: exclusive output keys = {sorted((v['exclusive'].get('output') or {}).keys())}")
        ex = (v["exclusive"].get("output") or {}).get("exclusiveDiarization")
        if ex:
            log(f"{iid}: first exclusive entry = {json.dumps(ex[0], ensure_ascii=False)}")
        break

    log("\n=== per item ===")
    summaries = []
    for iid, v in store.items():
        if "error" in v:
            log(f"{iid}: ERROR {v['error'][:120]}")
            continue
        bo = v["baseline"].get("output") or {}
        eo = v["exclusive"].get("output") or {}
        excl = eo.get("exclusiveDiarization")
        if excl is None:
            log(f"{iid}: NO exclusiveDiarization key -- {sorted(eo.keys())}")
            continue
        s = describe(by_id[iid], bo.get("diarization", []), excl, eo.get("diarization", []))
        summaries.append(s)
        log(json.dumps(s, ensure_ascii=False, indent=1))

    (SC / "exclusive_probe_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=1))
    log(f"\n-> {SC / 'exclusive_probe_summary.json'}")


if __name__ == "__main__":
    main()
