#!/usr/bin/env python3
"""Put every long deletion run back on the meeting timeline.

Stage two of the open-question-3 screen (see deletion_runs.py). The benchmark
report gives a window's reference as one block of text with no timing, so the
first stage could only say *where in the token sequence* a run sits. OpenCouncil's
PUBLIC meeting API serves the same transcript as timestamped utterances grouped
into speaker segments, which is what the run needs to be placed in time.

With that, the run can be asked the questions that matter:

  - does it coincide with utterance boundaries, or cut across them?
  - is it one speaker's turn, and does a speaker change happen at its edge?
  - how long is it in seconds, and what is the silence before it?
  - where does it fall relative to a 30 s whisper chunk of the decoded window?
  - was the reference text there written by a human editor rather than by ASR?

Free: the meeting API is public, no GPU, no ASR call. Transcripts are PII and are
cached under ~/.cache/oc-public/postjune-transcripts-2026-08, never written to git.
Only counts leave this script.

    REPORT=<report.json> python3 -m eval.controlled_eval.deletion_run_context
"""
from __future__ import annotations

import collections
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.deletion_runs import (  # noqa: E402
    BASE, INCUMBENT, LONG, OURS, SCRIBE, runs_of, trace,
)
from eval.controlled_eval.scoring import wtoks  # noqa: E402

REPORT = Path(os.environ.get("REPORT", ""))
TRANSCRIPTS = Path.home() / ".cache/oc-public/postjune-transcripts-2026-08"
OUT = ROOT / "eval" / "results_deletion_run_context.json"
CHUNK = 30.0  # whisper's encoder window


def load_utterances(city, meeting):
    """[(start, end, text, segment_id, edited_by_human, uncertain)] in time order."""
    f = TRANSCRIPTS / f"{city}__{meeting}.json"
    if not f.is_file():
        return None
    d = json.load(open(f))
    out = []
    for seg in d.get("transcript", []):
        for u in seg.get("utterances", []):
            out.append((
                u["startTimestamp"], u["endTimestamp"], u.get("text") or "",
                seg["id"], (u.get("lastModifiedBy") is not None), bool(u.get("uncertain")),
            ))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def window_tokens(utts, t0, t1):
    """Reference tokens of the window, each tagged with its utterance.

    Returns [(token, utt_index, tok_index_in_utt, utt_len)] plus the utterance list
    restricted to the window. An utterance counts as in-window when it overlaps the
    window at all, which is how the benchmark builds referenceText.
    """
    sel = [u for u in utts if u[1] > t0 and u[0] < t1]
    toks = []
    for ui, u in enumerate(sel):
        w = wtoks(u[2])
        for k, t in enumerate(w):
            toks.append((t, ui, k, len(w)))
    return toks, sel


def main():
    d = json.load(open(REPORT))
    items = d["items"]

    matched = unmatched = 0
    runs = []      # one record per long deletion run, with timeline context
    tok_ctx = {}   # itemId -> (tokens, sel)

    for it in items:
        utts = load_utterances(it["cityId"], it["meetingId"])
        if utts is None:
            unmatched += 1
            continue
        t0 = it["startSec"]
        t1 = t0 + it["durationSec"]
        toks, sel = window_tokens(utts, t0, t1)
        ref = wtoks(it["referenceText"])
        # the reconstruction must reproduce the benchmark's own reference exactly,
        # otherwise the token indices from stage one do not address the same words
        if [t[0] for t in toks] != ref:
            unmatched += 1
            continue
        matched += 1
        tok_ctx[it["itemId"]] = (toks, sel, t0, t1)

    res = {
        "substrate": {
            "items": len(items),
            "windows_reconstructed_from_utterances": matched,
            "windows_that_did_not_reconstruct_exactly": unmatched,
            "note": "only exactly-reconstructed windows are analysed; a mismatch means the "
                    "benchmark assembled referenceText differently and the indices would not line up",
        }
    }

    def utt_time(toks, sel, k):
        """Time of reference token k, interpolated inside its utterance."""
        _, ui, ki, n = toks[k]
        s, e = sel[ui][0], sel[ui][1]
        return s + (e - s) * ((ki + 0.5) / max(1, n))

    def collect(system):
        out = []
        for it in items:
            ctx = tok_ctx.get(it["itemId"])
            if ctx is None:
                continue
            toks, sel, t0, t1 = ctx
            p = it["perProvider"].get(system)
            if not p or p.get("status") != "ok":
                continue
            o, _ = trace(wtoks(it["referenceText"]), wtoks(p["hypothesisText"]))
            for st, L in runs_of(o):
                if L < LONG:
                    continue
                span = toks[st:st + L]
                uis = sorted({t[1] for t in span})
                segs = {sel[u][3] for u in uis}
                # exact utterance alignment: run begins at an utterance's first token
                # and ends at an utterance's last token
                exact = (span[0][2] == 0) and (span[-1][2] == span[-1][3] - 1)
                covered_whole = sum(
                    1 for u in uis if sum(1 for t in span if t[1] == u) == len(wtoks(sel[u][2]))
                )
                ts = utt_time(toks, sel, st)
                te = utt_time(toks, sel, st + L - 1)
                prev_end = sel[uis[0] - 1][1] if uis[0] > 0 else t0
                gap_before = sel[uis[0]][0] - prev_end
                # speaker change at the run's leading edge, inside the window
                seg_before = sel[uis[0] - 1][3] if uis[0] > 0 else None
                seg_after = sel[uis[-1] + 1][3] if uis[-1] + 1 < len(sel) else None
                out.append({
                    "item": it["itemId"], "city": it["cityId"], "meeting": it["meetingId"],
                    "len": L, "start_tok": st,
                    "utterances_touched": len(uis), "speaker_segments_touched": len(segs),
                    "whole_utterances_covered": covered_whole,
                    "exact_utterance_alignment": exact,
                    "t_start": ts, "t_end": te, "dur_s": max(0.0, te - ts),
                    "gap_before_s": gap_before,
                    "speaker_change_at_start": seg_before is not None and seg_before not in segs,
                    "speaker_change_at_end": seg_after is not None and seg_after not in segs,
                    "is_whole_speaker_segment": len(segs) == 1 and covered_whole == len(uis)
                                                and (seg_before is None or seg_before not in segs)
                                                and (seg_after is None or seg_after not in segs),
                    "human_edited_utterances": sum(1 for u in uis if sel[u][4]),
                    "uncertain_utterances": sum(1 for u in uis if sel[u][5]),
                    "phase_start": (ts - t0) % CHUNK,
                    "crosses_chunk_boundary": int((ts - t0) // CHUNK) != int((te - t0) // CHUNK),
                    "window_dur": t1 - t0,
                })
        return out

    def baseline(system, seed=20260824):
        """Same run lengths placed at random positions in the same windows."""
        rng = random.Random(seed)
        out = []
        for r in collect(system):
            ctx = tok_ctx[r["item"]]
            toks, sel, t0, t1 = ctx
            n = len(toks)
            L = r["len"]
            if n <= L:
                continue
            st = rng.randrange(0, n - L)
            span = toks[st:st + L]
            uis = sorted({t[1] for t in span})
            segs = {sel[u][3] for u in uis}
            exact = (span[0][2] == 0) and (span[-1][2] == span[-1][3] - 1)
            covered_whole = sum(
                1 for u in uis if sum(1 for t in span if t[1] == u) == len(wtoks(sel[u][2]))
            )
            seg_before = sel[uis[0] - 1][3] if uis[0] > 0 else None
            out.append({
                "len": L, "utterances_touched": len(uis),
                "speaker_segments_touched": len(segs),
                "exact_utterance_alignment": exact,
                "whole_utterances_covered": covered_whole,
                "speaker_change_at_start": seg_before is not None and seg_before not in segs,
                "human_edited_utterances": sum(1 for u in uis if sel[u][4]),
            })
        return out

    def summarise(rs, label):
        n = len(rs)
        if not n:
            return {}
        f = lambda k: sum(1 for r in rs if r.get(k)) / n
        return {
            "label": label, "runs": n,
            "tokens": sum(r["len"] for r in rs),
            "share_exact_utterance_alignment": f("exact_utterance_alignment"),
            "share_speaker_change_at_start": f("speaker_change_at_start"),
            "share_is_whole_speaker_segment": f("is_whole_speaker_segment") if "is_whole_speaker_segment" in rs[0] else None,
            "median_utterances_touched": statistics.median(r["utterances_touched"] for r in rs),
            "share_single_utterance": sum(1 for r in rs if r["utterances_touched"] == 1) / n,
            "share_single_speaker_segment": sum(1 for r in rs if r["speaker_segments_touched"] == 1) / n,
            "share_any_human_edited_utterance": sum(1 for r in rs if r["human_edited_utterances"] > 0) / n,
            "median_dur_s": statistics.median(r["dur_s"] for r in rs) if "dur_s" in rs[0] else None,
            "median_gap_before_s": statistics.median(r["gap_before_s"] for r in rs) if "gap_before_s" in rs[0] else None,
            "share_gap_before_over_1s": (sum(1 for r in rs if r["gap_before_s"] > 1.0) / n) if "gap_before_s" in rs[0] else None,
            "share_crossing_chunk_boundary": (sum(1 for r in rs if r["crosses_chunk_boundary"]) / n) if "crosses_chunk_boundary" in rs[0] else None,
            "phase_start_hist_5s": ([sum(1 for r in rs if int(r["phase_start"] // 5) == b) for b in range(6)]
                                    if "phase_start" in rs[0] else None),
        }

    out = {}
    for s in (OURS, SCRIBE, BASE, INCUMBENT):
        rs = collect(s)
        out[s] = {"observed": summarise(rs, "observed"),
                  "random_position_baseline": summarise(baseline(s), "baseline")}
    res["by_system"] = out

    # human-edited baseline over ALL reference tokens, for the reference-fault test
    tot = edited = 0
    for iid, (toks, sel, _, _) in tok_ctx.items():
        for t in toks:
            tot += 1
            edited += 1 if sel[t[1]][4] else 0
    res["reference_corpus"] = {
        "tokens": tot,
        "share_of_reference_tokens_in_human_edited_utterances": edited / tot if tot else None,
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
