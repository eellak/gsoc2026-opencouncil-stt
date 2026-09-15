#!/usr/bin/env python3
"""Run the diarization arms against pyannoteAI precision-2.

Arms, all on the same audio and the same frozen transcript:

==========  ===========================================  ====================
arm         request                                      information used
==========  ===========================================  ====================
a0          whole meeting, no speaker hint               audio only
a1          whole meeting, ``numSpeakers`` = reference   audio + privileged count
a2          whole meeting, ``minSpeakers`` = reference   audio + privileged floor
a3          each topic slice diarized separately         audio + slice rule
a0_local    a0's output cut at the a3 boundaries         none (free, no API)
==========  ===========================================  ====================

``a0_local`` is the control that makes ``a3`` readable. Independent slices give
each slice its own label space, so two people who never share a slice can never
contaminate one cluster — impurity falls for a mechanical reason, not a better
model. Scoring ``a3`` against ``a0_local`` compares like with like; scoring it
against ``a0`` does not.

Costs one audio-hour per meeting per API arm. ``a0_local`` costs nothing.
pyannote deletes job output 24 hours after completion, so every raw response is
persisted here as it arrives.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "controlled_eval"))
import exclusive_diar_api as api  # noqa: E402

CACHE = Path(os.environ.get("OC_CACHE", Path.home() / ".cache/oc-public"))
RUN_DIR = CACHE / "diarization-clustering-2026-09"
API_ARMS = ("a0", "a1", "a2", "a3", "b0", "b3", "b3t")
# b-arms call /identify with the meeting's enrolled voiceprints instead of
# /diarize. Identity then comes from enrollment rather than from how much voice a
# speaker happens to have in the piece, which is what the a-arms could not do.
VOICEPRINT_ARMS = ("b0", "b3", "b3t")
# Arms derived from results already on disk. They submit nothing and cost nothing.
FREE_ARMS = ("a0_local", "c3")
ALL_ARMS = API_ARMS + FREE_ARMS


# --- pure helpers (unit-tested; no network) --------------------------------

def namespace_speakers(segments: list[dict], slice_index: int) -> list[dict]:
    """Prefix speaker ids with their slice so identity cannot leak across slices."""
    return [dict(s, speaker=f"s{slice_index}:{s['speaker']}") for s in segments]


def slice_offsets(slices: list[dict], per_slice: list[list[dict]]) -> list[dict]:
    """Shift each slice's segments into the meeting timebase and namespace them.

    pyannote times are relative to the audio it was handed. Forgetting the shift
    produces a timeline where every slice starts at zero and every downstream
    number is quietly wrong.
    """
    out: list[dict] = []
    for sl, segs in zip(slices, per_slice):
        off = float(sl["start"])
        shifted = [
            {"start": float(s["start"]) + off,
             "end": float(s["end"]) + off,
             "speaker": s["speaker"]}
            for s in segs
        ]
        out.extend(namespace_speakers(shifted, int(sl["index"])))
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def localise(segments: list[dict], slices: list[dict]) -> list[dict]:
    """Cut whole-meeting segments at the slice boundaries, then namespace them.

    A segment spanning a boundary is split into its parts; total duration is
    conserved, so nothing is dropped and nothing is counted twice.
    """
    out: list[dict] = []
    for sl in slices:
        a, b, idx = float(sl["start"]), float(sl["end"]), int(sl["index"])
        clipped = []
        for s in segments:
            lo, hi = max(float(s["start"]), a), min(float(s["end"]), b)
            if hi > lo:
                clipped.append({"start": lo, "end": hi, "speaker": s["speaker"]})
        out.extend(namespace_speakers(clipped, idx))
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def link_by_overlap(local: list[dict], globl: list[dict]) -> list[dict]:
    """Relabel a sliced timeline using a whole-meeting one as the identity map.

    This is the missing linking stage, built from two runs we already paid for:
    the sliced arm supplies the boundaries, the whole-meeting arm supplies one
    identity per voice across the entire recording. Each local segment takes the
    global speaker it overlaps most.

    A local segment overlapping nothing global keeps its own label rather than
    being dropped — silently discarding it would shrink the denominator and
    flatter the arm.
    """
    out = []
    for seg in local:
        # Total overlap per speaker, not the single widest segment: a speaker
        # holding the floor in three short turns must beat one who appears once
        # for slightly longer than any one of them.
        totals: dict[str, float] = {}
        for g in globl:
            ov = min(seg["end"], g["end"]) - max(seg["start"], g["start"])
            if ov > 0:
                totals[g["speaker"]] = totals.get(g["speaker"], 0.0) + ov
        best = max(totals, key=totals.get) if totals else None
        out.append({"start": seg["start"], "end": seg["end"],
                    "speaker": best if best is not None else seg["speaker"]})
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def eval_result(diar: list[dict], excl: list[dict] | None) -> dict:
    """Shape the harness consumes (EvalDiarizeResult)."""
    speakers = sorted({s["speaker"] for s in diar} | {s["speaker"] for s in (excl or [])})
    return {
        "diarization": diar,
        "exclusiveDiarization": excl,
        "speakers": [{"speaker": s, "match": None, "confidence": {}} for s in speakers],
    }


# --- API work --------------------------------------------------------------

def cut_slice(src: Path, dst: Path, start: float, end: float) -> Path:
    """Re-encode the slice. Deliberately not ``-c copy``: mp3 frame-aligned
    copying shifts the actual start by up to a frame, which would desync the
    offset arithmetic the whole sliced arm rests on."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}",
           "-to", f"{end:.3f}", "-i", str(src),
           "-ac", "1", "-ar", "16000", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def ensure_succeeded(result: dict, tag: str) -> dict:
    """Reject any job that did not succeed, fresh or cached.

    Checking only fresh responses would let a failed job persisted on a previous
    attempt be reused silently on the next run, and a partial timeline scores as
    a real one.
    """
    if result.get("status") != "succeeded":
        raise RuntimeError(f"{tag}: job ended {result.get('status')}")
    return result


def diarize_media(path: Path, key: str, tag: str, hints: dict, raw_dir: Path) -> dict:
    """Upload + diarize + wait, persisting the raw response immediately."""
    raw_path = raw_dir / f"{tag}.json"
    if raw_path.exists():
        return ensure_succeeded(json.loads(raw_path.read_text()), tag)

    media = api.upload(path, key, tag)
    body_extra = {k: v for k, v in hints.items() if v is not None}
    job_id = _diarize(media, key, body_extra)
    result = api.wait_job(job_id, key, timeout=1800)
    result["_request"] = {"media": media, "hints": body_extra, "jobId": job_id}
    result["_retrieved_at"] = time.time()
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(result, ensure_ascii=False))
    if result.get("status") != "succeeded":
        raise RuntimeError(f"{tag}: job {job_id} ended {result.get('status')}")
    return result


def _diarize(media: str, key: str, extra: dict) -> str:
    """POST /v1/diarize with exclusive on and whatever speaker hint the arm sets."""
    body = {"url": media, "model": "precision-2", "exclusive": True, **extra}
    r = api.curl(["-X", "POST", f"{api.API}/diarize",
                  "-H", f"Authorization: Bearer {key}",
                  "-H", "Content-Type: application/json",
                  "-d", json.dumps(body)])
    if "jobId" not in r:
        raise RuntimeError(f"no jobId: {r}")
    return r["jobId"]


def identification_timeline(output: dict, slice_index: int | None,
                            known: set[str]) -> list[dict]:
    """Turn an /identify response into a timeline keyed by real identity.

    A segment counts as identified only when its label is one of the voiceprints
    we sent (``known``). Those labels are personIds and are STABLE across calls,
    which is the whole point: a person recurring in ten slices keeps one label
    instead of becoming ten voices, so no namespacing is applied to them.

    **pyannote does not return null for unmatched speech.** It puts the
    diarization label (`SPEAKER_00`) in the same ``speaker`` field, so trusting
    that field alone silently treats `SPEAKER_00` in slice 1 and slice 2 as the
    same person — measured here: chania returned 13 distinct labels for 2
    voiceprints sent. Anything not in ``known`` is therefore namespaced per
    slice, because two slices' anonymous speakers are unrelated.
    """
    out = []
    for e in output.get("identification") or []:
        who = e.get("speaker") or e.get("match")
        if not who or who not in known:
            anon = who or e.get("diarizationSpeaker") or "UNKNOWN"
            who = f"unmatched:{anon}" if slice_index is None else f"unmatched:s{slice_index}:{anon}"
        out.append({"start": float(e["start"]), "end": float(e["end"]), "speaker": who})
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def identify_media(path: Path, key: str, tag: str, voiceprints: list[dict],
                   raw_dir: Path, threshold: float | None = None) -> dict:
    """POST /identify with the meeting's voiceprints, persisting the raw response.

    ``threshold`` is ``matching.threshold``. Left at the API default of 0 the call
    **cannot abstain**: every second of speech is handed an enrolled label however
    poor the match, which is why sliced arms misname unenrolled speakers.
    """
    raw_path = raw_dir / f"{tag}.json"
    if raw_path.exists():
        return ensure_succeeded(json.loads(raw_path.read_text()), tag)

    media = api.upload(path, key, tag)
    matching: dict = {"exclusive": True}
    if threshold is not None:
        matching["threshold"] = threshold
    body = {
        "url": media,
        "voiceprints": [{"label": v["personId"], "voiceprint": v["voiceprint"]}
                        for v in voiceprints],
        "exclusive": True,
        # one voiceprint may not be claimed by two speakers at once
        "matching": matching,
    }
    r = api.curl(["-X", "POST", f"{api.API}/identify",
                  "-H", f"Authorization: Bearer {key}",
                  "-H", "Content-Type: application/json",
                  "-d", json.dumps(body)])
    if "jobId" not in r:
        raise RuntimeError(f"{tag}: no jobId: {r}")
    result = api.wait_job(r["jobId"], key, timeout=1800)
    result["_request"] = {"media": media, "jobId": r["jobId"],
                          "voiceprints": len(voiceprints)}
    result["_retrieved_at"] = time.time()
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(result, ensure_ascii=False))
    return ensure_succeeded(result, tag)


def arm_hints(arm: str, named: int) -> dict:
    """Exactly one hint per arm. A0 must send none — asserted by the tests."""
    if arm == "a0" or arm == "a3":
        return {}
    if arm == "a1":
        return {"numSpeakers": named}
    if arm == "a2":
        return {"minSpeakers": named}
    raise ValueError(arm)


def run_meeting(rec: dict, arm: str, key: str, out_root: Path, dry: bool,
                threshold: float = 50.0) -> dict | None:
    city, mid = rec["city"], rec["meeting"]
    mdir = out_root / f"{city}__{mid}"
    audio = Path(rec["audio_path"])
    slices = json.loads((mdir / "slices.json").read_text())
    named = rec["named_speakers"]
    # The threshold is part of the cache identity: without it, a run at 60 would
    # silently return the file written at 50.
    dest = mdir / (f"diarization_b3t{threshold:g}.json" if arm == "b3t"
                   else f"diarization_{arm}.json")

    if dest.exists():
        return json.loads(dest.read_text())

    if arm == "c3":
        # "Two diarizations": slice boundaries from a3, identity from a0.
        # Free — both arms are already on disk, no API call.
        need = [mdir / "diarization_a3.json", mdir / "diarization_a0.json"]
        if not all(p.exists() for p in need):
            print(f"  {city}/{mid}: c3 needs both a3 and a0 — skipping")
            return None
        a3 = json.loads(need[0].read_text())
        a0 = json.loads(need[1].read_text())
        g = a0.get("exclusiveDiarization") or a0["diarization"]
        l = a3.get("exclusiveDiarization") or a3["diarization"]
        linked = link_by_overlap(l, g)
        res = eval_result(linked, linked)
        dest.write_text(json.dumps(res, ensure_ascii=False))
        return res

    if arm == "a0_local":
        base = mdir / "diarization_a0.json"
        if not base.exists():
            print(f"  {city}/{mid}: a0_local needs a0 first — skipping")
            return None
        a0 = json.loads(base.read_text())
        res = eval_result(localise(a0["diarization"], slices),
                          localise(a0.get("exclusiveDiarization") or [], slices))
        dest.write_text(json.dumps(res, ensure_ascii=False))
        return res

    if arm in VOICEPRINT_ARMS:
        vp_path = (out_root / "voiceprints" / f"{city}__{mid}.json")
        if not vp_path.exists():
            print(f"  {city}/{mid}: no voiceprints on disk — skipping")
            return None
        vps = json.loads(vp_path.read_text())
        # Only these labels mean "a known person"; see identification_timeline.
        known_labels = {v["personId"] for v in vps}

        if arm == "b0":
            if dry:
                print(f"  {city}/{mid}: b0 would submit 1 identify job "
                      f"({rec['duration_s']/3600:.2f} h, {len(vps)} voiceprints)")
                return None
            r = identify_media(audio, key, f"{city}__{mid}__b0", vps, mdir / "raw")
            tl = identification_timeline(r.get("output") or {}, None, known_labels)
            res = eval_result(tl, tl)
            dest.write_text(json.dumps(res, ensure_ascii=False))
            return res

        # b3t: b3 with a non-zero matching.threshold, so the call can abstain.
        # Astra's cheapest-next-measurement. Only the threshold differs from b3.
        if arm == "b3t":
            if rec["slicing_is_null"]:
                print(f"  {city}/{mid}: one slice — b3t not run")
                return None
            if dry:
                print(f"  {city}/{mid}: b3t would submit {len(slices)} identify jobs "
                      f"at threshold={threshold}")
                return None
            merged: list[dict] = []
            for sl in slices:
                piece = cut_slice(audio, mdir / "slices" / f"s{sl['index']}.wav",
                                  sl["start"], sl["end"])
                r = identify_media(piece, key,
                                   f"{city}__{mid}__b3t{threshold:g}_s{sl['index']}",
                                   vps, mdir / "raw", threshold=threshold)
                off = float(sl["start"])
                for s in identification_timeline(r.get("output") or {},
                                                 int(sl["index"]), known_labels):
                    merged.append({"start": s["start"] + off, "end": s["end"] + off,
                                   "speaker": s["speaker"]})
            merged.sort(key=lambda s: (s["start"], s["end"]))
            res = eval_result(merged, merged)
            dest.write_text(json.dumps(res, ensure_ascii=False))
            return res

        # b3: same voiceprints, one identify call per slice. Identities are stable
        # across calls, so slices need no relinking — that is the whole point.
        if rec["slicing_is_null"]:
            print(f"  {city}/{mid}: one slice — b3 is identical to b0, not run")
            return None
        if dry:
            print(f"  {city}/{mid}: b3 would submit {len(slices)} identify jobs "
                  f"({rec['duration_s']/3600:.2f} h, {len(vps)} voiceprints each)")
            return None
        merged: list[dict] = []
        for sl in slices:
            piece = cut_slice(audio, mdir / "slices" / f"s{sl['index']}.wav",
                              sl["start"], sl["end"])
            r = identify_media(piece, key, f"{city}__{mid}__b3_s{sl['index']}",
                               vps, mdir / "raw")
            off = float(sl["start"])
            for s in identification_timeline(r.get("output") or {}, int(sl["index"]),
                                             known_labels):
                merged.append({"start": s["start"] + off, "end": s["end"] + off,
                               "speaker": s["speaker"]})
        merged.sort(key=lambda s: (s["start"], s["end"]))
        res = eval_result(merged, merged)
        dest.write_text(json.dumps(res, ensure_ascii=False))
        return res

    if arm == "a3":
        if rec["slicing_is_null"]:
            print(f"  {city}/{mid}: one slice — a3 is identical to a0, not run")
            return None
        if dry:
            print(f"  {city}/{mid}: a3 would submit {len(slices)} jobs "
                  f"({rec['duration_s']/3600:.2f} h)")
            return None
        per_slice = []
        for sl in slices:
            piece = cut_slice(audio, mdir / "slices" / f"s{sl['index']}.wav",
                              sl["start"], sl["end"])
            r = diarize_media(piece, key, f"{city}__{mid}__a3_s{sl['index']}",
                              {}, mdir / "raw")
            per_slice.append((r.get("output") or {}).get("exclusiveDiarization") or [])
        regular = []
        for sl in slices:
            raw = json.loads((mdir / "raw" / f"{city}__{mid}__a3_s{sl['index']}.json").read_text())
            regular.append((raw.get("output") or {}).get("diarization") or [])
        res = eval_result(slice_offsets(slices, regular), slice_offsets(slices, per_slice))
        dest.write_text(json.dumps(res, ensure_ascii=False))
        return res

    hints = arm_hints(arm, named)
    if dry:
        print(f"  {city}/{mid}: {arm} would submit 1 job "
              f"({rec['duration_s']/3600:.2f} h) hints={hints}")
        return None
    r = diarize_media(audio, key, f"{city}__{mid}__{arm}", hints, mdir / "raw")
    out = r.get("output") or {}
    res = eval_result(out.get("diarization") or [], out.get("exclusiveDiarization"))
    dest.write_text(json.dumps(res, ensure_ascii=False))
    return res


def voices(res: dict) -> int:
    return len({s["speaker"] for s in (res.get("exclusiveDiarization") or res["diarization"])})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["a0"],
                    help=f"billed: {', '.join(API_ARMS)} | free: {', '.join(FREE_ARMS)}")
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="matching.threshold for the b3t arm. 50 is what pyannote's "
                         "own documentation uses in both of its examples; it is "
                         "declared here before the arm was run, not chosen after.")
    ap.add_argument("--only", help="restrict to one city/meeting")
    a = ap.parse_args()

    cohort = json.loads((a.run_dir / "cohort.json").read_text())
    key = os.environ.get("PYANNOTE_API_KEY")
    needs_key = not a.dry_run and any(arm in API_ARMS for arm in a.arms)
    if not key and needs_key:
        raise SystemExit("PYANNOTE_API_KEY not set")

    billed = sum(r["duration_s"] for r in cohort) / 3600
    for arm in a.arms:
        cost = 0.0 if arm in FREE_ARMS else billed
        print(f"\n=== arm {arm} ({cost:.2f} audio-hours) ===")
        for rec in cohort:
            if a.only and f'{rec["city"]}/{rec["meeting"]}' != a.only:
                continue
            res = run_meeting(rec, arm, key, a.run_dir, a.dry_run, a.threshold)
            if res is not None:
                print(f'  {rec["city"]}/{rec["meeting"]:16} voices={voices(res):3} '
                      f'segments={len(res["diarization"]):5}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
