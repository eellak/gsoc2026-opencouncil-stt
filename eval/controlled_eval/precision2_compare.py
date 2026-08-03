#!/usr/bin/env python3
"""pyannoteAI precision-2 against community-1, judged by the human listening audit.

The blinded audit gave us 60 clips with a human speaker count: 40 centred on an overlap
event community-1 detected, 20 from windows where it detected none. community-1 scored
40/40 on the first set, so precision on overlap events is not where a better model can
help. The open question is RECALL — whether community-1 misses overlap that is really
there — and the 60 clips are exactly where that can be measured, because a human already
listened to them.

precision-2 is a cloud model, so the clips leave the machine. That is the same exposure
the benchmark already accepts for Scribe, Soniox and Gladia, and it is council audio, not
the corrections corpus.

Usage:
  python eval/controlled_eval/precision2_compare.py
Env: PYANNOTE_API_KEY (or ~/.cache/oc-overlap/pyannote_api_key)  CLIPS  OUT  MODEL
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
CLIPS = Path(os.environ.get("CLIPS", Path.home() / "oc-overlap-audit/clips"))
KEY_FILE = Path.home() / "oc-overlap-audit/KEY_DO_NOT_OPEN_UNTIL_DONE.json"
ANSWERS = SC / "overlap_audit_answers.json"
OUT = Path(os.environ.get("OUT", SC / "precision2_clips.json"))
MODEL = os.environ.get("MODEL", "precision-2")
API = "https://api.pyannote.ai/v1"


def api_key() -> str:
    k = os.environ.get("PYANNOTE_API_KEY")
    if k:
        return k
    return (SC / "pyannote_api_key").read_text().strip()


def log(*a):
    print(*a, flush=True)


def curl(args: list[str]) -> dict:
    """curl, not urllib: the same Cloudflare-fronted-API lesson as RunPod."""
    r = subprocess.run(["curl", "-sS", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed: {r.stderr[:300]}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"non-JSON response: {r.stdout[:300]}")


def upload(path: Path, key: str, name: str) -> str:
    media = f"media://oc-overlap/{name}"
    r = curl(["-X", "POST", f"{API}/media/input",
              "-H", f"Authorization: Bearer {key}",
              "-H", "Content-Type: application/json",
              "-d", json.dumps({"url": media})])
    put = r.get("url")
    if not put:
        raise RuntimeError(f"no presigned url: {r}")
    p = subprocess.run(["curl", "-sS", "-X", "PUT", put, "--data-binary", f"@{path}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"upload failed: {p.stderr[:300]}")
    return media


def diarize(media: str, key: str) -> str:
    body = {"url": media, "model": MODEL, "confidence": True}
    r = curl(["-X", "POST", f"{API}/diarize",
              "-H", f"Authorization: Bearer {key}",
              "-H", "Content-Type: application/json",
              "-d", json.dumps(body)])
    if "jobId" not in r:
        raise RuntimeError(f"no jobId: {r}")
    return r["jobId"]


def wait_job(job_id: str, key: str, timeout: float = 300) -> dict:
    t0 = time.time()
    delay = 2.0
    while time.time() - t0 < timeout:
        r = curl([f"{API}/jobs/{job_id}", "-H", f"Authorization: Bearer {key}"])
        st = r.get("status")
        if st in ("succeeded", "failed", "canceled"):
            return r
        time.sleep(delay)
        delay = min(delay * 1.4, 15)
    return {"status": "timeout", "jobId": job_id}


# ------------------------------------------------------------------ overlap features
def overlap_of(turns, duration: float) -> dict:
    """Same sweep as exp_overlap: merge per speaker FIRST, then count ≥2 active.

    Without the per-speaker merge, two adjacent turns of one speaker would be scored as
    that speaker overlapping themselves.
    """
    per = collections.defaultdict(list)
    for t in turns:
        s, e = max(0.0, float(t["start"])), min(duration, float(t["end"]))
        if e > s:
            per[t["speaker"]].append((s, e))
    events = []
    speech_by_spk = {}
    for spk, iv in per.items():
        iv.sort()
        merged = []
        for s, e in iv:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        speech_by_spk[spk] = sum(e - s for s, e in merged)
        for s, e in merged:
            events += [(s, 1), (e, -1)]
    events.sort()
    speech = overlap = 0.0
    active, prev = 0, 0.0
    for t, d in events:
        if active >= 1:
            speech += t - prev
        if active >= 2:
            overlap += t - prev
        active += d
        prev = t
    return {"n_speakers": len(per), "speech_sec": speech, "overlap_sec": overlap,
            "overlap_frac_of_speech": overlap / speech if speech else 0.0}


def main():
    key = api_key()
    manifest = {k["clip"]: k for k in json.loads(KEY_FILE.read_text())}
    answers = json.loads(ANSWERS.read_text())
    done = json.loads(OUT.read_text()) if OUT.exists() else {}

    clips = sorted(manifest)
    log(f"{len(clips)} clips, model={MODEL}, {len(done)} already done")

    for i, clip in enumerate(clips, 1):
        if clip in done:
            continue
        p = CLIPS / clip
        try:
            media = upload(p, key, clip)
            job = diarize(media, key)
            r = wait_job(job, key)
        except RuntimeError as e:
            done[clip] = {"error": str(e)[:300]}
            log(f"  {clip}: {e}")
        else:
            if r.get("status") != "succeeded":
                done[clip] = {"error": r.get("status"), "raw": r}
                log(f"  {clip}: {r.get('status')} {str(r)[:200]}")
            else:
                out = r.get("output", {})
                turns = out.get("diarization", [])
                done[clip] = {"turns": turns,
                              "feat": overlap_of(turns, manifest[clip]["dur"])}
        OUT.write_text(json.dumps(done, ensure_ascii=False))
        if i % 10 == 0 or i == len(clips):
            log(f"  {i}/{len(clips)}")

    # ------------------------------------------------------------------ comparison
    ok = {c: v for c, v in done.items() if "feat" in v}
    log(f"\n{len(ok)}/{len(clips)} succeeded")
    rows = []
    for c, v in ok.items():
        k, a = manifest[c], answers.get(c, {})
        rows.append({"clip": c, "kind": k["kind"], "human_n": a.get("n"),
                     "p2_speakers": v["feat"]["n_speakers"],
                     "p2_overlap_sec": round(v["feat"]["overlap_sec"], 2)})

    def agg(kind):
        r = [x for x in rows if x["kind"] == kind]
        det = sum(1 for x in r if x["p2_overlap_sec"] > 0.1)
        return r, det

    ov, ov_det = agg("overlap")
    ct, ct_det = agg("control")
    log(f"precision-2 finds overlap in {ov_det}/{len(ov)} of the clips community-1 flagged")
    log(f"precision-2 finds overlap in {ct_det}/{len(ct)} of the zero-overlap controls")

    # human count vs model count
    m = collections.Counter()
    for x in rows:
        m[(x["human_n"], min(x["p2_speakers"], 3))] += 1
    log("\nhuman count x precision-2 speaker count:")
    for hn in ("1", "2", "3+"):
        line = "  ".join(f"p2={p}:{m.get((hn, p), 0)}" for p in (1, 2, 3))
        log(f"  human {hn:2s}  {line}")

    res = {"model": MODEL, "n_clips": len(ok), "rows": rows,
           "overlap_clips_confirmed": [ov_det, len(ov)],
           "control_clips_with_overlap": [ct_det, len(ct)],
           "note": ("Human labels are speaker COUNTS in a 12s clip, not overlap "
                    "judgements: two speakers taking turns count as two. So the control "
                    "column bounds how often a zero-overlap window holds a second voice, "
                    "not how often community-1 missed overlap.")}
    dst = Path("/home/harold/opencouncil-fine-tuning/eval/controlled_eval/results_precision2.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"-> {dst}")


if __name__ == "__main__":
    main()
