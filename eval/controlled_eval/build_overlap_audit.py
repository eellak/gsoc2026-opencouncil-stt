#!/usr/bin/env python3
"""Cut the listening audit that decides whether WER can score an overlap fix at all.

`exp_overlap.py` found that in high-overlap windows Soniox emits five times as many words
that do not match the reference. Two opposite explanations produce that identical number:

  * it is correctly transcribing the interjector, and our human reference does not
    contain them, so a system doing the right thing is scored as wrong; or
  * it is hallucinating into bad audio.

No amount of arithmetic separates those. Someone has to listen.

This builds the package for that: short clips centred on pyannote-detected overlap, mixed
with control clips from windows where pyannote detected none, shuffled and anonymised, plus
a self-contained HTML page to annotate them in.

Blinding is the point. The annotator never sees any ASR output, never sees the reference,
and cannot tell an overlap clip from a control. Without that the audit measures agreement
with the machine rather than what is audible. The mapping lives in a key file that stays
closed until the annotations come back.

Everything is written OUTSIDE the repo. Council audio and its transcription are the PII
category the 2026-07-21 purge removed from git history.

Usage:
  HF_TOKEN=hf_... SC=/path python eval/controlled_eval/build_overlap_audit.py
Env: OUT_DIR (~/oc-overlap-audit) N_OVERLAP (40) N_CONTROL (20) CLIP_SEC (12) SEED (11)
"""
from __future__ import annotations

import collections
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.controlled_eval import bench_data as B                      # noqa: E402
from eval.controlled_eval.exp_overlap import read_wav, slice_audio    # noqa: E402

AUDIO = ROOT / "data/asr/audio"
SC = Path(os.environ.get("SC", "/tmp"))
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-overlap-audit"))
MODEL_ID = "pyannote/speaker-diarization-community-1"
N_OVERLAP = int(os.environ.get("N_OVERLAP", "40"))
N_CONTROL = int(os.environ.get("N_CONTROL", "20"))
CLIP_SEC = float(os.environ.get("CLIP_SEC", "12"))
MIN_EVENT = 0.5           # ignore overlaps shorter than this: not audibly a interruption
SEED = int(os.environ.get("SEED", "11"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def overlap_intervals(diarization, duration):
    """Time spans where two or more distinct speakers are active.

    Merged per speaker label first, for the same reason exp_overlap.py does it: one
    speaker's adjacent segments are not crosstalk.
    """
    per_speaker = collections.defaultdict(list)
    for seg, _, spk in diarization.itertracks(yield_label=True):
        s, e = max(0.0, seg.start), min(duration, seg.end)
        if e > s:
            per_speaker[spk].append((s, e))
    events = []
    for spans in per_speaker.values():
        merged = []
        for s, e in sorted(spans):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        for s, e in merged:
            events += [(s, 1), (e, -1)]
    events.sort()
    out, active, prev, start = [], 0, 0.0, None
    for t, d in events:
        if active >= 2 and start is None:
            start = prev
        if active < 2 and start is not None:
            out.append((start, prev))
            start = None
        active += d
        prev = t
    if start is not None and active >= 2:
        out.append((start, prev))
    return [(s, e) for s, e in out if e - s >= MIN_EVENT]


def main():
    rnd = random.Random(SEED)
    report = B.load_report()
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    by_id = {it["itemId"]: it for it in report["items"]}
    feats = json.loads((SC / "overlap_features.json").read_text())["features"]

    pool = []
    for it in items:
        f = feats.get(it["item_id"])
        p = AUDIO / f"{it['city_id']}__{it['meeting_id']}.mp3"
        if not f or not p.exists():
            continue
        raw = by_id[it["item_id"]]
        it.update(_audio=p, _start=raw["startSec"], _dur=raw["durationSec"], _f=f)
        pool.append(it)
    log(f"{len(pool)} windows with audio and features")

    # Ranked by detected overlap; the audit only needs windows where there is something
    # to listen to, plus controls from windows where pyannote says there is nothing.
    with_ov = sorted((it for it in pool if it["_f"]["overlap_sec"] >= MIN_EVENT),
                     key=lambda it: -it["_f"]["overlap_frac_of_speech"])
    without = [it for it in pool if it["_f"]["overlap_sec"] == 0]
    log(f"{len(with_ov)} windows with detected overlap, {len(without)} with none")

    # Spread across meetings and cities: 40 events from one loud meeting would measure
    # that meeting.
    seen_meetings = collections.Counter()
    picked = []
    for it in with_ov:
        if seen_meetings[it["meeting_id"]] >= 2:
            continue
        picked.append(it)
        seen_meetings[it["meeting_id"]] += 1
        if len(picked) >= N_OVERLAP:
            break
    log(f"selected {len(picked)} overlap windows across "
        f"{len(set(x['meeting_id'] for x in picked))} meetings, "
        f"{len(set(x['city_id'] for x in picked))} cities")

    import torch
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading {MODEL_ID} on {device} (re-running only the selected windows: the "
        f"first pass stored aggregates, not interval boundaries)")
    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipeline.to(torch.device(device))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips_dir = OUT_DIR / "clips"
    clips_dir.mkdir(exist_ok=True)
    tmp = OUT_DIR / "_tmp.wav"

    entries = []
    for i, it in enumerate(picked, 1):
        measured = slice_audio(it["_audio"], it["_start"], it["_dur"], tmp)
        ann = getattr(pipeline(read_wav(tmp)), "speaker_diarization", None)
        ivs = overlap_intervals(ann, measured)
        if not ivs:
            continue
        s, e = max(ivs, key=lambda x: x[1] - x[0])       # the longest event in the window
        mid = (s + e) / 2
        clip_start = max(0.0, min(mid - CLIP_SEC / 2, measured - CLIP_SEC))
        entries.append({"kind": "overlap", "item_id": it["item_id"],
                        "city_id": it["city_id"], "meeting_id": it["meeting_id"],
                        "abs_start": it["_start"] + clip_start, "dur": CLIP_SEC,
                        "audio": str(it["_audio"]),
                        "event_sec_in_clip": [round(s - clip_start, 2),
                                              round(e - clip_start, 2)],
                        "event_len": round(e - s, 2)})
        if i % 5 == 0:
            log(f"  {i}/{len(picked)} overlap windows")

    for it in rnd.sample(without, min(N_CONTROL, len(without))):
        off = rnd.uniform(10, max(11.0, it["_dur"] - CLIP_SEC - 10))
        entries.append({"kind": "control", "item_id": it["item_id"],
                        "city_id": it["city_id"], "meeting_id": it["meeting_id"],
                        "abs_start": it["_start"] + off, "dur": CLIP_SEC,
                        "audio": str(it["_audio"]),
                        "event_sec_in_clip": None, "event_len": None})
    tmp.unlink(missing_ok=True)

    rnd.shuffle(entries)
    key, manifest = [], []
    for n, ent in enumerate(entries, 1):
        name = f"clip_{n:03d}.wav"
        slice_audio(Path(ent["audio"]), ent["abs_start"], ent["dur"], clips_dir / name)
        key.append({"clip": name, **ent})
        manifest.append({"clip": name})       # everything identifying is withheld

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    (OUT_DIR / "KEY_DO_NOT_OPEN_UNTIL_DONE.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2))
    # The clip list is embedded rather than fetched: opening the page from the file
    # system makes fetch() fail on same-origin rules in every current browser, and this
    # has to work by double-clicking the file.
    (OUT_DIR / "annotate.html").write_text(
        HTML.replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False)))
    n_ov = sum(1 for k in key if k["kind"] == "overlap")
    log(f"{len(key)} clips ({n_ov} overlap, {len(key) - n_ov} control) -> {clips_dir}")
    log(f"open {OUT_DIR / 'annotate.html'} in a browser")
    log("the key file stays closed until the annotations come back")


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Ακρόαση επικάλυψης</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;
      line-height:1.5}
 .clip{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
 .clip.done{border-color:#2a7;background:#f6fdf9}
 textarea{width:100%;font:inherit;padding:.4rem;box-sizing:border-box}
 label{display:block;margin:.6rem 0 .2rem;font-weight:600;font-size:.9rem}
 audio{width:100%;margin:.5rem 0}
 button{font:inherit;padding:.5rem 1rem;border-radius:6px;border:1px solid #888;
        background:#fff;cursor:pointer}
 #bar{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #ddd}
 .hint{color:#666;font-size:.85rem}
</style>
<h1>Ακρόαση επικάλυψης</h1>
<p>Άκου κάθε κλιπ και γράψε <b>τι ακούς</b>. Μην ψάχνεις τι έγραψε κάποιο σύστημα, δεν
υπάρχει εδώ και δεν πρέπει να το δεις. Κάποια κλιπ έχουν δύο ομιλητές, κάποια έναν. Δεν
ξέρεις ποιο είναι ποιο, και αυτό είναι σκόπιμο.</p>
<p class="hint">Αν ο δεύτερος ομιλητής δεν ακούγεται καθαρά, γράψε
<code>[ακαταλαβίστικο]</code>. Αν δεν υπάρχει δεύτερος, άφησέ το κενό. Η δουλειά σώζεται
μόνη της στον browser. Μπορείς να σταματήσεις όποτε θες και να συνεχίσεις αργότερα.</p>
<div id="bar"><b><span id="n">0</span></b> συμπληρωμένα ·
  <button onclick="dl()">Κατέβασε τις απαντήσεις</button></div>
<div id="list"></div>
<script>
const K='ovaudit';
let A=JSON.parse(localStorage.getItem(K)||'{}');
function save(){localStorage.setItem(K,JSON.stringify(A));count()}
function count(){
  const n=Object.values(A).filter(v=>(v.main||'').trim()||(v.second||'').trim()||v.n).length;
  document.getElementById('n').textContent=n;
  document.querySelectorAll('.clip').forEach(d=>{
    const v=A[d.dataset.c]||{};
    d.classList.toggle('done',!!((v.main||'').trim()||(v.second||'').trim()||v.n));});
}
function dl(){
  const b=new Blob([JSON.stringify(A,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='overlap_audit_answers.json';a.click();
}
(function(m){
  document.getElementById('list').innerHTML=m.map((x,i)=>`
   <div class="clip" data-c="${x.clip}">
    <b>${i+1}. ${x.clip}</b>
    <audio controls preload="none" src="clips/${x.clip}"></audio>
    <label>Πόσους ομιλητές ακούς;</label>
    <select onchange="A['${x.clip}']=A['${x.clip}']||{};A['${x.clip}'].n=this.value;save()">
      <option value="">-</option><option>1</option><option>2</option><option>3+</option>
    </select>
    <label>Κύριος ομιλητής</label>
    <textarea rows="2" oninput="A['${x.clip}']=A['${x.clip}']||{};A['${x.clip}'].main=this.value;save()"></textarea>
    <label>Δεύτερος ομιλητής (κενό αν δεν υπάρχει)</label>
    <textarea rows="2" oninput="A['${x.clip}']=A['${x.clip}']||{};A['${x.clip}'].second=this.value;save()"></textarea>
   </div>`).join('');
  m.forEach(x=>{const v=A[x.clip];if(!v)return;
    const d=document.querySelector(`[data-c="${x.clip}"]`);
    if(v.n)d.querySelector('select').value=v.n;
    const t=d.querySelectorAll('textarea');
    t[0].value=v.main||'';t[1].value=v.second||'';});
  count();
})(__MANIFEST__);
</script>
"""


if __name__ == "__main__":
    main()
