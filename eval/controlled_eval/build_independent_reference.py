#!/usr/bin/env python3
"""Sample the windows for an audio-derived reference, and build the transcription page.

Everything this project has measured so far is agreement with the published OpenCouncil
transcript, because [that transcript IS the benchmark reference](../../docs/reports/2026-08-03-the-reference-problem.md).
The [delivery plan](../../docs/specs/gsoc-delivery-plan.md) makes an independent reference
the first and non-negotiable step, and this builds it.

Two design choices carry the whole thing.

**The meetings are ones this project has never touched.** They come from OpenCouncil's
public API, released after 2026-05-16, cross-checked against the training manifest and the
cached meeting JSON. Nothing here is in the training data or the benchmark.

**The windows are short.** 20 seconds, not two minutes. Transcribing Greek council audio
from scratch runs about ten times realtime, so an hour of audio is a week of somebody's
life and will not happen. 48 windows of 20 seconds is sixteen minutes of audio and roughly
two hours of work, which is a task that gets done, and it still yields ~1,600 words: enough
to estimate an omission rate of a couple of percent with a usable interval.

A frozen third is held back as a locked final test and never shown in any analysis until
the end. The split lives in the key file, not in the page.

Usage:
  SC=~/.cache/oc-public python eval/controlled_eval/build_independent_reference.py
Env: SC OUT_DIR N_WINDOWS WINDOW_SEC PER_MEETING
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-reference-audit"))
N_WINDOWS = int(os.environ.get("N_WINDOWS", "48"))
WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "20"))
PER_MEETING = int(os.environ.get("PER_MEETING", "2"))
SKIP_HEAD = 300.0        # first 5 minutes: roll call and procedure, not representative
SKIP_TAIL = 120.0
MIN_UTT = 4              # the window must actually contain speech
LOCKED_FRAC = 1 / 3      # held back as the final test set


def log(*a):
    print(*a, flush=True)


def h(*parts) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "big")


def utterances(rec) -> list[tuple[float, float, str]]:
    out = []
    for seg in rec.get("transcript") or []:
        for u in seg.get("utterances") or []:
            if u.get("startTimestamp") is None:
                continue
            out.append((float(u["startTimestamp"]), float(u["endTimestamp"]),
                        u.get("text") or ""))
    return sorted(out)


def main():
    recs = sorted((SC / "meetings").glob("*.json"))
    log(f"{len(recs)} fetched public meetings")

    pool = []
    for p in recs:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        m = d.get("meeting") or {}
        city, mid = m.get("cityId"), m.get("id")
        url = m.get("audioUrl")
        utts = utterances(d)
        if not url or len(utts) < 50 or d.get("transcriptHiddenForReview"):
            continue
        end = max(u[1] for u in utts)
        # candidate window starts on a 10 s grid inside the usable span
        cands = []
        t = SKIP_HEAD
        while t + WINDOW_SEC < end - SKIP_TAIL:
            n = sum(1 for s, e, _ in utts if e > t and s < t + WINDOW_SEC)
            if n >= MIN_UTT:
                cands.append(t)
            t += 10.0
        if cands:
            pool.append({"city_id": city, "meeting_id": mid, "audio_url": url,
                         "candidates": cands, "n_people": len(d.get("people") or [])})
    log(f"{len(pool)} meetings with usable windows, "
        f"{len({x['city_id'] for x in pool})} cities")

    # stratify: walk the cities round-robin so no city dominates, deterministic order
    by_city = {}
    for x in pool:
        by_city.setdefault(x["city_id"], []).append(x)
    for c in by_city:
        by_city[c].sort(key=lambda x: h("m", x["meeting_id"]))
    picks, cities = [], sorted(by_city)
    used = {c: 0 for c in cities}
    while len(picks) < N_WINDOWS:
        progressed = False
        for c in cities:
            if len(picks) >= N_WINDOWS:
                break
            if used[c] >= len(by_city[c]):
                continue
            m = by_city[c][used[c]]
            used[c] += 1
            progressed = True
            cs = sorted(m["candidates"], key=lambda t: h("w", m["meeting_id"], t))
            for t in cs[:PER_MEETING]:
                if len(picks) < N_WINDOWS:
                    picks.append({**{k: m[k] for k in
                                     ("city_id", "meeting_id", "audio_url", "n_people")},
                                  "start_sec": t, "dur_sec": WINDOW_SEC})
        if not progressed:
            break
    log(f"{len(picks)} windows from {len({p['meeting_id'] for p in picks})} meetings, "
        f"{len({p['city_id'] for p in picks})} cities")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clips").mkdir(exist_ok=True)
    cache = SC / "mp3"
    cache.mkdir(exist_ok=True)

    rng = random.Random(20260804)
    rng.shuffle(picks)
    n_locked = int(round(len(picks) * LOCKED_FRAC))
    rows = []
    for i, p in enumerate(picks, 1):
        mp3 = cache / f"{p['city_id']}__{p['meeting_id']}.mp3"
        if not mp3.exists():
            r = subprocess.run(["curl", "-sSfL", "-o", str(mp3), p["audio_url"]],
                               capture_output=True)
            if r.returncode != 0:
                log(f"  download failed: {p['city_id']}/{p['meeting_id']}")
                continue
        clip = f"rclip_{i:03d}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{p['start_sec']:.3f}",
                        "-t", f"{p['dur_sec']:.3f}", "-i", str(mp3), "-ac", "1",
                        "-ar", "16000", str(OUT_DIR / "clips" / clip)], check=True)
        rows.append({**p, "clip": clip,
                     "split": "locked" if i <= n_locked else "dev"})
        if i % 10 == 0:
            log(f"  cut {i}/{len(picks)}")

    (OUT_DIR / "KEY_DO_NOT_OPEN_UNTIL_DONE.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    (OUT_DIR / "index.html").write_text(HTML.replace(
        "__MANIFEST__", json.dumps([{"clip": r["clip"]} for r in rows],
                                   ensure_ascii=False)))
    log(f"{len(rows)} clips -> {OUT_DIR}  "
        f"({sum(r['dur_sec'] for r in rows) / 60:.0f} min of audio, "
        f"{sum(1 for r in rows if r['split'] == 'locked')} held back as the locked test)")


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Γράψε ό,τι ακούς</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem 6rem;
      line-height:1.5}
 .clip{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
 .clip.done{border-color:#2a7;background:#f6fdf9}
 .clip.active{border-color:#26c;box-shadow:0 0 0 2px #26c3}
 textarea{width:100%;font:inherit;padding:.5rem;box-sizing:border-box;min-height:5rem}
 audio{width:100%;margin:.4rem 0}
 .ctrl{display:flex;gap:.35rem;align-items:center;flex-wrap:wrap;margin:.3rem 0}
 .ctrl button{font:inherit;padding:.35rem .7rem;border-radius:6px;border:1px solid #888;
        background:#fff;cursor:pointer}
 .ctrl button:hover{background:#eef}
 select{font:inherit;padding:.3rem}
 #bar{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #ddd;
      z-index:50}
 #foot{position:fixed;bottom:0;left:0;right:0;background:#222;color:#eee;padding:8px;
       font:14px sans-serif;z-index:99}
 #foot button{font:inherit;padding:.4rem .8rem;border-radius:6px;border:1px solid #888;
        background:#fff;cursor:pointer}
 .hint{color:#666;font-size:.85rem}
 kbd{background:#eee;border:1px solid #bbb;border-radius:3px;padding:0 .3rem;
     font-size:.85em;color:#333}
</style>
<h1>Γράψε ό,τι ακούς</h1>
<p>Είκοσι δευτερόλεπτα το καθένα. Γράψε <b>όλα όσα ακούγονται</b>, όχι μόνο τον βασικό
ομιλητή. Αν μιλάει και κάποιος άλλος από μακριά ή εκτός μικροφώνου, γράψε και αυτόν.</p>
<p class="hint">Δεν χρειάζονται τελείες, κεφαλαία ή τονισμός. Αν κάτι δεν ακούγεται καθαρά,
βάλε <code>[?]</code> και προχώρα. Αν δεν μιλάει κανείς, άφησέ το κενό. Δεν βλέπεις τι
έγραψε κάποιο σύστημα και δεν πρέπει.</p>
<p class="hint"><b>Ενώ γράφεις:</b> <kbd>Ctrl</kbd>+<kbd>Space</kbd> παίζει και σταματάει ·
<kbd>Ctrl</kbd>+<kbd>←</kbd> πίσω 3 δευτ. · <kbd>Ctrl</kbd>+<kbd>→</kbd> μπροστά 3 δευτ. ·
<kbd>Ctrl</kbd>+<kbd>↓</kbd> πιο αργά. Τα ίδια κάνουν και τα κουμπιά κάτω από κάθε κλιπ.</p>
<div id="bar"><b><span id="n">0</span></b>/<span id="tot">0</span> ·
  <span id="words">0</span> λέξεις</div>
<div id="list"></div>
<div id="foot"><button id="pushbtn">Αποθήκευση στον server</button> <span id="sync">—</span></div>
<script>
const K='refaudit';
let A=JSON.parse(localStorage.getItem(K)||'{}');
let timer=null, current=null;
function count(){
  const vals=Object.values(A).filter(v=>(v||'').trim());
  document.getElementById('n').textContent=vals.length;
  document.getElementById('words').textContent=
    vals.join(' ').split(/\s+/).filter(Boolean).length;
}
function edit(clip,val,el){
  A[clip]=val; localStorage.setItem(K,JSON.stringify(A));
  el.closest('.clip').classList.toggle('done',!!val.trim()); count();
  clearTimeout(timer); timer=setTimeout(push,4000);
}
function audioOf(el){return el.closest('.clip').querySelector('audio');}
function focusClip(el){
  document.querySelectorAll('.clip').forEach(d=>d.classList.remove('active'));
  const d=el.closest('.clip'); d.classList.add('active'); current=d.querySelector('audio');
}
/* Seeking needs the server to answer Range requests. Without that the browser refetches
   from byte zero and playback jumps back to the start, which is exactly the bug this
   page had. */
function nudge(a,dt){ a.currentTime=Math.max(0,Math.min(a.duration||1e9,a.currentTime+dt)); }
function toggle(a){ if(a.paused){a.play();} else {a.pause();} }
function speed(a,v){ a.playbackRate=v; }
(function(m){
  document.getElementById('tot').textContent=m.length;
  document.getElementById('list').innerHTML=m.map((x,i)=>`
   <div class="clip" data-c="${x.clip}">
    <b>${i+1}</b>
    <audio controls preload="metadata" src="clips/${x.clip}"></audio>
    <div class="ctrl">
      <button type="button" onclick="nudge(audioOf(this),-3)">◀◀ 3δ</button>
      <button type="button" onclick="nudge(audioOf(this),-1)">◀ 1δ</button>
      <button type="button" onclick="toggle(audioOf(this))">▶ / ❚❚</button>
      <button type="button" onclick="nudge(audioOf(this),3)">3δ ▶▶</button>
      <select onchange="speed(audioOf(this),parseFloat(this.value))">
        <option value="1">1x</option><option value="0.75">0.75x</option>
        <option value="0.5">0.5x</option>
      </select>
    </div>
    <textarea placeholder="ό,τι ακούς..."
      onfocus="focusClip(this)"
      oninput="edit('${x.clip}',this.value,this)"></textarea>
   </div>`).join('');
  m.forEach(x=>{const v=A[x.clip]; if(!v) return;
    const d=document.querySelector(`[data-c="${x.clip}"]`);
    d.querySelector('textarea').value=v; d.classList.add('done');});
  count();
})(__MANIFEST__);
document.addEventListener('keydown',e=>{
  if(!e.ctrlKey && !e.metaKey) return;
  const a=current || document.querySelector('audio');
  if(!a) return;
  if(e.code==='Space'){ e.preventDefault(); toggle(a); }
  else if(e.key==='ArrowLeft'){ e.preventDefault(); nudge(a,-3); }
  else if(e.key==='ArrowRight'){ e.preventDefault(); nudge(a,3); }
  else if(e.key==='ArrowDown'){ e.preventDefault();
    speed(a, a.playbackRate>0.74?0.75:(a.playbackRate>0.5?0.5:1)); }
});
async function push(){
  try{const r=await fetch('/save',{method:'POST',
    headers:{'Content-Type':'application/json'},body:localStorage.getItem(K)||'{}'});
    document.getElementById('sync').textContent=r.ok?'αποθηκεύτηκε':'σφάλμα';
  }catch(e){document.getElementById('sync').textContent='offline';}
}
document.getElementById('pushbtn').onclick=push;
setInterval(push,60000);
window.addEventListener('beforeunload',()=>{
  navigator.sendBeacon('/save',localStorage.getItem(K)||'{}')});
</script>
"""

if __name__ == "__main__":
    main()
