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

2026-08-09: extended for the second dev build, without touching the sampling logic.
Four things were wrong or missing for a second run, and all four were found by review
before it ran rather than after.

**Eligibility came from cache presence.** The candidate pool was every cached meeting JSON,
but the population the data policy talks about is the `eligible` column of
`data/public-meetings/index.csv`. A file can sit in the cache without being eligible, so
candidates are now intersected with that allowlist (`ELIGIBLE_CSV`) instead of inferred.

**Nothing excluded meetings already spent.** `EXCLUDE_CSV` drops any `(city_id,
meeting_id)` already burned, so a second build cannot silently reuse the first one's
meetings. `INCLUDE_CSV` is the opposite and stronger tool: an explicit allowlist, used when
the permitted set is a policy decision rather than anything derivable from a date. The
2026-08-09 build uses one, because "May first, then the development pool, and never the
sixteen locked meetings" is not a filter, it is a written rule.

**Two picks from one meeting could overlap.** Candidate starts sit on a 10 s grid and the
window is 20 s, so adjacent picks shared half their audio and were scored as two
independent windows. Picks within a meeting now keep `MIN_GAP_SEC` between them.

**The key was written inside the served directory.** `audit_server.py` serves `OUT_DIR`
whole, so `KEY_DO_NOT_OPEN_UNTIL_DONE.json` was one URL away from the person meant to be
blind to it. It now goes to a sibling path outside the served tree.

Dates are filtered on `meeting.dateTime` as a half-open UTC interval, `[DATE_FROM,
DATE_TO)`, and a meeting with a missing or unparseable date is dropped and counted rather
than guessed at.

Usage:
  # the original 48-window build
  SC=~/.cache/oc-public python eval/controlled_eval/build_independent_reference.py
  # the 2026-08-09 dev build: May only, nothing already spent, nothing newly frozen
  LOCKED_FRAC=0 PER_MEETING=1 N_WINDOWS=40 \
  INCLUDE_CSV=data/public-meetings/dev_allowlist.csv \
  PRIORITY_CSV=data/public-meetings/dev_priority_may.csv \
  USED_KEYS=~/oc-reference-audit/KEY_DO_NOT_OPEN_UNTIL_DONE.json \
  OUT_DIR=~/oc-reference-audit-2 CLIP_PREFIX=r2clip \
  python eval/controlled_eval/build_independent_reference.py
Env: SC OUT_DIR N_WINDOWS WINDOW_SEC PER_MEETING DATE_FROM DATE_TO LOCKED_FRAC
     EXCLUDE_CSV INCLUDE_CSV PRIORITY_CSV ELIGIBLE_CSV USED_KEYS CLIP_PREFIX MIN_GAP_SEC
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-reference-audit")).expanduser()
N_WINDOWS = int(os.environ.get("N_WINDOWS", "48"))
WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "20"))
PER_MEETING = int(os.environ.get("PER_MEETING", "2"))
SKIP_HEAD = 300.0        # first 5 minutes: roll call and procedure, not representative
SKIP_TAIL = 120.0
MIN_UTT = 4              # the window must actually contain speech
LOCKED_FRAC = float(os.environ.get("LOCKED_FRAC", 1 / 3))   # held back as the final test
DATE_FROM = os.environ.get("DATE_FROM") or None      # inclusive, UTC calendar date
DATE_TO = os.environ.get("DATE_TO") or None          # exclusive, UTC calendar date
EXCLUDE_CSV = os.environ.get("EXCLUDE_CSV") or None
INCLUDE_CSV = os.environ.get("INCLUDE_CSV") or None   # explicit allowlist, wins over all
PRIORITY_CSV = os.environ.get("PRIORITY_CSV") or None  # drawn before anything else
USED_KEYS = [s for s in os.environ.get("USED_KEYS", "").split(":") if s]  # prior key files
ELIGIBLE_CSV = os.environ.get("ELIGIBLE_CSV", "data/public-meetings/index.csv")
CLIP_PREFIX = os.environ.get("CLIP_PREFIX", "rclip")
MIN_GAP_SEC = float(os.environ.get("MIN_GAP_SEC", WINDOW_SEC))   # no overlapping picks


def log(*a):
    print(*a, flush=True)


def pairs(csv_path) -> set[tuple[str, str]]:
    p = Path(csv_path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open() as f:
        return {(r["city_id"], r["meeting_id"]) for r in csv.DictReader(f)}


def eligible_pairs() -> set[tuple[str, str]] | None:
    """The `eligible` column of the public-meetings index, which is the population the
    data policy is written about. Cache presence is not eligibility."""
    if not ELIGIBLE_CSV:
        return None
    p = Path(ELIGIBLE_CSV)
    if not p.is_absolute():
        p = ROOT / p
    with p.open() as f:
        return {(r["city_id"], r["meeting_id"])
                for r in csv.DictReader(f) if r["eligible"] == "True"}


def used_windows() -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Windows earlier builds already handed to a transcriber, from their key files.

    Window choice is a deterministic hash of (meeting, start), so re-running over a meeting
    that has been sampled before returns *the same twenty seconds*. Without this, a second
    build asks for work that is already done, and the first run of this build did exactly
    that for 31 of 40 windows.
    """
    out: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for path in USED_KEYS:
        for r in json.loads(Path(path).expanduser().read_text()):
            out.setdefault((r["city_id"], r["meeting_id"]), []).append(
                (float(r["start_sec"]), float(r["dur_sec"])))
    return out


def in_date_window(dt_raw) -> bool | None:
    """None when the date is missing or unparseable, so the caller can drop and count it."""
    if DATE_FROM is None and DATE_TO is None:
        return True
    if not dt_raw:
        return None
    try:
        dt = datetime.fromisoformat(str(dt_raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    dt = (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None
          else dt.astimezone(timezone.utc))
    if DATE_FROM and dt < datetime.fromisoformat(DATE_FROM).replace(tzinfo=timezone.utc):
        return False
    if DATE_TO and dt >= datetime.fromisoformat(DATE_TO).replace(tzinfo=timezone.utc):
        return False
    return True


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
    if DATE_FROM and DATE_TO and DATE_FROM >= DATE_TO:
        sys.exit(f"DATE_FROM {DATE_FROM} is not before DATE_TO {DATE_TO}")
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        sys.exit(f"{OUT_DIR} is not empty; refusing to build over an existing package")

    recs = sorted((SC / "meetings").glob("*.json"))
    log(f"{len(recs)} fetched public meetings")
    keep = pairs(INCLUDE_CSV) if INCLUDE_CSV else eligible_pairs()
    drop = pairs(EXCLUDE_CSV) if EXCLUDE_CSV else set()
    if keep is not None:
        log(f"{len(keep)} meetings allowed by {INCLUDE_CSV or ELIGIBLE_CSV}")
    if drop:
        log(f"{len(drop)} meetings excluded by {EXCLUDE_CSV}")

    used = used_windows()
    if used:
        log(f"{sum(len(v) for v in used.values())} windows already spent in "
            f"{len(used)} meetings, from {len(USED_KEYS)} key file(s)")
    pool, n_nodate, n_ineligible, n_excluded, n_reused = [], 0, 0, 0, 0
    for p in recs:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        m = d.get("meeting") or {}
        city, mid = m.get("cityId"), m.get("id")
        url = m.get("audioUrl")
        # Cheap identity filters first: never open the audio of a meeting we may not use.
        if (city, mid) in drop:
            n_excluded += 1
            continue
        if keep is not None and (city, mid) not in keep:
            n_ineligible += 1
            continue
        ok = in_date_window(m.get("dateTime"))
        if ok is None:
            n_nodate += 1
            log(f"  no usable dateTime, dropped: {city}/{mid}")
            continue
        if not ok:
            continue
        utts = utterances(d)
        if not url or len(utts) < 50 or d.get("transcriptHiddenForReview"):
            log(f"  too short or unusable, dropped: {city}/{mid} ({len(utts)} utterances)")
            continue
        end = max(u[1] for u in utts)
        # candidate window starts on a 10 s grid inside the usable span
        spent = used.get((city, mid), [])
        cands = []
        t = SKIP_HEAD
        while t + WINDOW_SEC < end - SKIP_TAIL:
            n = sum(1 for s, e, _ in utts if e > t and s < t + WINDOW_SEC)
            if n >= MIN_UTT and not any(t < s + d and s < t + WINDOW_SEC for s, d in spent):
                cands.append(t)
            t += 10.0
        if spent:
            n_reused += len(spent)
        if cands:
            pool.append({"city_id": city, "meeting_id": mid, "audio_url": url,
                         "candidates": cands, "n_people": len(d.get("people") or [])})
    log(f"{len(pool)} meetings with usable windows, "
        f"{len({x['city_id'] for x in pool})} cities "
        f"(dropped: {n_excluded} excluded, {n_ineligible} ineligible, {n_nodate} no date; "
        f"{n_reused} already-spent windows blocked)")

    # stratify: walk the cities round-robin so no city dominates, deterministic order
    by_city = {}
    for x in pool:
        by_city.setdefault(x["city_id"], []).append(x)
    # "Build from May first, and only then from the development pool" is a written rule,
    # so it belongs in the draw order rather than in whoever runs this remembering it.
    prio = pairs(PRIORITY_CSV) if PRIORITY_CSV else set()
    for c in by_city:
        by_city[c].sort(key=lambda x: (0 if (x["city_id"], x["meeting_id"]) in prio else 1,
                                       h("m", x["meeting_id"])))
    if prio:
        log(f"{sum(1 for x in pool if (x['city_id'], x['meeting_id']) in prio)} of "
            f"{len(prio)} priority meetings are in the usable pool")
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
            # Candidate starts sit on a 10 s grid and the window is 20 s, so taking the
            # first two in hash order could hand out two windows sharing half their audio.
            taken: list[float] = []
            for t in cs:
                if len(taken) >= PER_MEETING or len(picks) >= N_WINDOWS:
                    break
                if any(abs(t - u) < MIN_GAP_SEC for u in taken):
                    continue
                taken.append(t)
                picks.append({**{k: m[k] for k in
                                 ("city_id", "meeting_id", "audio_url", "n_people")},
                              "start_sec": t, "dur_sec": WINDOW_SEC})
            if len(taken) < PER_MEETING:
                log(f"  only {len(taken)}/{PER_MEETING} separated windows in "
                    f"{m['city_id']}/{m['meeting_id']}")
        if not progressed:
            break
    # by (city, meeting): meeting ids like `may20_2026` repeat across cities, and counting
    # them alone under-reports the number of clusters the bootstrap actually has.
    log(f"{len(picks)} windows from "
        f"{len({(p['city_id'], p['meeting_id']) for p in picks})} meetings, "
        f"{len({p['city_id'] for p in picks})} cities")
    if len(picks) < N_WINDOWS:
        log(f"  SHORT: asked for {N_WINDOWS}, the pool yields {len(picks)}")

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
        clip = f"{CLIP_PREFIX}_{i:03d}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{p['start_sec']:.3f}",
                        "-t", f"{p['dur_sec']:.3f}", "-i", str(mp3), "-ac", "1",
                        "-ar", "16000", str(OUT_DIR / "clips" / clip)], check=True)
        rows.append({**p, "clip": clip,
                     "split": "locked" if i <= n_locked else "dev"})
        if i % 10 == 0:
            log(f"  cut {i}/{len(picks)}")

    # Outside OUT_DIR: audit_server.py serves that directory whole, so a key written into
    # it is one URL away from the person who is supposed to be blind to it.
    key = OUT_DIR.parent / f"{OUT_DIR.name}-KEY_DO_NOT_OPEN_UNTIL_DONE.json"
    key.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (OUT_DIR / "index.html").write_text(HTML.replace(
        "__MANIFEST__", json.dumps([{"clip": r["clip"]} for r in rows],
                                   ensure_ascii=False)))
    log(f"{len(rows)} clips -> {OUT_DIR}  "
        f"({sum(r['dur_sec'] for r in rows) / 60:.0f} min of audio, "
        f"{sum(1 for r in rows if r['split'] == 'locked')} held back as the locked test)")
    log(f"key -> {key}")


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
<p class="hint"><b>Πλήκτρα, δουλεύουν ενώ γράφεις:</b>
<kbd>Esc</kbd> παίζει και σταματάει · <kbd>F2</kbd> πίσω 3 δευτ. · <kbd>F4</kbd> μπροστά
3 δευτ. · <kbd>F8</kbd> αλλάζει ταχύτητα. Όταν ξαναρχίζεις μετά από παύση, γυρίζει μόνο
του λίγο πίσω. Δεν πειράζονται τα <kbd>Ctrl</kbd>+<kbd>←</kbd>/<kbd>→</kbd>, που τα
χρειάζεσαι για να κινείσαι μέσα στο κείμενο.</p>
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
function toggle(a){
  if(a.paused){ nudge(a,-0.7); a.play(); }   // back up a little on resume, as pedals do
  else { a.pause(); }
}
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
/* Shortcuts a transcriber can actually use. Deliberately NOT Ctrl+arrows: those are
   word-by-word cursor movement in a textarea and stealing them makes writing worse. Esc
   and the function keys do nothing inside a text box, so they are free. F1/F3/F5/F6/F7
   are taken by the browser (help, find, reload, address bar, caret browsing), F2/F4/F8
   are not. */
document.addEventListener('keydown',e=>{
  const a=current || document.querySelector('audio');
  if(!a) return;
  const k=e.key;
  if(k==='Escape' || (k===' ' && (e.ctrlKey||e.metaKey))){ e.preventDefault(); toggle(a); }
  else if(k==='F2'){ e.preventDefault(); nudge(a,-3); }
  else if(k==='F4'){ e.preventDefault(); nudge(a,3); }
  else if(k==='F8'){ e.preventDefault();
    const next={1:0.75,0.75:0.5,0.5:1}[a.playbackRate]||1;
    speed(a,next); flash(next+'x'); }
});
function flash(msg){
  const el=document.getElementById('sync'); el.textContent=msg;
  setTimeout(()=>{el.textContent='—';},1200);
}
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
