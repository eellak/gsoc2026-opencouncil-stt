#!/usr/bin/env python3
"""Cut the approval package for the corrections nobody ever listened to.

The train split holds 11.16h of corrections, but only 3.40h of it was read and heard by
a person. The other 7.76h was chosen by a pipeline — filters, an LLM judge, and a partial
audio check — and the benchmark work points at that portion as the weak half of the data.
9,048 candidates passed the judge and were then never audio-checked and never seen by a
human. Each one a reviewer approves moves from algorithm-selected to human-verified, the
same tier as the 3.40h.

The reviewer is not writing corrections here. The correction already exists; the only
question is whether it matches the audio. That is a one-keystroke judgement, so everything
in this package exists to make the keystroke arrive faster: the word-level diff is computed
here rather than in the browser, the clip is cut to the utterance and nothing else, and the
page carries no text field to tab into.

Spans and text come from `data/hf-dataset/rows.parquet` rather than the meeting JSON,
because the row IS the training item — verifying the meeting JSON's version of the span
would verify something the model never sees. `data/eval/chains.parquet` is read only as a
cross-check and to report divergence.

Everything is written OUTSIDE the repo. Council audio and its transcription are the PII
category the 2026-07-21 purge removed from git history, and that purge came after a leak.

Usage:
  python eval/controlled_eval/build_approval_audit.py
Env: OUT_DIR (~/oc-approve-audit) PAD (0.3) SEED (20260805) JOBS (8) LIMIT (0 = all)
"""
from __future__ import annotations

import concurrent.futures as cf
import difflib
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
AUDIO = ROOT / "data/asr/audio"
NB = ROOT / "data/next-batch"
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-approve-audit"))
PAD = float(os.environ.get("PAD", "0.3"))
SEED = int(os.environ.get("SEED", "20260805"))
JOBS = int(os.environ.get("JOBS", "8"))
LIMIT = int(os.environ.get("LIMIT", "0"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def diff_tokens(before: str, after: str):
    """Whitespace-token diff as two [token, changed] arrays.

    Done here and not in the page for two reasons: the reviewer must never wait on a
    diff, and a diff computed once is a diff that cannot render differently on a reload.
    Equality is on the raw token — the pipeline's corrections are frequently punctuation
    or accent only, and normalising those away would hide exactly the edits under review.
    """
    b, a = before.split(), after.split()
    bt = [[t, 1] for t in b]
    at = [[t, 1] for t in a]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, a, autojunk=False).get_opcodes():
        if op != "equal":
            continue
        for k in range(i1, i2):
            bt[k][1] = 0
        for k in range(j1, j2):
            at[k][1] = 0
    return bt, at


def cut(src: Path, start: float, dur: float, dst: Path) -> bool:
    """One utterance as a small mono mp3. `-ss` before `-i` so ffmpeg seeks, not decodes.

    mp3 rather than wav: 8,958 clips is 120 MB encoded against 1 GB raw, and the page
    is served over a tailnet where that difference is the reviewer's waiting time.
    """
    tmp = dst.with_suffix(".part.mp3")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", str(src), "-ac", "1", "-ar", "16000", "-b:a", "48k", str(tmp)],
        capture_output=True)
    # 48 kbps and not 32: the source is already lossy, and a second pass that low starts
    # eating Greek word endings — which is the thing the reviewer is listening for.
    # A truncated clip has to fail rather than land, so size is checked against the
    # 6 kB/s the bitrate implies before the rename makes it visible.
    ok = r.returncode == 0 and tmp.exists() and tmp.stat().st_size >= 0.5 * 6000 * dur
    if not ok:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


def probe(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return -1.0


def main():
    import pandas as pd

    # The requirement that no transcript or audio touches git is enforced here and not
    # only by the default: a leaked transcript is how this project learned the rule.
    if OUT_DIR.resolve() == ROOT or ROOT in OUT_DIR.resolve().parents:
        raise SystemExit(f"OUT_DIR {OUT_DIR} is inside the repo; audio and transcripts "
                         f"must not be written there")

    sel = list(dict.fromkeys(json.loads((NB / "selected_utterance_ids.json").read_text())))
    faithful = set(json.loads((NB / "final_audio/faithful_ids.json").read_text()))
    bad = set(json.loads((NB / "verified_bad_ids.json").read_text()))
    bad |= set(json.loads((NB / "verified_bad_ids_tier2.json").read_text()))
    after_faithful = [u for u in sel if u not in faithful]
    cand = [u for u in after_faithful if u not in bad]
    log(f"{len(sel)} passed the judge; -{len(sel) - len(after_faithful)} already "
        f"audio-confirmed; -{len(after_faithful) - len(cand)} verified bad "
        f"-> {len(cand)} never heard")

    rows = pd.read_parquet(ROOT / "data/hf-dataset/rows.parquet")
    rows = rows.drop_duplicates("utterance_id").set_index("utterance_id")
    have = [u for u in cand if u in rows.index]
    log(f"{len(have)} of them are dataset rows; {len(cand) - len(have)} are not and have "
        f"no cached audio, so they cannot be reviewed here")

    chains = pd.read_parquet(ROOT / "data/eval/chains.parquet")
    chains = chains.drop_duplicates("utterance_id").set_index("utterance_id")

    items, no_audio, bad_pair, bad_span, diverged = [], 0, 0, 0, []
    for uid in have:
        r = rows.loc[uid]
        mp3 = AUDIO / f"{r.city_id}__{r.meeting_id}.mp3"
        if not mp3.exists():
            no_audio += 1
            continue
        before = (r.initial_before_text or "").strip()
        after = (r.final_after_text or "").strip()
        if not before or not after or before == after:
            bad_pair += 1
            continue
        # A handful of rows have a different text in chains.parquet. Which one the
        # reviewer heard would decide what the approval means, so they are dropped
        # rather than resolved by a coin flip.
        if uid in chains.index:
            c = chains.loc[uid]
            if (c.input_raw or "").strip() != before or (c.gold_final or "").strip() != after:
                diverged.append(uid)
                continue
        try:
            s0, e0 = float(r.start), float(r.end)
        except (TypeError, ValueError):
            bad_span += 1
            continue
        if not (e0 > s0) or s0 != s0 or e0 != e0 or e0 - s0 > 120:
            bad_span += 1
            continue
        start = max(0.0, s0 - PAD)
        dur = e0 + PAD - start
        bt, at = diff_tokens(before, after)
        items.append({"id": uid, "city": r.city_id, "meeting": r.meeting_id,
                      "split": r.split, "dur": round(e0 - s0, 3),
                      "b": bt, "a": at,
                      "_mp3": str(mp3), "_start": round(start, 3), "_dur": round(dur, 3)})
    log(f"{len(items)} reviewable items; dropped {no_audio} without local audio, "
        f"{bad_pair} with an empty or no-op edit, {bad_span} with an unusable span, "
        f"{len(diverged)} whose chains.parquet text disagrees with the dataset row")
    if diverged:
        log(f"  divergent ids: {' '.join(diverged)}")

    # Fixed-seed shuffle. A session that stops after 400 items has to still be a sample of
    # the pool; in dataset order the first 400 would be one or two cities.
    random.Random(SEED).shuffle(items)
    if LIMIT:
        items = items[:LIMIT]
        log(f"LIMIT={LIMIT}, keeping the first {len(items)} of the shuffled order")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = OUT_DIR / "clips"
    clips.mkdir(exist_ok=True)

    # Grouped by meeting so the 8 workers stay inside a handful of mp3s at a time; the
    # source files are 100 MB each and the page cache is doing real work here.
    todo = sorted(range(len(items)), key=lambda i: items[i]["_mp3"])
    done = failed = 0

    def work(i):
        it = items[i]
        dst = clips / f"{it['id']}.mp3"
        if dst.exists() and dst.stat().st_size > 0:
            return i, True
        return i, cut(Path(it["_mp3"]), it["_start"], it["_dur"], dst)

    with cf.ThreadPoolExecutor(JOBS) as ex:
        for i, ok in ex.map(work, todo):
            if ok:
                done += 1
            else:
                failed += 1
                items[i] = None
            if (done + failed) % 500 == 0:
                log(f"  {done + failed}/{len(todo)} clips ({failed} failed)")
    items = [it for it in items if it]
    log(f"{done} clips cut, {failed} failed -> {clips}")

    # A random sample is probed rather than every clip: 8,958 ffprobe calls cost more
    # than they prove, and a systematic cut would show up in twenty of them.
    smp = random.Random(SEED).sample(items, min(20, len(items)))
    devs = [probe(clips / f"{it['id']}.mp3") - it["_dur"] for it in smp]
    log(f"clip duration vs requested span, {len(smp)} sampled: "
        f"max deviation {max(map(abs, devs)):.3f}s")

    public = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]
    (OUT_DIR / "items.json").write_text(json.dumps(public, ensure_ascii=False))
    # The build id namespaces the browser's saved answers. Rebuilding with a different
    # item set must not let a stale localStorage blob resume against the new order.
    build = hashlib.sha256("".join(it["id"] for it in items).encode()).hexdigest()[:12]
    total_h = sum(it["dur"] for it in items) / 3600
    meta = {"built": time.strftime("%Y-%m-%d %H:%M"), "build": build, "seed": SEED,
            "pad_sec": PAD, "candidates_never_heard": len(cand),
            "reviewable": len(items), "hours": round(total_h, 3),
            "clip_dur_max_dev_sec": round(max(map(abs, devs)), 3),
            "cities": len(set(it["city"] for it in items)),
            "meetings": len(set((it["city"], it["meeting"]) for it in items))}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    (OUT_DIR / "index.html").write_text(HTML.replace("__BUILD__", build))
    log(json.dumps(meta, ensure_ascii=False))
    log(f"serve with: AUDIT_DIR={OUT_DIR} PORT=8776 "
        f"python eval/controlled_eval/audit_server.py")


HTML = r"""<!doctype html>
<meta charset="utf-8">
<!-- viewport-fit=cover so the fixed control bar can pay for the home-indicator inset
     itself; without it iOS letterboxes the page and the bar sits above the safe area. -->
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Έγκριση διορθώσεων</title>
<style>
 :root{--ok:#1a7f4b;--bad:#b3261e;--uns:#8a6d00;--barh:9rem}
 *{box-sizing:border-box}
 body{font-family:system-ui,sans-serif;margin:0;background:#fafafa;color:#111;
      line-height:1.45;-webkit-text-size-adjust:100%}
 #bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:.5rem 1rem;
      display:flex;gap:1.2rem;align-items:center;font-size:.9rem;flex-wrap:wrap}
 #prog{flex:1;min-width:120px;height:6px;background:#e6e6e6;border-radius:3px}
 #prog>i{display:block;height:100%;background:var(--ok);border-radius:3px;width:0}
 .num{font-variant-numeric:tabular-nums;font-weight:600}
 /* Next-clip readiness, as one dot. Anything with words in it would be read, and reading
    it is time the reviewer does not have. */
 #nxt{width:.55rem;height:.55rem;border-radius:50%;background:#d8d8d8;flex:none}
 #nxt.wait{background:#e0b400}
 #nxt.on{background:var(--ok)}
 /* --barh is measured from the real bar at runtime: the bar is two rows on a phone and
    one on a laptop, and a hardcoded pad either hides the last line of the correction or
    wastes a third of a phone screen. */
 main{max-width:900px;margin:1.5rem auto;padding:0 max(1rem,env(safe-area-inset-right,0px))
      calc(var(--barh) + 1rem) max(1rem,env(safe-area-inset-left,0px))}
 .meta{color:#777;font-size:.8rem;margin-bottom:.6rem;font-variant-numeric:tabular-nums}
 /* anywhere, not break-word: a single long token (a URL, a run-on number) is the one
    thing that can push the page wider than a 320px phone. */
 .line{font-size:1.35rem;padding:.7rem .9rem;border-radius:8px;margin:.4rem 0;
       border:1px solid #e2e2e2;background:#fff;overflow-wrap:anywhere}
 .line .lab{display:block;font-size:.7rem;letter-spacing:.08em;color:#888;
            margin-bottom:.25rem;font-weight:600}
 .old{color:#8a8a8a}
 .old .ch{color:var(--bad);background:#fdeceb;text-decoration:line-through;
          border-radius:3px;padding:0 .15em}
 .new .ch{color:var(--ok);background:#e7f6ee;border-radius:3px;padding:0 .15em;
          font-weight:600}
 .new .same{color:#666}
 /* The whole control bar is one tap surface: touch-action kills the double-tap zoom
    delay, and no child may scroll, so a fling can never start on a button. */
 #keys{position:fixed;bottom:0;left:0;right:0;background:#1b1b1b;
       padding:.35rem max(.35rem,env(safe-area-inset-right,0px))
               calc(.35rem + env(safe-area-inset-bottom,0px))
               max(.35rem,env(safe-area-inset-left,0px));
       display:flex;flex-direction:column;gap:.35rem;touch-action:manipulation}
 #keys .row{display:flex;gap:.35rem}
 #keys button{flex:1;min-height:56px;border:0;border-radius:10px;color:#fff;font:inherit;
       display:flex;flex-direction:column;align-items:center;justify-content:center;
       gap:.1rem;line-height:1.15;touch-action:manipulation;cursor:pointer;
       -webkit-tap-highlight-color:transparent;user-select:none}
 #keys button:active{filter:brightness(1.25)}
 #keys b{font-size:1.05rem;font-weight:700}
 #keys kbd{font:inherit;font-size:.68rem;opacity:.6;letter-spacing:.04em}
 .dec.ok{background:var(--ok)}.dec.bad{background:var(--bad)}.dec.uns{background:var(--uns)}
 .sec{background:#333;color:#ddd;min-height:44px}
 /* Blocked autoplay turns the play button into the loudest thing on the screen, because
    a silent clip with a quiet hint is how a reviewer answers from the text alone. */
 #btnplay.blocked{background:#0b57d0;animation:pulse 1.1s infinite}
 @keyframes pulse{50%{filter:brightness(1.45)}}
 /* hover+fine, not width alone: a landscape phone is wider than 760px and still needs
    the big thumb targets. */
 @media(min-width:760px) and (hover:hover) and (pointer:fine){
  #keys button{min-height:40px}
  #keys b{font-size:1rem}
  #keys kbd{font-size:.72rem;opacity:.75}
 }
 @media(max-width:600px){
  #bar{font-size:.78rem;gap:.7rem;padding:.4rem .7rem}
  main{margin:.9rem auto}
  .line{font-size:1.15rem;padding:.55rem .65rem}
  #ask{font-size:.8rem}
 }
 #flash{position:fixed;inset:0;pointer-events:none;opacity:0;transition:opacity .12s}
 #done{display:none;text-align:center;padding:4rem 1rem;font-size:1.1rem}
 audio{display:none}
 .warn{color:#b3261e;font-weight:600}
 #ask{margin:.4rem 0 1rem;color:#555;font-size:.9rem}
</style>
<div id="bar">
  <span><span class="num" id="n">0</span>/<span id="tot">0</span></span>
  <div id="prog"><i></i></div>
  <span><span class="num" id="rate">–</span> /λεπτό</span>
  <span><span class="num" id="eta">–</span> λεπτά ακόμη</span>
  <span id="sync">—</span>
  <span id="nxt" title="επόμενο κλιπ"></span>
</div>
<main>
  <div id="card">
    <div id="ask">Ερώτηση: <b>αποδίδει η ΔΙΟΡΘΩΣΗ αυτό που ακούγεται;</b> Όχι αν είναι
      απλώς καλύτερη από το ΑΣΡ — αν είναι σωστή.</div>
    <div class="meta" id="meta"></div>
    <div class="line old" id="before"><span class="lab">ΑΣΡ</span><span id="bt"></span></div>
    <div class="line new" id="after"><span class="lab">ΔΙΟΡΘΩΣΗ</span><span id="at"></span></div>
  </div>
  <div id="done">Τέλος. Όλα τα στοιχεία απαντήθηκαν.<br><br>
    <span id="tally"></span></div>
</main>
<audio id="au" playsinline preload="auto"></audio>
<div id="flash"></div>
<!-- Decisions sit in the bottom row, where a thumb on a one-handed phone actually lands.
     The secondary row above holds the two controls that cannot record an answer. -->
<div id="keys">
  <div class="row">
    <button type="button" class="sec" id="btnback"><b>↩ πίσω</b><kbd>Backspace</kbd></button>
    <button type="button" class="sec" id="btnplay"><b id="playlab">▶ ήχος</b><kbd>Space</kbd></button>
  </div>
  <div class="row">
    <button type="button" class="dec ok" data-v="ok"><b>✓ σωστή</b><kbd>J</kbd></button>
    <button type="button" class="dec bad" data-v="bad"><b>✗ λάθος</b><kbd>F</kbd></button>
    <button type="button" class="dec uns" data-v="unsure"><b>? αβέβαιο</b><kbd>K</kbd></button>
  </div>
</div>
<script>
const K='approve_audit___BUILD__';
let IT=[], A=JSON.parse(localStorage.getItem(K)||'{}'), i=0, t_item=0;
const sess=[];                       // decision times this session, for the rate readout
const el=id=>document.getElementById(id);
let blocked=false;                   // true while the browser refuses to play unprompted
let gen=0;                           // bumped per clip, so a stale play() promise is ignored

function paint(node,toks){
  node.textContent='';
  toks.forEach((t,k)=>{
    if(k) node.appendChild(document.createTextNode(' '));
    const s=document.createElement('span');
    s.className=t[1]?'ch':'same';
    s.textContent=t[0];              // transcript text never goes through innerHTML
    node.appendChild(s);
  });
}
function render(){
  const it=IT[i];
  if(!it){el('card').style.display='none';el('done').style.display='block';
    const c={ok:0,bad:0,unsure:0};Object.values(A).forEach(v=>c[v.v]!==undefined&&c[v.v]++);
    el('tally').textContent=`σωστές ${c.ok} · λάθος ${c.bad} · αβέβαιες ${c.unsure}`;
    // Nothing left to hear: stop the audio and give the blobs back rather than hold the
    // window until the tab closes.
    el('au').pause(); armed=false; clearAll(); dot();
    return;}
  el('card').style.display='';el('done').style.display='none';
  el('meta').textContent=`${it.city} · ${it.meeting} · ${it.dur.toFixed(1)}s`
    +(A[it.id]?`  ·  ήδη: ${A[it.id].v}`:'');
  paint(el('bt'),it.b); paint(el('at'),it.a);
  const a=el('au');
  a.pause();                         // otherwise a fast run leaves two clips overlapping
  gen++;
  const g=gen;
  a.src=url(it);
  sweep();                           // the player moved: retired blobs can go now
  play();
  t_item=performance.now();
  // Prefetch waits until this clip can play. On a thin link the two compete for the same
  // few kB/s, and the clip the reviewer is staring at has to win every time.
  armed=false;
  // Reaching a clip that was not prefetched means the reviewer outran the window. Dropping
  // the requests in flight hands the whole link to the clip being waited on: measured at
  // 15 kB/s, that clip went from 3.5 s to 1.4 s. The dropped ones start again afterwards.
  if(!(pf.get(it.id)||{}).u) yieldNow();
  setTimeout(()=>{if(g===gen) arm();},4000);   // a clip that never loads must not freeze it
  schedule();                        // still runs: it aborts what is now behind us
  stats();
}
function net(id){return 'clips/'+encodeURIComponent(id)+'.mp3';}
// A prefetched clip plays from its blob; anything else plays from the server over a plain
// URL, which is what keeps the 206 seeking path in use for every clip we did not reach in
// time. Seeking inside a blob needs no Range at all: the whole file is already here.
function url(it){const e=pf.get(it.id); return (e&&e.s==='done'&&e.u)?e.u:net(it.id);}

// ---- prefetch -------------------------------------------------------------------
// The bytes are held as blobs rather than left to the browser's HTTP cache. Measured on a
// throttled link: the cache does nothing here, because the clips carry no Cache-Control
// and their Last-Modified is minutes old after a build, so heuristic freshness is about
// zero and every clip is revalidated — a round trip we are trying to remove. A blob is
// under our control and cannot be evicted behind our back.
// It does NOT survive a reload: blobs die with the page, and a reload re-warms the window
// in a few seconds. That is the price of not depending on the server's cache headers.
let AHEAD=4, BEHIND=2;               // clips kept warm in front of and behind the reviewer
const PAR=2;                         // in-flight prefetches, so a thin link is not saturated
const CAP=16;                        // hard ceiling on held blobs (~320 kB) whatever happens
const pf=new Map();                  // id -> {s:'load'|'done'|'err'|'gone', c, n, u, at}
let inflight=0, armed=false, wake=null;
let retire=[];                       // blob URLs evicted while still under the player

// Forward, this walks the items the reviewer will actually see: answering skips whatever
// is already decided, so on a resumed session a plain i+1..i+4 window prefetches clips
// that get jumped over — measured as a full 1.7 s wait on the first advance. Backwards is
// index-based, because that is what the back button does.
function nextIds(n){
  const out=[];
  for(let k=i+1;k<IT.length&&out.length<n;k++) if(!A[IT[k].id]) out.push(IT[k].id);
  return out;
}
// next, then previous, then the rest ahead: the two the reviewer can reach with one press
// come first, so Backspace is as warm as J even right after a clip loaded from the server.
function want(){
  const f=nextIds(AHEAD), b=[];
  for(let k=1;k<=BEHIND;k++) if(IT[i-k]) b.push(IT[i-k].id);
  const out=[];
  if(f.length) out.push(f.shift());
  if(b.length) out.push(b.shift());
  return out.concat(f,b);
}
function arm(){if(!armed){armed=true;schedule();}}
function drop(id,e){
  if(e.s==='load'&&e.c) e.c.abort();
  // Never revoke what is playing — that is silence with no way back. It cannot be revoked
  // now and cannot be forgotten either, so it waits for the player to move off it.
  if(e.u){if(e.u===el('au').currentSrc||e.u===el('au').src) retire.push(e.u);
          else URL.revokeObjectURL(e.u);}
  pf.delete(id);
}
function sweep(){                    // revoke retired URLs once the player has moved off
  const a=el('au');
  retire=retire.filter(u=>{if(u===a.currentSrc||u===a.src) return true;
    URL.revokeObjectURL(u); return false;});
}
function clearAll(){for(const [id,e] of pf) drop(id,e); sweep();}
// Deleted rather than marked failed: an abort we caused ourselves must not spend one of
// the attempts a clip gets.
function yieldNow(){
  for(const [id,e] of pf) if(e.s==='load'){if(e.c) e.c.abort(); pf.delete(id);}
}
function later(ms){                  // one pending wake-up, so a dead link re-warms itself
  if(wake) return;
  wake=setTimeout(()=>{wake=null;schedule();},Math.max(500,ms));
}
function schedule(){
  const w=want(), keep=new Set(w); keep.add(IT[i]&&IT[i].id);
  for(const [id,e] of pf) if(!keep.has(id)) drop(id,e);   // Map never outgrows the window
  if(pf.size>CAP) for(const [id,e] of pf){                // belt and braces
    if(pf.size<=CAP) break;
    if(id!==(IT[i]&&IT[i].id)) drop(id,e);
  }
  if(armed) for(const id of w){
    if(inflight>=PAR) break;
    const e=pf.get(id);
    if(e){
      if(e.s==='gone') continue;                // a 404 does not become a clip by waiting
      if(e.s!=='err') continue;                 // done, or already in flight
      // A failure backs off instead of giving up: a phone that loses the link for a minute
      // must warm the window again by itself, and never hammer while it is down. The timer
      // is what makes "by itself" true when the reviewer has stopped pressing keys.
      const d=Math.min(2000*e.n,30000), left=d-(Date.now()-e.at);
      if(left>0){later(left); continue;}
    }
    const c=new AbortController();
    // The record, not the id, is the identity of this attempt. yieldNow() can delete an
    // entry and schedule() can recreate it before the aborted fetch settles; without this
    // the old attempt would hand its blob, or its failure, to the new one.
    const rec={s:'load',c:c,n:(e?e.n:0)+1,u:null,at:0};
    pf.set(id,rec);
    inflight++;
    Promise.resolve().then(()=>fetch(net(id),{signal:c.signal}))
      // 200 only: a partial body would become a truncated blob that plays as a truncated
      // clip, which is the one failure a reviewer cannot see.
      .then(r=>{if(r.status!==200) throw {perm:r.status>=400&&r.status<500&&
                                                r.status!==408&&r.status!==429};
                return r.blob();})
      .then(b=>{if(pf.get(id)!==rec) return;     // evicted mid-flight: let the blob die
        rec.s='done';rec.c=null;rec.u=URL.createObjectURL(b);})
      // A failed prefetch is still just a prefetch: the clip loads normally from the
      // server when the reviewer reaches it. Marked, backed off, never reported.
      .catch(err=>{if(pf.get(id)!==rec) return;
        rec.s=(err&&err.perm)?'gone':'err';rec.c=null;rec.at=Date.now();})
      .then(()=>{inflight--; dot(); schedule();});
  }
  dot();
}
// One dot, no text: the reviewer needs to know the next clip is warm without reading.
function dot(){
  const nx=nextIds(1)[0], e=nx&&pf.get(nx);
  el('nxt').className=!nx?'':(e&&e.s==='done'?'on':(e&&e.s==='load'?'wait':''));
}
// Arming on the element's own readiness, not on a timer, is what keeps prefetch behind
// the current clip on a slow link.
['canplaythrough','playing'].forEach(ev=>el('au').addEventListener(ev,arm));
// A clip still trickling in takes the link back. 'waiting' also fires on every source
// switch, so only a clip coming from the server counts — a blob never waits on the link.
['stalled','waiting'].forEach(ev=>el('au').addEventListener(ev,()=>{
  if(el('au').currentSrc.startsWith('blob:')) return;
  armed=false; yieldNow();
}));
// The reviewer must never be left with a clip that will not play and no way to know. A
// broken blob falls back to the server; a broken server load says so next to the button
// that retries it.
el('au').addEventListener('error',()=>{
  const it=IT[i];
  if(it){
    const e=pf.get(it.id);
    if(e&&e.u&&el('au').currentSrc===e.u){
      drop(it.id,e); el('au').src=net(it.id); start(); arm(); return;
    }
    el('sync').textContent='ο ήχος δεν φόρτωσε — πάτα ▶';el('sync').className='warn';
  }
  arm();
});

// The button label is driven by the element's own events, never by what we just asked it
// to do: on a phone the request and the playback are not the same thing, and a control
// that lies about the state costs more than one that is a beat late.
function playUI(){
  const on=!el('au').paused && !el('au').ended;
  el('playlab').textContent=on?'⏸ παύση':(blocked?'▶ πάτα για ήχο':'▶ ήχος');
  el('btnplay').classList.toggle('blocked',blocked&&!on);
}
function start(){
  const g=gen;                       // which clip this attempt belongs to
  const p=el('au').play();
  if(!p) return;                     // older browsers return nothing from play()
  p.then(()=>{if(g!==gen) return; blocked=false;playUI();
     if(el('sync').dataset.block){el('sync').dataset.block='';
       el('sync').textContent='—';el('sync').className='';}})
   .catch(err=>{
     // Advancing fast replaces the src mid-play, which rejects the previous promise with
     // AbortError. Reporting that as blocked audio would light the play button up on a
     // clip that is playing fine, so only a real permission refusal counts.
     if(g!==gen||(err&&err.name==='AbortError')) return;
     blocked=true;playUI();
     el('sync').dataset.block='1';
     el('sync').textContent='πάτα ▶ για ήχο';el('sync').className='warn';});
}
function play(){el('au').currentTime=0;start();}   // Space and advance mean "from the top"
function toggle(){                   // the button pauses and resumes; only Space restarts
  const a=el('au');
  if(!a.paused&&!a.ended){a.pause();playUI();return;}
  if(a.ended) a.currentTime=0;
  start();
}

function stats(){
  const n=Object.keys(A).length;
  el('n').textContent=n; el('tot').textContent=IT.length;
  el('prog').firstElementChild.style.width=(100*n/Math.max(1,IT.length))+'%';
  if(sess.length>=3){
    const last=sess.slice(-30), ms=last.reduce((a,b)=>a+b,0)/last.length;
    const rpm=60000/ms;
    el('rate').textContent=rpm.toFixed(1);
    el('eta').textContent=Math.round((IT.length-n)/rpm);
  }
}

function answer(v){
  const it=IT[i]; if(!it) return;
  const ms=Math.round(performance.now()-t_item);
  if(!A[it.id]) sess.push(Math.min(ms,60000));     // re-answers would flatter the rate
  A[it.id]={v:v, ms:ms, t:Date.now()};
  localStorage.setItem(K,JSON.stringify(A));
  flash({ok:'#1a7f4b',bad:'#b3261e',unsure:'#8a6d00'}[v]);
  push();
  i++; while(i<IT.length && A[IT[i].id]) i++;      // skip anything already decided
  render();
}
function back(){
  let j=i-1; if(j<0) return;
  i=j; render();
}
function flash(c){const f=el('flash');f.style.background=c;f.style.opacity=.18;
  setTimeout(()=>f.style.opacity=0,110);}

// Plain letters, no modifiers. An earlier audit bound Ctrl+Arrow and took word-jump away
// from the reviewer for the rest of the session.
addEventListener('keydown',e=>{
  if(e.ctrlKey||e.metaKey||e.altKey||e.repeat||e.isComposing) return;
  const k=e.key.toLowerCase();
  if(k==='j'){answer('ok');}
  else if(k==='f'){answer('bad');}
  else if(k==='k'){answer('unsure');}
  else if(e.code==='Space'){play();}
  else if(k==='backspace'){back();}
  else return;
  e.preventDefault();
});

// ---- touch ----------------------------------------------------------------------
// playsInline or iOS takes the clip fullscreen the moment it plays. (preload=auto is on
// the element itself: the clip is ~22 kB and the point is that it is already there.)
el('au').playsInline=true;
['play','pause','ended'].forEach(ev=>el('au').addEventListener(ev,playUI));

// A phone fires a click for a finger that started on a button and then dragged away
// mid-fling, and two for a double tap. Neither is a decision. `click` is the only
// activation event bound here on purpose — adding pointerup or touchend beside it is the
// classic way to record one decision twice. The keyboard path never reaches this code.
// lastTap starts far in the past, not at 0: performance.now() counts from page load, so
// a zero would make the 350 ms lock swallow the first decision of every session.
let lastTap=-1e9, down=null;
function tap(node,fn){
  node.addEventListener('pointerdown',e=>{
    down={x:e.clientX,y:e.clientY,sy:scrollY,id:IT[i]&&IT[i].id};});
  // pointercancel only. A touch tap gets implicit pointer capture and always ends with
  // lostpointercapture, so treating that as a cancellation rejects every tap on a phone.
  node.addEventListener('pointercancel',()=>{down='cancel';});
  node.addEventListener('click',e=>{
    e.preventDefault();
    const p=down; down=null;
    if(p==='cancel') return;                    // the gesture became a scroll or a swipe
    if(!p){                                     // no pointer: Enter on a focused button
      if(e.detail!==0) return;                  // a click with neither is not a real press
      fn(); return;                             // keep the focus a keyboard user is using
    }
    if(e.detail>1) return;                      // second tap of a double tap
    if(Math.hypot(e.clientX-p.x,e.clientY-p.y)>12) return;   // slid: a scroll, not a tap
    if(Math.abs(scrollY-p.sy)>4) return;        // the page moved under the finger
    if(p.id!==(IT[i]&&IT[i].id)) return;        // this press belongs to the previous item
    node.blur();                                // or a later Space would re-press this button
    fn();
  });
}
// One decision per 350 ms, as a backstop only — the pointer checks above are what makes
// a stray tap harmless.
function decide(v){const t=performance.now(); if(t-lastTap<350) return; lastTap=t; answer(v);}
document.querySelectorAll('.dec').forEach(b=>tap(b,()=>decide(b.dataset.v)));
tap(el('btnback'),back);
tap(el('btnplay'),toggle);
// While autoplay is blocked the card itself plays, so a reviewer who taps the text
// instead of the button still gets sound rather than silence.
tap(el('card'),()=>{if(blocked) toggle();});

// The bar is two rows on a phone and one on a laptop; measure it instead of guessing.
function barh(){document.documentElement.style
  .setProperty('--barh',el('keys').offsetHeight+'px');}
addEventListener('resize',barh);
if(window.ResizeObserver) new ResizeObserver(barh).observe(el('keys'));
barh(); playUI();

// One save in flight at a time. Two overlapping POSTs can land out of order, and while
// the server merges rather than replaces, a failed save has to stay dirty and retry
// instead of being forgotten — losing an hour of clicking silently is the worst outcome
// this page has.
let timer=null, dirty=false, busy=false;
function push(){dirty=true; if(timer||busy) return;
  timer=setTimeout(async()=>{timer=null; if(!dirty) return;
    busy=true; const snap=JSON.stringify(A); dirty=false;
    try{const r=await fetch('/save',{method:'POST',
        headers:{'Content-Type':'application/json'},body:snap});
      if(r.ok){el('sync').textContent='αποθηκεύτηκε';el('sync').className='';}
      else{el('sync').textContent='σφάλμα αποθήκευσης';el('sync').className='warn';dirty=true;}
    }catch(err){el('sync').textContent='χωρίς σύνδεση';el('sync').className='warn';dirty=true;}
    busy=false; if(dirty) push();
  },600);}
addEventListener('beforeunload',()=>{
  navigator.sendBeacon('/save',new Blob([JSON.stringify(A)],{type:'application/json'}));});

function mergeAnswers(x,y){          // same rule as the server: newer timestamp wins
  const out=Object.assign({},x);
  for(const k in y){const o=out[k];
    if(o&&typeof o==='object'&&(o.t||0)>(y[k].t||0)) continue;
    out[k]=y[k];}
  return out;}

(async()=>{
  IT=await (await fetch('items.json')).json();
  // A fresh browser profile has nothing locally; a browser that outran the last save has
  // more than the server. Merging both is the only version that loses neither.
  try{const s=await (await fetch('answers.json',{cache:'no-store'})).json();
    A=mergeAnswers(s,A); localStorage.setItem(K,JSON.stringify(A));
  }catch(e){}
  i=0; while(i<IT.length && A[IT[i].id]) i++;
  render(); stats();
})();
</script>
"""


if __name__ == "__main__":
    sys.exit(main())
