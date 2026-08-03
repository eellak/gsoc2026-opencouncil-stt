#!/usr/bin/env python3
"""Is the reference missing words? A yes/no audit, not a transcription task.

The [reference problem](../../docs/reports/2026-08-03-the-reference-problem.md) showed the
benchmark's reference is the published transcript, and the listening audit showed it omits
audible speech. What is unknown is the RATE, and estimating it by transcribing hours of
audio is not going to happen.

So this asks a cheaper question. Where Soniox and Scribe INDEPENDENTLY emit the same word
that the reference does not contain, that word is a candidate omission — two systems
inventing the same Greek word in the same place is far less likely than both hearing it.
Each candidate becomes a short clip and one question: was that word said?

Sampling is by candidate word, one per window at most, so no single busy window dominates
and the estimate can be weighted back to the corpus.

Usage:
  SC=~/.cache/oc-overlap python eval/controlled_eval/build_omission_audit.py
Env: SC OUT_DIR N_ITEMS CLIP_SEC
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as B  # noqa: E402
import scoring as S  # noqa: E402

ROOT = Path("/home/harold/opencouncil-fine-tuning")
AUDIO = ROOT / "data/asr/audio"
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-omission-audit"))
N_ITEMS = int(os.environ.get("N_ITEMS", "120"))
CLIP_SEC = float(os.environ.get("CLIP_SEC", "10"))
A, C = "soniox", "scribe-v2-clean"
MIN_LEN = 4          # skip short function words: too common to localise or judge


def log(m):
    print(m, flush=True)


def candidates(it):
    """Words both systems emit more often than the reference holds them."""
    ref = collections.Counter(S.wtoks(it["ref"]))
    a = collections.Counter(S.wtoks(it["hyp"][A]))
    c = collections.Counter(S.wtoks(it["hyp"][C]))
    out = []
    for w in set(a) & set(c):
        if len(w) < MIN_LEN:
            continue
        if min(a[w], c[w]) - ref[w] > 0:
            out.append(w)
    return sorted(out)


def locate(word, it, dur):
    """Rough time of the word, from its position in Soniox's word sequence.

    There are no word timestamps in the stored hypotheses, so this assumes an even
    speaking rate across the window. It only has to put the listener within a few seconds;
    the clip is 10 s and the word is named, so a small error is recoverable by ear.
    """
    toks = S.wtoks(it["hyp"][A])
    if word not in toks:
        return dur / 2
    i = toks.index(word)
    return (i + 0.5) / max(1, len(toks)) * dur


def main():
    report = B.load_report()
    items = B.common_items(report, B.provider_ids(report))
    by_id = {x["itemId"]: x for x in report["items"]}

    pool = []
    for it in items:
        src = AUDIO / f"{it['city_id']}__{it['meeting_id']}.mp3"
        if not src.exists():
            continue
        cands = candidates(it)
        if not cands:
            continue
        raw = by_id[it["item_id"]]
        # one candidate per window, chosen by a stable hash so the pick is reproducible
        h = int(hashlib.sha256(it["item_id"].encode()).hexdigest()[:8], 16)
        w = cands[h % len(cands)]
        pool.append({"item_id": it["item_id"], "city_id": it["city_id"],
                     "meeting_id": it["meeting_id"], "word": w,
                     "n_candidates_in_window": len(cands),
                     "audio": str(src), "abs_start": raw["startSec"],
                     "dur": raw["durationSec"],
                     "at": locate(w, it, raw["durationSec"])})
    log(f"{len(pool)} windows carry at least one candidate omission")

    rng = random.Random(20260803)
    rng.shuffle(pool)
    picks = pool[:N_ITEMS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clips").mkdir(exist_ok=True)
    rows = []
    for i, p in enumerate(picks, 1):
        start = max(0.0, min(p["at"] - CLIP_SEC / 2, p["dur"] - CLIP_SEC))
        clip = f"oclip_{i:03d}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss",
                        f"{p['abs_start'] + start:.3f}", "-t", f"{CLIP_SEC:.3f}",
                        "-i", p["audio"], "-ac", "1", "-ar", "16000",
                        str(OUT_DIR / "clips" / clip)], check=True)
        rows.append({**p, "clip": clip, "clip_start_in_window": start})

    public = [{"clip": r["clip"], "word": r["word"]} for r in rows]
    (OUT_DIR / "KEY_DO_NOT_OPEN_UNTIL_DONE.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    (OUT_DIR / "index.html").write_text(
        HTML.replace("__MANIFEST__", json.dumps(public, ensure_ascii=False)))
    log(f"{len(rows)} clips -> {OUT_DIR}")
    log(f"cities: {len({r['city_id'] for r in rows})}, "
        f"meetings: {len({r['meeting_id'] for r in rows})}")


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Ειπώθηκε αυτή η λέξη;</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem 5rem;
      line-height:1.5}
 .clip{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
 .clip.done{border-color:#2a7;background:#f6fdf9}
 .w{font-size:1.6rem;font-weight:700;color:#a40;display:block;margin:.3rem 0}
 audio{width:100%;margin:.5rem 0}
 button{font:inherit;padding:.5rem 1rem;border-radius:6px;border:1px solid #888;
        background:#fff;cursor:pointer;margin-right:.4rem}
 button.sel{background:#2a7;color:#fff;border-color:#2a7}
 #bar{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #ddd}
 #foot{position:fixed;bottom:0;left:0;right:0;background:#222;color:#eee;padding:8px;
       font:14px sans-serif;z-index:99}
 .hint{color:#666;font-size:.85rem}
</style>
<h1>Ειπώθηκε αυτή η λέξη;</h1>
<p>Κάθε κλιπ είναι 10 δευτερόλεπτα και από πάνω μια λέξη. Δύο ανεξάρτητα συστήματα
ισχυρίζονται ότι ακούγεται εκεί· η αναφορά μας δεν την έχει. <b>Ακούγεται;</b></p>
<p class="hint">Ο εντοπισμός είναι κατά προσέγγιση, οπότε η λέξη μπορεί να είναι
οπουδήποτε μέσα στο κλιπ. Αν την ακούς έστω μία φορά, «ναι». Αν δεν είσαι σίγουρος, «δεν
ξέρω» — μετράει κανονικά και είναι προτιμότερο από μαντεψιά. Κάθε απάντηση σώζεται αμέσως.</p>
<div id="bar"><b><span id="n">0</span></b>/<span id="tot">0</span> απαντημένα</div>
<div id="list"></div>
<div id="foot"><button id="pushbtn">Αποθήκευση στον server</button> <span id="sync">—</span></div>
<script>
const K='omaudit';
let A=JSON.parse(localStorage.getItem(K)||'{}');
function count(){document.getElementById('n').textContent=Object.keys(A).length;}
function pick(clip,val,el){
  A[clip]=val; localStorage.setItem(K,JSON.stringify(A));
  el.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('sel'));
  el.classList.add('sel'); el.closest('.clip').classList.add('done'); count(); push();
}
(function(m){
  document.getElementById('tot').textContent=m.length;
  document.getElementById('list').innerHTML=m.map((x,i)=>`
   <div class="clip" data-c="${x.clip}">
    <b>${i+1}.</b><span class="w">${x.word}</span>
    <audio controls preload="none" src="clips/${x.clip}"></audio>
    <div>
     <button onclick="pick('${x.clip}','yes',this)">Ναι, ακούγεται</button>
     <button onclick="pick('${x.clip}','no',this)">Όχι</button>
     <button onclick="pick('${x.clip}','unsure',this)">Δεν ξέρω</button>
    </div>
   </div>`).join('');
  m.forEach(x=>{const v=A[x.clip]; if(!v) return;
    const d=document.querySelector(`[data-c="${x.clip}"]`); d.classList.add('done');
    d.querySelectorAll('button')[{yes:0,no:1,unsure:2}[v]].classList.add('sel');});
  count();
})(__MANIFEST__);
async function push(){
  try{const r=await fetch('/save',{method:'POST',
    headers:{'Content-Type':'application/json'},body:localStorage.getItem(K)||'{}'});
    document.getElementById('sync').textContent=r.ok?'αποθηκεύτηκε':'σφάλμα';
  }catch(e){document.getElementById('sync').textContent='offline';}
}
document.getElementById('pushbtn').onclick=push;
window.addEventListener('beforeunload',()=>{
  navigator.sendBeacon('/save',localStorage.getItem(K)||'{}')});
</script>
"""

if __name__ == "__main__":
    main()
