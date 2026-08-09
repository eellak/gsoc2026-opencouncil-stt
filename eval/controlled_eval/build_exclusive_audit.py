#!/usr/bin/env python3
"""Blinded listening package for the regular-vs-exclusive attribution disagreements.

Each item: the utterance clip (±2 s) and the two candidate speakers, each represented
by reference anchor clips cut from that speaker's own segments elsewhere in the same
window — "SPEAKER_02" means nothing to a listener, two seconds of that voice does.

Blinding: the two candidates are shown as A and B in an order drawn per item from
`random.Random(20260807)`, and nothing on the page or in the served directory says
which candidate came from which variant. The mapping lives in
KEY_DO_NOT_OPEN_UNTIL_DONE.json, written outside the served tree.

Usage: python eval/controlled_eval/build_exclusive_audit.py
Env: OUT_DIR (default ~/oc-exclusive-audit)
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, log  # noqa: E402

REPLAY = SC / "exclusive_phase2_replay.json"
DIAR = SC / "exclusive_phase2_diar.json"
WAV = SC / "winwav"
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-exclusive-audit"))
SEED = 20260807
PAD = 2.0          # context around the utterance
ANCHOR_MAX = 4.0   # each anchor clip
ANCHOR_MIN = 1.2   # a segment shorter than this is not a usable voice sample
N_ANCHORS = 2
FAR_ENOUGH = 3.0   # anchors must sit this far outside the utterance


def slice_audio(src: Path, start: float, dur: float, dst: Path):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}",
                    "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
                   check=True)


def anchors_for(segs, speaker, u0, u1):
    """Longest segments of `speaker` that sit clear of the utterance."""
    cand = [s for s in segs
            if s["speaker"] == speaker
            and (s["end"] < u0 - FAR_ENOUGH or s["start"] > u1 + FAR_ENOUGH)
            and s["end"] - s["start"] >= ANCHOR_MIN]
    cand.sort(key=lambda s: -(s["end"] - s["start"]))
    return cand[:N_ANCHORS]


def main():
    rep = json.loads(REPLAY.read_text())
    diar = json.loads(DIAR.read_text())
    rng = random.Random(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clips").mkdir(exist_ok=True)

    key_rows, public = [], []
    dropped = 0
    for n, it in enumerate(rep["adjudication_items"], 1):
        wid = it["window_id"]
        src = WAV / f"{wid}.wav"
        segs = diar[wid]["diarization"]
        u0, u1 = it["start"], it["end"]
        cands = [("regular", it["speaker_regular"]), ("exclusive", it["speaker_exclusive"])]
        anch = {v: anchors_for(segs, spk, u0, u1) for v, spk in cands}
        if any(len(a) == 0 for a in anch.values()):
            dropped += 1
            continue

        rng.shuffle(cands)  # A/B order, per item
        stem = f"x{n:03d}"
        main_clip = f"{stem}_main.wav"
        slice_audio(src, u0 - PAD, (u1 - u0) + 2 * PAD, OUT_DIR / "clips" / main_clip)

        pub = {"id": stem, "main": main_clip, "cands": []}
        for side, (variant, spk) in zip("AB", cands):
            clips = []
            for j, s in enumerate(anch[variant], 1):
                name = f"{stem}_{side}{j}.wav"
                dur = min(ANCHOR_MAX, s["end"] - s["start"])
                slice_audio(src, s["start"], dur, OUT_DIR / "clips" / name)
                clips.append(name)
            pub["cands"].append({"side": side, "clips": clips})
        public.append(pub)
        key_rows.append({"id": stem, "window_id": wid, "utt_index": it["utt_index"],
                         "start": u0, "end": u1, "text": it["text"],
                         "A": cands[0][0], "B": cands[1][0],
                         "speaker_A": cands[0][1], "speaker_B": cands[1][1]})

    (OUT_DIR.parent / "oc-exclusive-audit-KEY_DO_NOT_OPEN_UNTIL_DONE.json").write_text(
        json.dumps(key_rows, ensure_ascii=False, indent=1))
    (OUT_DIR / "index.html").write_text(
        HTML.replace("__MANIFEST__", json.dumps(public, ensure_ascii=False)))
    log(f"{len(public)} items -> {OUT_DIR}  ({dropped} dropped: no usable anchor)")
    log(f"key -> {OUT_DIR.parent / 'oc-exclusive-audit-KEY_DO_NOT_OPEN_UNTIL_DONE.json'}")


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Ποιος μιλάει;</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem 5rem;
      line-height:1.5}
 .item{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1.2rem 0}
 .item.done{border-color:#2a7;background:#f6fdf9}
 audio{width:100%;margin:.3rem 0}
 .cands{display:flex;gap:1rem;margin:.6rem 0}
 .cand{flex:1;border:1px solid #ddd;border-radius:6px;padding:.5rem}
 .cand b{display:block;margin-bottom:.3rem}
 button{font:inherit;padding:.45rem .9rem;border-radius:6px;border:1px solid #888;
        background:#fff;cursor:pointer;margin:.2rem .3rem .2rem 0}
 button.sel{background:#2a7;color:#fff;border-color:#2a7}
 #bar{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #ddd}
 #foot{position:fixed;bottom:0;left:0;right:0;background:#222;color:#eee;padding:8px;
       font:14px sans-serif;z-index:99}
 .hint{color:#666;font-size:.85rem}
</style>
<h1>Ποιος μιλάει;</h1>
<p>Σε κάθε στοιχείο ακούς ένα <b>κύριο κλιπ</b> και δύο υποψήφιες φωνές, <b>Α</b> και
<b>Β</b>, με δείγματα από αλλού μέσα στην ίδια συνεδρίαση. Ερώτηση: <b>ποια από τις δύο
φωνές μιλάει στο κύριο κλιπ</b>;</p>
<p class="hint">Το κύριο κλιπ έχει 2 δευτερόλεπτα περιθώριο πριν και μετά — κρίνε τη φωνή
που λέει το κυρίως κομμάτι. Αν Α και Β σου ακούγονται ο <b>ίδιος άνθρωπος</b>, πάτα «ίδια
φωνή». Αν μιλάει κάποιος <b>τρίτος</b>, πάτα «κανένας από τους δύο». Αν δεν ξεχωρίζεις,
πάτα «δεν ξέρω» — είναι χρήσιμη απάντηση, μην μαντεύεις.</p>
<div id="bar"><b><span id="n">0</span></b>/<span id="tot">0</span> απαντημένα</div>
<div id="list"></div>
<div id="foot"><button id="pushbtn">Αποθήκευση στον server</button>
  <span id="sync">—</span></div>
<script>
const K='exclaudit';
let A=JSON.parse(localStorage.getItem(K)||'{}');
const OPTS=[['a','Η Α μιλάει'],['b','Η Β μιλάει'],['same','Ίδια φωνή'],
            ['neither','Κανένας από τους δύο'],['unsure','Δεν ξέρω']];
function count(){document.getElementById('n').textContent=Object.keys(A).length;}
function pick(id,val,el){
  A[id]=val; localStorage.setItem(K,JSON.stringify(A));
  el.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('sel'));
  el.classList.add('sel'); el.closest('.item').classList.add('done'); count(); push();
}
(function(m){
  document.getElementById('tot').textContent=m.length;
  document.getElementById('list').innerHTML=m.map((x,i)=>`
   <div class="item" data-c="${x.id}">
    <b>${i+1}.</b> κύριο κλιπ
    <audio controls preload="none" src="clips/${x.main}"></audio>
    <div class="cands">${x.cands.map(c=>`
      <div class="cand"><b>Φωνή ${c.side}</b>
        ${c.clips.map(f=>`<audio controls preload="none" src="clips/${f}"></audio>`).join('')}
      </div>`).join('')}</div>
    <div>${OPTS.map(o=>`<button onclick="pick('${x.id}','${o[0]}',this)">${o[1]}</button>`).join('')}</div>
   </div>`).join('');
  m.forEach(x=>{const v=A[x.id]; if(!v) return;
    const d=document.querySelector(`[data-c="${x.id}"]`);
    d.classList.add('done');
    const idx=OPTS.findIndex(o=>o[0]===v);
    if(idx>=0) d.querySelectorAll('.item > div:last-child button')[idx].classList.add('sel');});
  count();
})(__MANIFEST__);
async function push(){
  try{
    const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:localStorage.getItem(K)||'{}'});
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
