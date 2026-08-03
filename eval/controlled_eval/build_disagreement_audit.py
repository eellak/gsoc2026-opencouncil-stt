#!/usr/bin/env python3
"""Blinded listening package for the 34 windows where the two diarizers disagree.

The first audit asked "how many speakers do you hear", which cannot tell turn-taking
apart from simultaneous speech — 7 of the 20 zero-overlap controls came back with two
speakers and the answer was uninterpretable. This one asks the question that was actually
needed: **do two people speak at the same time**, yes or no, at a marked moment.

Only the disagreement strata are worth human time. Both detectors agreeing that overlap
is present has already been validated at 40/40, and both agreeing it is absent is where
neither can help, since they share the off-mic blind spot.

Nothing in the package says which detector flagged which clip.

Usage:
  SC=~/.cache/oc-overlap HF_TOKEN=hf_... python eval/controlled_eval/build_disagreement_audit.py
Env: SC OUT_DIR CLIP_SEC HF_TOKEN (only needed for the community-1-only clips)
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as B  # noqa: E402

ROOT = Path("/home/harold/opencouncil-fine-tuning")
AUDIO = ROOT / "data/asr/audio"
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-disagree-audit"))
CLIP_SEC = float(os.environ.get("CLIP_SEC", "12"))
MIN_EVENT = 0.3          # shorter than the first audit: median real event is 0.52 s


def log(m):
    print(m, flush=True)


def intervals_from_turns(turns, duration):
    """Spans where >=2 distinct speakers are active, merged per speaker first."""
    per = {}
    for t in turns:
        s, e = max(0.0, float(t["start"])), min(duration, float(t["end"]))
        if e > s:
            per.setdefault(t["speaker"], []).append((s, e))
    events = []
    for spk, iv in per.items():
        iv.sort()
        merged = []
        for s, e in iv:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        for s, e in merged:
            events += [(s, 1), (e, -1)]
    events.sort()
    out, active, prev, start = [], 0, 0.0, None
    for t, d in events:
        if active >= 2 and t > prev and start is None:
            start = prev
        if active + d < 2 and start is not None:
            if t - start >= MIN_EVENT:
                out.append((start, t))
            start = None
        active += d
        prev = t
    return out


def slice_audio(src: Path, start: float, dur: float, dst: Path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}",
                    "-t", f"{dur:.3f}", "-i", str(src), "-ac", "1", "-ar", "16000",
                    str(dst)], check=True)


def community1_intervals(item):
    """Re-diarize one window with community-1; its cached features kept no boundaries."""
    import torch
    from pyannote.audio import Pipeline
    global _PIPE
    if "_PIPE" not in globals():
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN needed for the community-1-only clips")
        _PIPE = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                         token=token)
        _PIPE.to(torch.device(os.environ.get("DEVICE", "cpu")))
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "w.wav"
        slice_audio(item["_audio"], item["_start"], item["_dur"], w)
        raw = w.read_bytes()[44:]
        import numpy as np
        x = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        out = _PIPE({"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": 16000})
    ann = getattr(out, "speaker_diarization", out)
    turns = [{"start": s.start, "end": s.end, "speaker": spk}
             for s, _, spk in ann.itertracks(yield_label=True)]
    return intervals_from_turns(turns, item["_dur"])


def main():
    c1 = json.loads((SC / "overlap_features.json").read_text())["features"]
    p2 = {k: v for k, v in json.loads((SC / "precision2_corpus.json").read_text()).items()
          if "feat" in v}
    report = B.load_report()
    items = {it["item_id"]: it for it in B.common_items(report, B.provider_ids(report))}
    by_id = {it["itemId"]: it for it in report["items"]}

    picks = []
    for iid, it in items.items():
        if iid not in c1 or iid not in p2:
            continue
        a = c1[iid]["overlap_sec"] > 0
        b = p2[iid]["feat"]["overlap_sec"] > 0
        if a == b:
            continue
        p = AUDIO / f"{it['city_id']}__{it['meeting_id']}.mp3"
        if not p.exists():
            continue
        raw = by_id[iid]
        it["_audio"], it["_start"], it["_dur"] = p, raw["startSec"], raw["durationSec"]
        it["_stratum"] = "p2_only" if b else "c1_only"
        picks.append(it)
    log(f"{len(picks)} disagreement windows "
        f"({sum(1 for x in picks if x['_stratum'] == 'p2_only')} precision-2 only, "
        f"{sum(1 for x in picks if x['_stratum'] == 'c1_only')} community-1 only)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clips").mkdir(exist_ok=True)

    rows = []
    for it in picks:
        if it["_stratum"] == "p2_only":
            ivs = intervals_from_turns(p2[it["item_id"]]["turns"], it["_dur"])
        else:
            ivs = community1_intervals(it)
        if not ivs:
            log(f"  {it['item_id']}: no event >= {MIN_EVENT}s, skipped")
            continue
        s, e = max(ivs, key=lambda x: x[1] - x[0])
        mid = (s + e) / 2
        start = max(0.0, min(mid - CLIP_SEC / 2, it["_dur"] - CLIP_SEC))
        rows.append({"item_id": it["item_id"], "city_id": it["city_id"],
                     "meeting_id": it["meeting_id"], "stratum": it["_stratum"],
                     "abs_start": it["_start"] + start, "dur": CLIP_SEC,
                     "audio": str(it["_audio"]),
                     "event_sec_in_clip": [round(s - start, 2), round(e - start, 2)],
                     "event_len": round(e - s, 2)})

    random.Random(20260803).shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["clip"] = f"dclip_{i:03d}.wav"
        slice_audio(Path(r["audio"]), r["abs_start"], r["dur"],
                    OUT_DIR / "clips" / r["clip"])

    # The manifest the page sees carries the marked moment and nothing else.
    public = [{"clip": r["clip"], "mark": r["event_sec_in_clip"]} for r in rows]
    (OUT_DIR / "KEY_DO_NOT_OPEN_UNTIL_DONE.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    (OUT_DIR / "index.html").write_text(
        HTML.replace("__MANIFEST__", json.dumps(public, ensure_ascii=False)))
    log(f"{len(rows)} clips -> {OUT_DIR}")


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Ταυτόχρονη ομιλία;</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem 5rem;
      line-height:1.5}
 .clip{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
 .clip.done{border-color:#2a7;background:#f6fdf9}
 audio{width:100%;margin:.5rem 0}
 button{font:inherit;padding:.5rem 1rem;border-radius:6px;border:1px solid #888;
        background:#fff;cursor:pointer;margin-right:.4rem}
 button.sel{background:#2a7;color:#fff;border-color:#2a7}
 #bar{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #ddd}
 #foot{position:fixed;bottom:0;left:0;right:0;background:#222;color:#eee;padding:8px;
       font:14px sans-serif;z-index:99}
 .hint{color:#666;font-size:.85rem}
 .mark{font-variant-numeric:tabular-nums;color:#a40}
</style>
<h1>Ταυτόχρονη ομιλία;</h1>
<p>Σε κάθε κλιπ υπάρχει ένα <b>σημαδεμένο σημείο</b>. Άκου εκεί και απάντα ένα πράγμα:
<b>μιλάνε δύο άνθρωποι ταυτόχρονα</b>, ή απλώς εναλλάσσονται;</p>
<p class="hint">Ταυτόχρονα σημαίνει ότι για κάποια στιγμή ακούγονται και οι δύο μαζί, έστω
και για μισό δευτερόλεπτο. Αν ο ένας τελειώνει και ο άλλος αρχίζει, αυτό είναι εναλλαγή,
όχι επικάλυψη. Αν δεν είσαι σίγουρος, πάτα «δεν ξέρω» — είναι χρήσιμη απάντηση.</p>
<div id="bar"><b><span id="n">0</span></b>/<span id="tot">0</span> απαντημένα</div>
<div id="list"></div>
<div id="foot"><button id="pushbtn">Αποθήκευση στον server</button>
  <span id="sync">—</span></div>
<script>
const K='dgaudit';
let A=JSON.parse(localStorage.getItem(K)||'{}');
function count(){document.getElementById('n').textContent=Object.keys(A).length;}
function pick(clip,val,el){
  A[clip]=val; localStorage.setItem(K,JSON.stringify(A));
  el.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('sel'));
  el.classList.add('sel');
  el.closest('.clip').classList.add('done'); count(); push();
}
(function(m){
  document.getElementById('tot').textContent=m.length;
  document.getElementById('list').innerHTML=m.map((x,i)=>`
   <div class="clip" data-c="${x.clip}">
    <b>${i+1}.</b> σημάδι στο <span class="mark">${x.mark[0]}s – ${x.mark[1]}s</span>
    <audio controls preload="none" src="clips/${x.clip}"></audio>
    <div>
     <button onclick="pick('${x.clip}','yes',this)">Ναι, ταυτόχρονα</button>
     <button onclick="pick('${x.clip}','no',this)">Όχι, εναλλάξ</button>
     <button onclick="pick('${x.clip}','one',this)">Μόνο ένας μιλάει</button>
     <button onclick="pick('${x.clip}','unsure',this)">Δεν ξέρω</button>
    </div>
   </div>`).join('');
  m.forEach(x=>{const v=A[x.clip]; if(!v) return;
    const d=document.querySelector(`[data-c="${x.clip}"]`);
    d.classList.add('done');
    const map={yes:0,no:1,one:2,unsure:3};
    d.querySelectorAll('button')[map[v]].classList.add('sel');});
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
