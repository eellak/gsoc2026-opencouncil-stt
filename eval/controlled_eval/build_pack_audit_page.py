"""Cut real packs from meeting audio and serve them for a fast human sanity check.

This is the cheap test of the overlap-free selection: the reviewer hears exactly what
the model would be trained on, with the target text beside it, and answers three yes/no
questions per pack. No aggregate is computed here; scoring lives in score_pack_audit.

Audio and text stay under ~/.cache, never in git.

    .venv-eval/bin/python -m eval.controlled_eval.build_pack_audit_page --packs 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

from eval.controlled_eval.boundary_snap import snap
from eval.controlled_eval.overlap_clip_census import (
    build_packs, census, contiguous_packs, load_jobs, pad_bounds, transcript_rows,
    pack_speech_sec,
)

DEFAULT_DIAR = Path.home() / ".cache/oc-public/training-diarization-2026-08"
DEFAULT_TRANSCRIPTS = Path.home() / ".cache/oc-public/training-transcripts-2026-08"
DEFAULT_OUT = Path.home() / ".cache/oc-public/pack-audit-2026-08"
SEPARATOR_SEC = 0.4
FADE_MS = 20
SEED = 20260819


def utterance_text(transcripts: Path, city: str, meeting: str) -> dict[str, str]:
    payload = json.loads((transcripts / f"{city}__{meeting}.json").read_text())
    return {u["id"]: (u.get("text") or "")
            for segment in payload["transcript"] for u in segment.get("utterances", [])}


def decode(audio: Path, cache: Path):
    """Whole meeting as 16 kHz mono float32, cached: boundary search needs raw samples."""
    import numpy as np
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        run = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
                              "-ar", "16000", "-ac", "1", "-f", "f32le", str(cache)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"decode failed: {run.stderr[-300:]}")
    return np.memmap(cache, dtype="float32", mode="r")


def snapped_bounds(samples, turns, row, audio_end, search: float = 0.25):
    """Acoustic cut inside the space diarization says belongs to nobody else."""
    lo, hi = pad_bounds(turns, row["start"], row["end"], audio_end, max_pad=search)
    start = snap(samples, 16000, row["start"], lo, min(row["start"] + 0.08, hi), "start")
    end = snap(samples, 16000, row["end"], max(row["end"] - 0.08, lo), hi, "end")
    return start, max(end, start + 0.2)


def cut_pack(audio: Path, pack: list[dict], out_wav: Path) -> None:
    """Concatenate the pack's utterances with faded silence between them."""
    parts = []
    for i, row in enumerate(pack):
        if i:
            parts.append(f"aevalsrc=0:d={SEPARATOR_SEC}:s=16000[s{i}]")
        parts.append(f"[0:a]atrim=start={row['start']}:end={row['end']},"
                     f"asetpts=PTS-STARTPTS,afade=t=in:d={FADE_MS/1000},"
                     f"afade=t=out:st={max(0.0, row['end']-row['start']-FADE_MS/1000)}:"
                     f"d={FADE_MS/1000}[a{i}]")
    labels = "".join(f"[a{i}]" if i == 0 else f"[s{i}][a{i}]" for i in range(len(pack)))
    graph = ";".join(parts) + f";{labels}concat=n={2*len(pack)-1}:v=0:a=1[out]"
    run = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
                          "-filter_complex", graph, "-map", "[out]", "-ar", "16000",
                          "-ac", "1", str(out_wav)], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {run.stderr[-300:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_DIAR)
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--packs", type=int, default=12)
    parser.add_argument("--mode", choices=["spliced", "contiguous", "both"], default="both")
    args = parser.parse_args()

    jobs = load_jobs(args.root)
    rows, _ = census(transcript_rows(args.transcripts, jobs), jobs)
    turns = {url: job["turns"] for url, job in jobs.items()}
    ends = {url: job["audio_end"] for url, job in jobs.items()}
    rng = random.Random(SEED)
    served = args.out / "served"
    (served / "clips").mkdir(parents=True, exist_ok=True)

    wanted = args.packs if args.mode != "both" else args.packs // 2
    plan = []
    if args.mode in ("spliced", "both"):
        spliced = [p for p in build_packs(rows) if pack_speech_sec(p) >= 17.6]
        plan += [("spliced", p) for p in rng.sample(spliced, min(wanted, len(spliced)))]
    if args.mode in ("contiguous", "both"):
        whole = contiguous_packs(rows, turns, ends)
        plan += [("contiguous", p) for p in rng.sample(whole, min(wanted, len(whole)))]
    rng.shuffle(plan)

    items, hidden = [], []
    for mode, pack in plan:
        first = pack[0] if mode == "spliced" else pack
        city, meeting, audio_url = first["city_id"], first["meeting_id"], first["audio_url"]
        audio = args.root / "audio" / (hashlib.sha256(audio_url.encode()).hexdigest()[:16] + ".mp3")
        samples = decode(audio, args.out / "pcm" / (audio.stem + ".f32"))
        ids = [r["utterance_id"] for r in pack] if mode == "spliced" else pack["utterance_ids"]
        item_id = hashlib.sha256(f"{mode}|{ids[0]}".encode()).hexdigest()[:16]
        wav = served / "clips" / f"{item_id}.wav"
        if not wav.is_file():
            if mode == "spliced":
                cuts = []
                for row in pack:
                    start, end = snapped_bounds(samples, turns[audio_url], row, ends[audio_url])
                    cuts.append({**row, "start": start, "end": end})
                cut_pack(audio, cuts, wav)
            else:
                cut_pack(audio, [pack], wav)
        texts = utterance_text(args.transcripts, city, meeting)
        speech = (pack_speech_sec(pack) if mode == "spliced" else pack["speech_sec"])
        items.append({
            "review_id": item_id,
            "audio": f"clips/{wav.name}",
            "text": " ".join(texts.get(i, "") for i in ids).strip(),
            "speech_sec": speech,
            "utterances": len(ids),
        })
        hidden.append({"review_id": item_id, "mode": mode, "city_id": city,
                       "meeting_id": meeting, "utterance_ids": ids})

    (args.out / "hidden-key.json").write_text(json.dumps(hidden, ensure_ascii=False, indent=1))
    (served / "index.html").write_text(render(items), encoding="utf-8")
    print(json.dumps({"served": str(served), "packs_served": len(items),
                      "by_mode": {mode: sum(h["mode"] == mode for h in hidden)
                                  for mode in sorted({h["mode"] for h in hidden})}},
                     ensure_ascii=False))


def render(items: list[dict]) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return r"""<!doctype html>
<html lang="el"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Έλεγχος packs</title><style>
body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f5f5f2;color:#181818}
header{position:sticky;top:0;z-index:2;padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
main{max-width:900px;margin:auto;padding:16px}.card{background:#fff;border:1px solid #ddd;border-radius:9px;padding:16px;margin:14px 0}
.muted{color:#666;font-size:13px}.saved{color:#287a3c}.q{margin-top:12px}
button{font:inherit;padding:6px 11px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button.on{background:#245b9e;color:#fff;border-color:#245b9e}audio{width:100%}.text{font-size:17px;margin:10px 0;white-space:pre-wrap}
</style></head><body>
<header><b>Έλεγχος packs εκπαίδευσης</b><span id="progress" class="muted"></span><span id="saved" class="saved"></span></header>
<main><p>Κάθε pack είναι ό,τι θα δει το μοντέλο σε ένα παράθυρο: ενωμένα κομμάτια του ίδιου
ομιλητή με 0,4 s σιωπή ανάμεσα, και από κάτω το κείμενο-στόχος. Άκου και απάντησε τρία
πράγματα. Δεν χρειάζεται να γράψεις τίποτα.</p><div id="items"></div></main>
<script>
const ITEMS=__ITEMS__;const KEY='oc-pack-audit-v1';const V=JSON.parse(localStorage.getItem(KEY)||'{}');let timer=null;
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function done(it){const v=V[it.review_id]||{};return v.one_speaker&&v.text_matches&&v.boundaries_ok}
function progress(){document.getElementById('progress').textContent=ITEMS.filter(done).length+'/'+ITEMS.length+' απαντημένα'}
async function save(){localStorage.setItem(KEY,JSON.stringify(V));try{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(V)});document.getElementById('saved').textContent=r.ok?'αποθηκεύτηκε':'μόνο τοπικά'}catch(e){document.getElementById('saved').textContent='μόνο τοπικά'}}
function mark(id,field,value){V[id]=V[id]||{};V[id][field]=value;V[id].t=Date.now();draw();document.getElementById('saved').textContent='αποθήκευση…';clearTimeout(timer);timer=setTimeout(save,300)}
function buttons(it,field,choices){const v=(V[it.review_id]||{})[field];return '<div>'+choices.map(c=>'<button class="'+(v===c[0]?'on':'')+'" data-id="'+it.review_id+'" data-f="'+field+'" data-v="'+c[0]+'">'+c[1]+'</button> ').join('')+'</div>'}
function draw(){const root=document.getElementById('items');
root.innerHTML=ITEMS.map((it,n)=>'<section class="card"><div class="muted">Pack '+(n+1)+' · '+it.speech_sec.toFixed(1)+'s ομιλίας · '+it.utterances+' κομμάτια</div>'+
'<audio controls preload="none" src="'+it.audio+'"></audio><div class="text">'+esc(it.text)+'</div>'+
'<div class="q"><b>Μιλάει μόνο ένας;</b>'+buttons(it,'one_speaker',[['yes','ναι'],['no','όχι'],['unsure','δεν ξέρω']])+'</div>'+
'<div class="q"><b>Το κείμενο ταιριάζει με ό,τι ακούγεται;</b>'+buttons(it,'text_matches',[['yes','ναι'],['minor','μικρές διαφορές'],['no','όχι'],['unsure','δεν ξέρω']])+'</div>'+
'<div class="q"><b>Τα κοψίματα είναι καθαρά (δεν κόβουν λέξη);</b>'+buttons(it,'boundaries_ok',[['yes','ναι'],['no','όχι'],['unsure','δεν ξέρω']])+'</div></section>').join('');
root.querySelectorAll('button[data-f]').forEach(b=>b.onclick=()=>mark(b.dataset.id,b.dataset.f,b.dataset.v));progress()}
window.addEventListener('beforeunload',()=>navigator.sendBeacon('/save',new Blob([JSON.stringify(V)],{type:'application/json'})));
fetch('answers.json',{cache:'no-store'}).then(r=>r.ok?r.json():{}).then(s=>{for(const k in s){const l=V[k];if(!l||(s[k].t||0)>=(l.t||0))V[k]=s[k]}draw()}).catch(()=>draw());
draw();
</script></body></html>""".replace("__ITEMS__", data)


if __name__ == "__main__":
    main()
