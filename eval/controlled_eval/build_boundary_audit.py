#!/usr/bin/env python3
"""Do the training clips actually contain the words we tell Whisper they contain?

Ten clips, one listen each. This is the cheap first pass of
`docs/specs/clip-alignment-audit.md`, whose Stage 0 has never been run.

The question is NOT whether the correction is right (that is the approval audit). It is
whether the CUT is right: the timestamps came from the original ASR pipeline and human
edits fixed text only, never times. So a clip can truncate its own first or last word, or
carry a neighbour's speech that no reference token covers. Either one trains the model on
a lie, and the spec ranks a truncated reference word as the worst case of the four.

Two audio elements per clip, and the pairing is the whole method:

  exact.wav  — [start, end] verbatim, the bytes the trainer feeds the encoder
  wide.wav   — [start-2, end+2], the same span with its surroundings restored

You answer from `exact`. You use `wide` only to hear what `exact` cut off. A first word
that sounds fine alone but turns out to be the tail of a longer word in `wide` is exactly
the failure this audit exists to catch, and it is invisible without the pair.

Everything is written OUTSIDE the repo: council audio and its transcription are the PII
category the 2026-07-21 purge removed from git history.

Usage:
  python eval/controlled_eval/build_boundary_audit.py
  AUDIT_DIR=~/oc-boundary-audit PORT=8781 python eval/controlled_eval/audit_server.py

Env: OUT_DIR (~/oc-boundary-audit) N (10) PAD (2.0) SEED (20260809)
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
AUDIO = ROOT / "data/asr/audio"
MANIFEST = ROOT / "data/asr/train_manifest.csv"
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-boundary-audit")).expanduser()
N = int(os.environ.get("N", "10"))
PAD = float(os.environ.get("PAD", "2.0"))
SEED = int(os.environ.get("SEED", "20260809"))
STRATIFY = os.environ.get("STRATIFY", "1") != "0"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cut(src: Path, start: float, end: float, dst: Path) -> bool:
    """Decode-accurate cut: -ss AFTER -i, so ffmpeg seeks on decoded samples.

    Fast seek (-ss before -i) lands on the nearest keyframe, which on a 3-hour mp3 can be
    hundreds of milliseconds off. That error is the same size as the thing being measured,
    so this audit would end up auditing ffmpeg.
    """
    start = max(0.0, start)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(src), "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-ac", "1", "-ar", "16000", str(dst),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and dst.exists()


def main() -> None:
    if not MANIFEST.exists():
        sys.exit(f"missing manifest: {MANIFEST}")

    rows = []
    with MANIFEST.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                s, e = float(r["utterance_start"]), float(r["utterance_end"])
            except (TypeError, ValueError):
                continue
            src = AUDIO / f"{r['city_id']}__{r['meeting_id']}.mp3"
            text = (r.get("text") or "").strip()
            # Need PAD of real audio on the left, or `wide` silently becomes `exact` on the
            # leading edge and the pair stops being a comparison.
            if e - s < 0.3 or not text or not src.exists() or s < PAD:
                continue
            r.update(_s=s, _e=e, _src=src, _text=text, _dur=e - s)
            rows.append(r)

    if not rows:
        sys.exit("no usable rows (is data/asr/audio populated?)")
    log(f"{len(rows)} usable rows")

    # Stratified, not uniform. A uniform draw of ten from a distribution whose mass sits at
    # 2-4s would show ten typical clips and say nothing about the tails, and the spec's
    # own Stage 0 asks for the extremes by duration and by chars/sec on purpose: a clip
    # whose text is too long for its span is the signature of a bad cut.
    rng = random.Random(SEED)
    if not STRATIFY:
        # Round two. Round one drew the extremes on purpose and found defects in 7 of 10,
        # which measures the extremes and nothing else. A uniform draw is the only thing
        # that answers the question that actually decides the next move: what fraction of
        # ORDINARY training clips is broken.
        picks = [("uniform", r) for r in rng.sample(rows, min(N, len(rows)))]
        rng.shuffle(picks)
        return _emit(picks)
    by_dur = sorted(rows, key=lambda r: r["_dur"])
    by_rate = sorted(rows, key=lambda r: len(r["_text"]) / r["_dur"])
    picks, seen = [], set()
    quotas = [
        ("shortest", by_dur[:200]),
        ("longest", by_dur[-200:]),
        ("dense-text", by_rate[-200:]),   # more characters than the span plausibly holds
        ("sparse-text", by_rate[:200]),   # span far longer than the words in it
    ]
    per = max(1, N // (len(quotas) + 1))
    for name, pool in quotas:
        for r in rng.sample(pool, min(per, len(pool))):
            if r["utterance_id"] not in seen:
                seen.add(r["utterance_id"])
                picks.append((name, r))
    for r in rng.sample(rows, min(len(rows), N * 4)):   # fill the rest with typical clips
        if len(picks) >= N:
            break
        if r["utterance_id"] not in seen:
            seen.add(r["utterance_id"])
            picks.append(("typical", r))
    picks = picks[:N]
    rng.shuffle(picks)   # so the stratum is not guessable from position on the page

    return _emit(picks)


def _emit(picks) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audio").mkdir(exist_ok=True)

    items = []
    for i, (stratum, r) in enumerate(picks, 1):
        cid = f"c{i:02d}"
        ok_e = cut(r["_src"], r["_s"], r["_e"], OUT_DIR / "audio" / f"{cid}_exact.wav")
        ok_w = cut(r["_src"], r["_s"] - PAD, r["_e"] + PAD, OUT_DIR / "audio" / f"{cid}_wide.wav")
        if not (ok_e and ok_w):
            log(f"  {cid}: ffmpeg failed, skipped")
            continue
        items.append({
            "id": cid,
            "text": r["_text"],
            "dur": round(r["_dur"], 2),
            "rate": round(len(r["_text"]) / r["_dur"], 1),
            "stratum": stratum,
            "utterance_id": r["utterance_id"],
            "city": r["city_id"],
            "meeting": r["meeting_id"],
            "start": round(r["_s"], 3),
            "end": round(r["_e"], 3),
        })
        log(f"  {cid} [{stratum}] {r['_dur']:.1f}s {r['city_id']}/{r['meeting_id']}")

    if not items:
        sys.exit("no clips were cut")

    # The metadata that would bias the ear (stratum, duration, chars/sec, city) stays out of
    # the page and lives here, next to the audio but never rendered.
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    page = {"items": [{"id": it["id"], "text": it["text"]} for it in items]}
    (OUT_DIR / "index.html").write_text(HTML.replace("__DATA__", json.dumps(page, ensure_ascii=False)), encoding="utf-8")

    log(f"{len(items)} clips -> {OUT_DIR}")
    print(f"\n  AUDIT_DIR={OUT_DIR} PORT=8781 python eval/controlled_eval/audit_server.py\n")


HTML = r"""<!doctype html><meta charset="utf-8"><title>Boundary audit</title>
<style>
 body{font:16px/1.5 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}
 .c{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1.2rem 0}
 .t{background:#f6f6f6;padding:.6rem .8rem;border-radius:6px;margin:.5rem 0;font-size:1.05rem}
 audio{width:100%;margin:.3rem 0}
 .lab{font-size:.85rem;color:#666;margin-top:.6rem}
 .q{margin:.7rem 0}
 .q b{display:block;font-size:.9rem;font-weight:600;margin-bottom:.25rem}
 button{font:inherit;padding:.3rem .7rem;margin-right:.35rem;border:1px solid #bbb;
        background:#fff;border-radius:5px;cursor:pointer}
 button[aria-pressed=true]{background:#2b6;color:#fff;border-color:#2b6}
 #bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:.6rem 0;
      display:flex;gap:1rem;align-items:center}
 @media(prefers-color-scheme:dark){
   body{background:#111;color:#eee}.c{border-color:#444}.t{background:#1c1c1c}
   button{background:#222;color:#eee;border-color:#555}#bar{background:#111;border-color:#333}}
</style>
<div id="bar"><b>Boundary audit</b><span id="n"></span></div>
<p>Απάντα ακούγοντας το <b>exact</b>. Το <b>wide</b> (±2s) είναι μόνο για να ακούσεις
τι έκοψε το exact. Οι ερωτήσεις αφορούν το <em>κόψιμο</em>, όχι αν το κείμενο είναι σωστό.</p>
<div id="app"></div>
<script>
const D=__DATA__, K='boundary-audit';
const S=JSON.parse(localStorage.getItem(K)||'{}');
const Q=[['first','Η ΠΡΩΤΗ λέξη του κειμένου ακούγεται ολόκληρη στο exact;',
          [['ok','ναι, ολόκληρη'],['cut','κομμένη στη μέση'],['gone','δεν ακούγεται καθόλου']]],
         ['last','Η ΤΕΛΕΥΤΑΙΑ λέξη του κειμένου ακούγεται ολόκληρη στο exact;',
          [['ok','ναι, ολόκληρη'],['cut','κομμένη στη μέση'],['gone','δεν ακούγεται καθόλου']]],
         ['extra','Ακούγεται στο exact ομιλία που ΔΕΝ υπάρχει στο κείμενο;',
          [['no','όχι'],['bit','λίγο (μισή λέξη)'],['lots','ναι, ολόκληρες λέξεις']]]];
function save(){localStorage.setItem(K,JSON.stringify(S));
  const n=Object.keys(S).length; document.getElementById('n').textContent=n+'/'+(D.items.length*3)+' απαντήσεις';
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(S)}).catch(()=>{});}
document.getElementById('app').innerHTML=D.items.map((it,i)=>`
 <div class="c"><b>${i+1}. ${it.id}</b>
  <div class="t">${it.text.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</div>
  <div class="lab">exact — αυτό ακριβώς τρώει το Whisper</div><audio controls preload=none src="audio/${it.id}_exact.wav"></audio>
  <div class="lab">wide — ±2 δευτ. γύρω, μόνο για έλεγχο</div><audio controls preload=none src="audio/${it.id}_wide.wav"></audio>
  ${Q.map(([k,q,opts])=>`<div class="q"><b>${q}</b>${opts.map(([v,l])=>
    `<button data-k="${it.id}.${k}" data-v="${v}">${l}</button>`).join('')}</div>`).join('')}
 </div>`).join('');
document.querySelectorAll('button').forEach(b=>{
  if((S[b.dataset.k]||{}).v===b.dataset.v)b.setAttribute('aria-pressed','true');
  b.onclick=()=>{S[b.dataset.k]={v:b.dataset.v,t:Date.now()};
    b.parentElement.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));
    b.setAttribute('aria-pressed','true');save();};});
save();
addEventListener('beforeunload',()=>navigator.sendBeacon('/save',new Blob([JSON.stringify(S)],{type:'application/json'})));
</script>"""

if __name__ == "__main__":
    main()
