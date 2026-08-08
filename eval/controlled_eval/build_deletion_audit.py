"""Blinded listening package for the deletion question.

The same-stack measurement killed the number the diagnosis rested on: with one decoder,
the fine-tune deletes +0.19 points more than base, interval [−0.50, +0.85]. What it did
not do is say what the remaining deletions *are*. Both systems drop runs of reference
words — 120 sites where base drops and the fine-tune keeps, 153 the other way — and the
mechanism on the table says the fine-tune's should be short second-speaker turns, because
it was trained on single-utterance clips.

That is an audible claim, so listen to it. Each clip carries a marked span where one
system produced nothing and the other produced words. Nothing in the package says which
system did which, and the two directions are shuffled together.

The retaining system's word timestamps locate the span: it is the only side that knows
where in the audio those words are. Only the windows that actually contribute a site get
decoded again, so this costs about an hour of CPU rather than another full pass.

  SC=~/.cache/oc-public .venv-eval/bin/python -m eval.controlled_eval.build_deletion_audit
Env: SC OUT_DIR N_PER_SIDE PAD_SEC
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.controlled_eval import bench_data as B  # noqa: E402
from eval.controlled_eval.scoring import wtoks  # noqa: E402

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
WINDOWS = SC / "bench_windows"
HYPS = SC / "bench_hyps_fw"
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "oc-deletion-audit"))
N_PER_SIDE = int(os.environ.get("N_PER_SIDE", "24"))
PAD_SEC = float(os.environ.get("PAD_SEC", "2.5"))
MIN_RUN = 4            # shorter runs are single dropped words, not a lost turn
MAX_PER_WINDOW = 2     # spread the sample over meetings instead of over one bad window
FT_MODEL = "/home/harold/oc-asr-serve/ct2"


def log(m):
    print(m, flush=True)


def align(ref: list[str], hyp: list[str]) -> list[int | None]:
    """For each reference word, the hypothesis index it aligns to, or None if deleted.

    Ties break toward substitution, same as everywhere else in this project: a reference
    word sitting opposite another word is a wrong word, not a missing one.
    """
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def deleted_runs(gone: list[int | None], kept: list[int | None]) -> list[tuple[int, int]]:
    """Reference spans one system dropped entirely while the other matched every word."""
    res, start = [], None
    for k in range(len(gone)):
        ok = gone[k] is None and kept[k] is not None
        if ok and start is None:
            start = k
        elif not ok:
            if start is not None and k - start >= MIN_RUN:
                res.append((start, k))
            start = None
    if start is not None and len(gone) - start >= MIN_RUN:
        res.append((start, len(gone)))
    return res


def main() -> None:
    base = json.loads((HYPS / "base-fw.json").read_text())
    ft = json.loads((HYPS / "finetune-fw.json").read_text())
    report = B.load_report()
    by_id = {it["itemId"]: it for it in report["items"]}
    ids = sorted(set(base) & set(ft) & set(by_id))
    log(f"{len(ids)} windows")

    sites = []
    for iid in ids:
        ref = wtoks(by_id[iid]["referenceText"])
        ab, af = align(ref, wtoks(base[iid])), align(ref, wtoks(ft[iid]))
        for who, gone, kept in (("base", ab, af), ("finetune", af, ab)):
            for s, e in deleted_runs(gone, kept)[:MAX_PER_WINDOW]:
                sites.append({"item_id": iid, "deleted_by": who, "kept_by":
                              "finetune" if who == "base" else "base",
                              "ref_from": s, "ref_to": e,
                              "ref_words": " ".join(ref[s:e])})

    rnd = random.Random(11)
    chosen = []
    for who in ("base", "finetune"):
        pool = [x for x in sites if x["deleted_by"] == who]
        rnd.shuffle(pool)
        seen = {}
        for x in pool:
            if seen.get(x["item_id"], 0) >= MAX_PER_WINDOW:
                continue
            seen[x["item_id"]] = seen.get(x["item_id"], 0) + 1
            chosen.append(x)
            if sum(1 for c in chosen if c["deleted_by"] == who) >= N_PER_SIDE:
                break
        log(f"{who} deletes: {len(pool)} sites available, "
            f"{sum(1 for c in chosen if c['deleted_by'] == who)} chosen")

    # Only the retaining system knows where those words are, so decode per (window, side).
    need: dict[str, set[str]] = {}
    for x in chosen:
        need.setdefault(x["kept_by"], set()).add(x["item_id"])
    log(f"decodes needed: " + ", ".join(f"{k} {len(v)}" for k, v in need.items()))

    from faster_whisper import WhisperModel
    words_cache: dict[tuple[str, str], list] = {}
    for side, items in need.items():
        model = WhisperModel("large-v3" if side == "base" else FT_MODEL,
                             device="cpu", compute_type="int8", cpu_threads=16)
        for n, iid in enumerate(sorted(items), 1):
            segs, _ = model.transcribe(str(WINDOWS / f"{iid}.wav"), language="el",
                                       beam_size=5, word_timestamps=True,
                                       condition_on_previous_text=False)
            ws = [(w.word, float(w.start), float(w.end))
                  for s in segs for w in (s.words or [])]
            words_cache[(side, iid)] = ws
            if n % 5 == 0:
                log(f"  {side}: {n}/{len(items)}")
        del model

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clips").mkdir(exist_ok=True)
    rows = []
    for x in chosen:
        ws = words_cache[(x["kept_by"], x["item_id"])]
        ref = wtoks(by_id[x["item_id"]]["referenceText"])
        keep_hyp = wtoks(ft[x["item_id"]] if x["kept_by"] == "finetune"
                         else base[x["item_id"]])
        a = align(ref, keep_hyp)
        idxs = [a[k] for k in range(x["ref_from"], x["ref_to"]) if a[k] is not None]
        # The word list from this decode and the stored transcript are the same decoder
        # on the same audio, but not literally the same call, so the index can run past
        # the end on a rare mismatch. Drop the site rather than mark the wrong second.
        if not idxs or max(idxs) >= len(ws):
            continue
        t0, t1 = ws[min(idxs)][1], ws[max(idxs)][2]
        dur = float(by_id[x["item_id"]]["durationSec"])
        c0, c1 = max(0.0, t0 - PAD_SEC), min(dur, t1 + PAD_SEC)
        rows.append({**x, "mark": [round(t0 - c0, 2), round(t1 - c0, 2)],
                     "clip_start": round(c0, 3), "clip_dur": round(c1 - c0, 3)})

    rnd.shuffle(rows)
    for n, r in enumerate(rows, 1):
        r["clip"] = f"{n:03d}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r['clip_start']:.3f}",
                        "-t", f"{r['clip_dur']:.3f}",
                        "-i", str(WINDOWS / f"{r['item_id']}.wav"), "-ac", "1",
                        "-ar", "16000", str(OUT_DIR / "clips" / r["clip"])], check=True)

    public = [{"clip": r["clip"], "mark": r["mark"]} for r in rows]
    (OUT_DIR / "KEY_DO_NOT_OPEN_UNTIL_DONE.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    (OUT_DIR / "index.html").write_text(
        HTML.replace("__MANIFEST__", json.dumps(public, ensure_ascii=False)))
    n_base = sum(1 for r in rows if r["deleted_by"] == "base")
    log(f"{len(rows)} clips ({n_base} base-deleted, {len(rows)-n_base} finetune-deleted)"
        f" -> {OUT_DIR}")


HTML = """
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ακρόαση: τι λείπει</title>
<style>
 body{font:16px/1.5 system-ui;margin:0;padding:16px;max-width:760px}
 .clip{border:1px solid #ccc;border-radius:8px;padding:12px;margin:12px 0}
 .clip.done{background:#f2f8f2;border-color:#8bc48b}
 audio{width:100%;margin:8px 0}
 button{font:15px system-ui;padding:10px 12px;margin:3px 3px 0 0;border-radius:6px;
        border:1px solid #999;background:#fff}
 button.sel{background:#2b6;color:#fff;border-color:#2b6}
 .mark{font-weight:700}
 header{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd}
</style>
<header>
 <b>Σημαδεμένο διάστημα:</b> ένα σύστημα δεν έγραψε τίποτα εκεί, ένα άλλο έγραψε λέξεις.
 Δεν λέγεται ποιο. <span id="n">0</span>/<span id="tot">0</span> · <span id="sync"></span>
 <button id="pushbtn">αποθήκευση</button>
</header>
<div id="list"></div>
<script>
const K='delaudit';
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
     <button onclick="pick('${x.clip}','main',this)">Κύριος ομιλητής, συνεχής ομιλία</button>
     <button onclick="pick('${x.clip}','second',this)">Δεύτερος ομιλητής ή παρεμβολή</button>
     <button onclick="pick('${x.clip}','change',this)">Αλλαγή ομιλητή μέσα στο σημάδι</button>
     <button onclick="pick('${x.clip}','unclear',this)">Δεν ακούγεται καθαρά</button>
    </div>
   </div>`).join('');
  m.forEach(x=>{const v=A[x.clip]; if(!v) return;
    const d=document.querySelector(`[data-c="${x.clip}"]`);
    d.classList.add('done');
    const map={main:0,second:1,change:2,unclear:3};
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
