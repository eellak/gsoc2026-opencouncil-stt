"""Build the private interval repair page for the two insertion-heavy validation windows.

Frozen protocol: docs/specs/2026-08-19-dense-reference-repair.md.

The window is cut into fixed 20 s intervals. Each interval shows its own audio and the
published words that the forced alignment places inside it, editable in place, so the
reviewer can see what was already written down while listening. Run
`align_published_reference` first; the page refuses to build without those times.

The served tree carries opaque review ids, audio symlinks and the published reference
only: no hypothesis, arm, seed, WER, S/D/I, selection reason or real window id. Audio and
text stay under ~/.cache, never in git.

    .venv-eval/bin/python -m eval.controlled_eval.build_reference_repair_page
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# Frozen by id in the spec. Never recomputed from insertion deltas again.
WINDOW_IDS = ("win_argos_oct31__2_2025_2353650", "win_argos_sep24_2025_371824")
CHUNK_SEC = 20.0

DEFAULT_DENSE = Path.home() / ".cache/oc-public/dense-eval-recovery-cx9ra8iluhr225"
DEFAULT_AUDIT = Path.home() / ".cache/oc-public/training-listening-audit-2026-08"
DEFAULT_OUT = Path.home() / ".cache/oc-public/dense-reference-repair-2026-08"
REPO = Path(__file__).resolve().parents[2]


def chunk_bounds(duration_sec: float, chunk_sec: float = CHUNK_SEC) -> list[tuple[float, float]]:
    """Fixed navigation intervals covering the whole window; the last one is short."""
    n = max(1, math.ceil(duration_sec / chunk_sec))
    return [(round(i * chunk_sec, 3), round(min((i + 1) * chunk_sec, duration_sec), 3))
            for i in range(n)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def review_id(window_id: str) -> str:
    return hashlib.sha256(f"dense-reference-repair|{window_id}".encode()).hexdigest()[:16]


def link_audio(source: Path, clips: Path, item_id: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = clips / f"{item_id}{source.suffix.lower() or '.audio'}"
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"audio link collision: {target}")
    elif target.exists():
        raise RuntimeError(f"refusing to replace non-symlink: {target}")
    else:
        target.symlink_to(source.resolve())
    return f"clips/{target.name}"


def prior_verdicts(audit_root: Path) -> dict[str, str]:
    """Map frozen window id -> the completeness verdict the blind audit recorded."""
    key = json.loads((audit_root / "hidden-key.json").read_text())
    answers = json.loads((audit_root / "served/answers.json").read_text())
    rows = [row for row in key if row["kind"] == "validation_window"]
    seen = {row["window_id"] for row in rows}
    if seen != set(WINDOW_IDS) or len(rows) != len(WINDOW_IDS):
        raise RuntimeError(f"prior audit key does not map exactly the frozen windows: {sorted(seen)}")
    verdicts = {}
    for row in rows:
        verdict = (answers.get(row["review_id"]) or {}).get("reference_completeness")
        if verdict != "material_omission":
            raise RuntimeError(
                f"window {row['window_id']} was not judged material_omission ({verdict!r}); "
                "the repair queue is frozen and is not re-derived")
        verdicts[row["window_id"]] = verdict
    return verdicts


def chunk_texts(tokens: list[str], times: list[list[float]],
                bounds: list[tuple[float, float]]) -> list[str]:
    """Assign every published token to the interval its aligned start falls in.

    Tokens travel by occurrence index, never by string, and every token lands in exactly
    one interval, so joining the intervals reproduces the published reference verbatim.
    """
    if len(tokens) != len(times):
        raise RuntimeError(f"{len(times)} aligned times for {len(tokens)} published tokens")
    buckets: list[list[str]] = [[] for _ in bounds]
    for token, (start, _end) in zip(tokens, times):
        index = next((i for i, (lo, hi) in enumerate(bounds) if lo <= start < hi), len(bounds) - 1)
        buckets[index].append(token)
    return [" ".join(bucket) for bucket in buckets]


def build_package(dense_root: Path, audit_root: Path, out: Path,
                  aligned: dict[str, list[list[float]]] | None = None) -> tuple[list[dict], dict]:
    out = out.resolve()
    if out == REPO or REPO in out.parents:
        raise RuntimeError(f"refusing to write an audio/transcript package inside the repository: {out}")
    prior_verdicts(audit_root)

    validation = dense_root / "corrected-input/eval/validation"
    manifest = json.loads((validation / "manifest.json").read_text())["eval_windows"]
    by_id: dict[str, list[dict]] = {}
    for row in manifest:
        by_id.setdefault(row["window_id"], []).append(row)
    for window_id in WINDOW_IDS:
        rows = by_id.get(window_id, [])
        if len(rows) != 1:
            raise RuntimeError(f"frozen window {window_id} has {len(rows)} manifest rows, expected 1")

    if aligned is None:
        aligned_path = out / "aligned.json"
        if not aligned_path.is_file():
            raise RuntimeError(
                f"missing {aligned_path}: run "
                "`.venv-eval/bin/python -m eval.controlled_eval.align_published_reference` first")
        aligned = json.loads(aligned_path.read_text())

    served = out / "served"
    clips = served / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    items, windows = [], []
    for window_id in WINDOW_IDS:
        row = by_id[window_id][0]
        item_id = review_id(window_id)
        audio = validation / row["audio_local"]
        if validation.resolve() not in audio.resolve().parents:
            raise RuntimeError(f"audio escapes the validation directory: {audio}")
        tokens = list(row["ref_tokens"])
        bounds = chunk_bounds(row["duration_sec"])
        texts = chunk_texts(tokens, aligned[window_id], bounds)
        if " ".join(text for text in texts if text) != " ".join(tokens):
            raise RuntimeError(f"interval split does not reproduce the reference: {window_id}")
        items.append({
            "review_id": item_id,
            "audio": link_audio(audio, clips, item_id),
            "duration_sec": row["duration_sec"],
            "chunks": [{"start": lo, "end": hi, "text": text}
                       for (lo, hi), text in zip(bounds, texts)],
            "published_text": " ".join(tokens),
        })
        windows.append({
            "review_id": item_id,
            "window_id": window_id,
            "published_tokens": tokens,
            "audio_sha256": sha256_file(audio),
            "duration_sec": row["duration_sec"],
        })

    fingerprint = hashlib.sha256(json.dumps(
        ["interval-v2"] +
        [[w["window_id"], w["audio_sha256"], " ".join(w["published_tokens"]),
          aligned[w["window_id"]]] for w in windows],
        ensure_ascii=False).encode()).hexdigest()[:16]
    sources = {
        "spec": "docs/specs/2026-08-19-dense-reference-repair.md",
        "source_fingerprint": fingerprint,
        "windows": windows,
    }
    (out / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1) + "\n")
    return items, sources


def render(items: list[dict], fingerprint: str) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="el"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Διόρθωση αναφορών</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f5f5f2;color:#181818}
header{position:sticky;top:0;z-index:2;padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
main{max-width:920px;margin:auto;padding:16px}
.card{background:#fff;border:1px solid #ddd;border-radius:9px;padding:16px;margin:16px 0}
.muted{color:#666;font-size:13px}.saved{color:#287a3c}
button{font:inherit;padding:6px 11px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button.on{background:#245b9e;color:#fff;border-color:#245b9e}
button.done{border-color:#287a3c;color:#287a3c}
.chunk{border-top:1px solid #eee;padding:10px 0}
.chunk h4{margin:0 0 6px;font-size:14px;font-weight:600;display:flex;gap:9px;align-items:center}
.chunk .t{font-variant-numeric:tabular-nums;color:#666;font-weight:400}
.chunk.edited h4{color:#8b5a00}
audio{width:100%}textarea{width:100%;font:inherit;box-sizing:border-box;min-height:64px}
.locked{opacity:.5;pointer-events:none}
</style></head><body>
<header><b>Διόρθωση αναφοράς με βάση τον ήχο</b><span id="progress" class="muted"></span><span id="saved" class="saved"></span><button id="export">Export</button></header>
<main>
<p>Κάθε παράθυρο είναι κομμένο σε διαστήματα 20 δευτερολέπτων. Σε κάθε διάστημα βλέπεις
<b>τις λέξεις που έχει ήδη η δημοσιευμένη αναφορά εκεί</b> (τοποθετημένες στον χρόνο με
αυτόματη ευθυγράμμιση, οπότε μια λέξη μπορεί να πέσει λίγο πριν ή μετά το όριο).</p>
<p>Δουλειά σου: πάτα ▶, άκου το διάστημα, και <b>διόρθωσε το κείμενο ώστε να λέει ό,τι
ακούγεται</b> — πρόσθεσε ό,τι λείπει, σβήσε ό,τι δεν ειπώθηκε, φτιάξε λάθος λέξεις. Όταν
μιλάνε παράλληλα, γράψε ό,τι ξεχωρίζεις, με τη σειρά που αρχίζει· ό,τι είναι ακατάληπτο
το προσπερνάς ή βάζεις <code>[?]</code>. Στίξη, κεφαλαία και τόνοι δεν μετράνε. Καμία
έξοδος μοντέλου δεν εμφανίζεται εδώ.</p>
<div id="items"></div></main>
<script>
const ITEMS=__ITEMS__;const FP='__FP__';const KEY='oc-reference-repair-'+FP;
let V=JSON.parse(localStorage.getItem(KEY)||'{}');let timer=null;
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function st(id){V[id]=V[id]||{chunks:{}};V[id].chunks=V[id].chunks||{};return V[id]}
function textOf(it,i){const v=st(it.review_id);const t=v.chunks[i];return t===undefined?it.chunks[i].text:t}
function joined(it){return it.chunks.map((c,i)=>textOf(it,i).trim()).filter(Boolean).join(' ')}
function edited(it,i){return textOf(it,i)!==it.chunks[i].text}
function done(it){const v=st(it.review_id);return !!(v.finalized&&v.listened_to_full_audio&&joined(it))}
function progress(){document.getElementById('progress').textContent=ITEMS.filter(done).length+'/'+ITEMS.length+' κλειδωμένα'}
async function save(){localStorage.setItem(KEY,JSON.stringify(V));try{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(V)});document.getElementById('saved').textContent=r.ok?'αποθηκεύτηκε':'μόνο τοπικά'}catch(e){document.getElementById('saved').textContent='μόνο τοπικά'}}
function schedule(){document.getElementById('saved').textContent='αποθήκευση…';clearTimeout(timer);timer=setTimeout(save,350)}
function touch(it){const v=st(it.review_id);v.source_fingerprint=FP;v.repaired_text=joined(it);v.t=Date.now();schedule()}
function play(it,from,until){const a=document.querySelector('audio[data-id="'+it.review_id+'"]');a.dataset.until=until;a.currentTime=from;a.play()}
function draw(){const root=document.getElementById('items');
root.innerHTML=ITEMS.map((it,n)=>{const v=st(it.review_id);
const chunks=it.chunks.map((c,i)=>'<div class="chunk'+(edited(it,i)?' edited':'')+'"><h4><button data-play="'+it.review_id+'" data-i="'+i+'">▶</button><span class="t">'+c.start.toFixed(0)+'–'+c.end.toFixed(0)+'s</span>'+(edited(it,i)?'αλλαγμένο':'')+'</h4><textarea data-chunk="'+it.review_id+'" data-i="'+i+'" placeholder="(η αναφορά δεν έχει λέξεις εδώ — γράψε ό,τι ακούγεται)">'+esc(textOf(it,i))+'</textarea></div>').join('');
return '<section class="card'+(done(it)?' ':'')+'"><div class="muted">Παράθυρο '+(n+1)+' · '+Math.round(it.duration_sec)+'s'+(done(it)?' · κλειδωμένο':'')+'</div>'+
'<audio controls preload="none" data-id="'+it.review_id+'" src="'+it.audio+'"></audio>'+
'<div'+(done(it)?' class="locked"':'')+'>'+chunks+
'<div class="chunk"><label><input type="checkbox" data-full="'+it.review_id+'"'+(v.listened_to_full_audio?' checked':'')+'> άκουσα ολόκληρο τον ήχο του παραθύρου</label> '+
'<button class="'+(v.finalized?'done':'')+'" data-final="'+it.review_id+'">'+(v.finalized?'κλειδωμένο ✓':'κλείδωμα')+'</button></div></div></section>'}).join('');
root.querySelectorAll('button[data-play]').forEach(b=>b.onclick=()=>{const it=ITEMS.find(x=>x.review_id===b.dataset.play);const c=it.chunks[b.dataset.i];play(it,c.start,c.end)});
root.querySelectorAll('textarea[data-chunk]').forEach(t=>t.oninput=()=>{const it=ITEMS.find(x=>x.review_id===t.dataset.chunk);st(it.review_id).chunks[t.dataset.i]=t.value;touch(it)});
root.querySelectorAll('input[data-full]').forEach(c=>c.onchange=()=>{const it=ITEMS.find(x=>x.review_id===c.dataset.full);st(it.review_id).listened_to_full_audio=c.checked;touch(it);draw()});
root.querySelectorAll('button[data-final]').forEach(b=>b.onclick=()=>{const it=ITEMS.find(x=>x.review_id===b.dataset.final);const v=st(it.review_id);v.finalized=!v.finalized;touch(it);draw()});
document.querySelectorAll('audio').forEach(a=>a.ontimeupdate=()=>{const u=parseFloat(a.dataset.until||'0');if(u&&a.currentTime>=u){a.pause();a.dataset.until=''}});
progress()}
document.getElementById('export').onclick=()=>{const b=new Blob([JSON.stringify(V,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='reference-repair-answers.json';a.click()};
window.addEventListener('beforeunload',()=>navigator.sendBeacon('/save',new Blob([JSON.stringify(V)],{type:'application/json'})));
fetch('answers.json',{cache:'no-store'}).then(r=>r.ok?r.json():{}).then(server=>{for(const k in server){const s=server[k],l=V[k];if(!s||s.source_fingerprint!==FP)continue;if(!l||(s.t||0)>=(l.t||0))V[k]=s}draw()}).catch(()=>draw());
draw();
</script></body></html>"""
    return template.replace("__ITEMS__", data).replace("__FP__", fingerprint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    items, sources = build_package(args.dense_root, args.audit_root, args.out)
    (args.out / "served/index.html").write_text(render(items, sources["source_fingerprint"]),
                                                encoding="utf-8")
    print(json.dumps({
        "served": str(args.out / "served"),
        "sources": str(args.out / "sources.json"),
        "windows": len(items),
        "source_fingerprint": sources["source_fingerprint"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
