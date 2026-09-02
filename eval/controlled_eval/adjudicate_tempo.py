"""Blind adjudication of the words the tempo arms disagree about.

Spec: docs/specs/2026-08-28-tempo-probe.md, the "adjudication" section.
Record: exp-2026-08-28-tempo-probe.

The pilot can only measure recovery of words the human already wrote down. Whether a
slowed arm surfaces words that are REALLY THERE but missing from a reference written at
1.0x needs a human, and only for the words that differ between arms. That is a few dozen
judgements instead of a re-transcription.

Blind in three ways, because an unblinded judgement here would be worthless: the page
never says which arm produced a candidate, the candidates are shuffled by a seed rather
than grouped, and the audio played is always the ORIGINAL-speed episode -- listening to
the stretched version would let the judge hear the arm.

A word counts as present in an arm only if it appears in at least 2 of the 3 cycles.
Streaming output is not deterministic; a one-cycle apparition is noise, and adjudicating
noise wastes the scarcest resource in this project, which is Harold's attention.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import tempo_probe as TP  # noqa: E402

OUT = TP.OUT
EPISODE_CLIPS = OUT / "episode-audio"
ADJUDICATION_PATH = OUT / "adjudication.json"
PAD_S = 1.0
MIN_CYCLES = 2                 # present in >= 2 of 3 cycles, else it is streaming noise
VERDICTS = ("supported", "unsupported", "cannot_tell")
SEED = 20260828


def _token(cell: str, ep_index: int, word: str) -> str:
    """Opaque handle for one candidate. Opacity, not secrecy: it keeps the arm, the
    meeting and the ordering from arriving unasked in front of the judge."""
    return hashlib.sha256(
        f"{cell}#{ep_index}#{word}".encode("utf-8")).hexdigest()[:16]


def arm_words(cell: str, arm: str, ep: TP.Episode,
              min_cycles: int = MIN_CYCLES) -> set:
    """Normalized words this arm put inside the episode in at least `min_cycles`."""
    counts: collections.Counter = collections.Counter()
    for cycle in range(TP.CYCLES):
        toks = TP.tokens_of(cell, arm, cycle)
        inside = {TP.greek_normalize(t.text).strip()
                  for t in toks
                  if ep.start <= ((t.start_ms + t.end_ms) / 2000.0) <= ep.end}
        counts.update({w for w in inside if w})
    return {w for w, n in counts.items() if n >= min_cycles}


def candidates(cells: dict, control: str = TP.CONTROL) -> list:
    """Words a non-control arm has that the control and the human reference do not.

    Only new words are adjudicated. A gold word an arm loses is already counted by the
    recall metric, and a non-gold word an arm loses tells us nothing we can check.
    """
    out = []
    for cell, blocks in sorted(cells.items()):
        for i, ep in enumerate(TP.episodes(blocks)):
            gold = set(ep.certain_words) | set(ep.uncertain_words)
            ctrl = arm_words(cell, control, ep)
            for arm in TP.ARMS:
                if arm == control:
                    continue
                for w in sorted(arm_words(cell, arm, ep) - ctrl - gold):
                    out.append({
                        "token": _token(cell, i, w),
                        "cell": cell, "episode": i, "word": w, "arm": arm,
                        "start": max(0.0, ep.start - PAD_S), "end": ep.end + PAD_S,
                    })
    return out


CHURN_CLASS = "__control_churn__"


def _words_one_cycle(cell: str, arm: str, cycle: int, ep) -> set:
    toks = TP.tokens_of(cell, arm, cycle)
    return {TP.greek_normalize(t.text).strip() for t in toks
            if ep.start <= ((t.start_ms + t.end_ms) / 2000.0) <= ep.end} - {""}


def control_churn_candidates(cells: dict, control: str = TP.CONTROL) -> list:
    """The same measurement, control against ITSELF: words one control cycle has and
    another does not.

    These are noise by construction -- same audio, same settings, same everything but
    the session. Mixed unlabelled into the adjudication they become the null class: if
    Harold calls the slowed arms' extra words real at the same rate as these, the arms
    have surfaced nothing. Without them a supported-rate has no yardstick, because
    nobody knows what fraction of ANY streaming word is audible on a second listen.
    """
    out, seen = [], set()
    for cell, blocks in sorted(cells.items()):
        for i, ep in enumerate(TP.episodes(blocks)):
            gold = set(ep.certain_words) | set(ep.uncertain_words)
            for a, b in ((0, 1), (0, 2), (1, 2)):
                A = _words_one_cycle(cell, control, a, ep) - gold
                B = _words_one_cycle(cell, control, b, ep) - gold
                for w in sorted(A - B):
                    tok = _token(cell, i, w)
                    if tok in seen:
                        continue
                    seen.add(tok)
                    out.append({"token": tok, "cell": cell, "episode": i, "word": w,
                                "arm": CHURN_CLASS,
                                "start": max(0.0, ep.start - PAD_S),
                                "end": ep.end + PAD_S})
    return out


def page_items(cands: list, seed: int = SEED) -> list:
    """What the browser may know: a token, a word, and how long the audio is.

    Not the arm, not the cell, not the meeting. Candidates found by two arms collapse to
    one presentation -- the same word in the same episode is one question, and asking it
    twice would let the judge notice which arm is being probed.
    """
    by_token: dict = {}
    for c in cands:
        by_token.setdefault(c["token"], c)
    items = [{"token": t, "word": c["word"],
              "dur": round(c["end"] - c["start"], 2)}
             for t, c in by_token.items()]
    random.Random(seed).shuffle(items)
    return items


def save_verdict(verdicts: dict, cands: list, token: str, verdict: str,
                 now: str) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}; frozen are {VERDICTS}")
    if token not in {c["token"] for c in cands}:
        raise ValueError(f"{token} is not a candidate in this session")
    out = dict(verdicts)
    out[token] = {"verdict": verdict, "saved_at": now}
    return out


def tally(cands: list, verdicts: dict) -> dict:
    """Per-arm supported / unsupported / cannot-tell counts over adjudicated words.

    A word found by two arms counts for both: the question is whether THAT arm surfaced
    a real word, and both did.
    """
    per = collections.defaultdict(lambda: dict.fromkeys(VERDICTS, 0))
    for c in cands:
        v = verdicts.get(c["token"])
        if v:
            per[c["arm"]][v["verdict"]] += 1
    out = {}
    for arm, counts in per.items():
        judged = sum(counts.values())
        resolved = counts["supported"] + counts["unsupported"]
        out[arm] = {
            **counts,
            "judged": judged,
            "supported_rate": round(counts["supported"] / resolved, 4) if resolved else None,
            "is_null_class": arm == CHURN_CLASS,
        }
    return out


def cut_episode(cand: dict) -> Path:
    """The ORIGINAL-speed audio of the episode. Never the stretched file: hearing the
    stretch would tell the judge which arm produced the word."""
    dst = EPISODE_CLIPS / f"{cand['token']}.wav"
    if dst.exists():
        return dst
    src = TP.CLIPS / f"{cand['cell']}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.wav")
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{cand['start']:.3f}",
         "-i", str(src), "-t", f"{cand['end'] - cand['start']:.3f}",
         "-ac", "1", "-ar", "16000", str(tmp)],
        capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ffmpeg failed for {cand['token']}")
    tmp.replace(dst)
    return dst


PAGE = """<!doctype html><html lang="el"><meta charset="utf-8">
<title>tempo-probe</title>
<style>
 body{font:16px/1.5 system-ui;margin:0;background:#111;color:#eee}
 main{max-width:40rem;margin:0 auto;padding:1.5rem}
 h1{font-size:1rem;color:#9ab;margin:0 0 .3rem}
 .sub{color:#667;font-size:.85rem;margin:0 0 1rem}
 .w{font-size:2rem;font-weight:700;margin:1rem 0;color:#fd7}
 audio{width:100%;margin:.5rem 0}
 button{font:15px system-ui;padding:.5rem 1rem;border-radius:6px;border:0;
   background:#2d6;color:#052;font-weight:600;cursor:pointer;margin-right:.4rem}
 button.no{background:#a44;color:#fee}
 button.idk{background:#333;color:#ccc;font-weight:400}
 .n{color:#789;font-variant-numeric:tabular-nums}
</style>
<main>
 <h1>Ακούγεται αυτή η λέξη μέσα στο απόσπασμα;</h1>
 <p class="sub">Δύο εκδοχές του ίδιου ήχου διαφώνησαν σε αυτή τη λέξη. Δεν σου λέμε
  ποια την είπε — γι' αυτό υπάρχει η σελίδα. Άκου και πες αν ακούγεται.</p>
 <div class="bar"><span class="n" id="pos"></span></div>
 <audio id="a" controls preload="none"></audio>
 <div class="w" id="w"></div>
 <div>
  <button id="yes">Ακούγεται</button>
  <button class="no" id="no">Δεν ακούγεται</button>
  <button class="idk" id="idk">Δεν καταλαβαίνω</button>
  <span class="n" id="msg"></span>
 </div>
</main>
<script>
const Q = __DATA__; let i = 0, busy = false;
const $ = s => document.querySelector(s);
function show(){
  const c = Q[i];
  $('#pos').textContent = Q.length ? (i+1) + ' / ' + Q.length : 'τέλος';
  $('#w').textContent = c ? c.word : '';
  $('#a').src = c ? '/clip/' + c.token + '.wav' : '';
  $('#msg').textContent = '';
  ['#yes','#no','#idk'].forEach(b => $(b).disabled = !c);
}
async function commit(v){
  if(busy) return; const c = Q[i]; if(!c) return;
  busy = true;
  try{
    const r = await fetch('/verdict', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({token: c.token, verdict: v})});
    if(!r.ok){ $('#msg').textContent = await r.text(); return; }
    if(i < Q.length-1){ i++; show(); } else { $('#msg').textContent = 'τέλος'; }
  } finally { busy = false; }
}
$('#yes').onclick = () => commit('supported');
$('#no').onclick = () => commit('unsupported');
$('#idk').onclick = () => commit('cannot_tell');
show();
</script></html>"""


def main() -> None:  # pragma: no cover - operational entry point
    import datetime as _dt
    import http.server
    import socketserver
    import urllib.parse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["build", "serve", "tally"])
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()

    cells = TP.load_cells()
    cands = candidates(cells) + control_churn_candidates(cells)
    by_token = {c["token"]: c for c in cands}
    verdicts = json.loads(ADJUDICATION_PATH.read_text()) \
        if ADJUDICATION_PATH.exists() else {}

    if a.mode == "tally":
        print(json.dumps({"judged": len(verdicts), "candidates": len(by_token),
                          "by_arm": tally(cands, verdicts)},
                         ensure_ascii=False, indent=2))
        return

    if a.mode == "build":
        for c in by_token.values():
            cut_episode(c)
        print(json.dumps({"candidates": len(cands), "presentations": len(by_token),
                          "by_arm": collections.Counter(c["arm"] for c in cands)},
                         ensure_ascii=False, indent=2))
        return

    from eval.controlled_eval.blind_listen import _bindable
    if not _bindable(a.host):
        raise SystemExit(f"refused: {a.host!r}; loopback or tailnet only")

    todo = [it for it in page_items(cands) if it["token"] not in verdicts]
    for c in by_token.values():
        cut_episode(c)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a_):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = urllib.parse.urlparse(self.path).path
            if p in ("/", "/index.html"):
                html = PAGE.replace("__DATA__", json.dumps(todo, ensure_ascii=False))
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if p.startswith("/clip/") and p.endswith(".wav"):
                tok = p[len("/clip/"):-len(".wav")]
                c = by_token.get(tok)
                if not c:
                    return self.send_error(404)
                return self._send(200, cut_episode(c).read_bytes(), "audio/wav")
            return self.send_error(404)

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/verdict":
                return self.send_error(404)
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                if not 0 < n <= 8192:
                    raise ValueError("bad Content-Length")
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                verdicts.update(save_verdict(verdicts, cands, body.get("token"),
                                             body.get("verdict"), now))
            except (ValueError, TypeError, AttributeError, UnicodeDecodeError) as e:
                return self._send(400, str(e).encode(), "text/plain; charset=utf-8")
            tmp = ADJUDICATION_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, ADJUDICATION_PATH)
            return self._send(200, b'{"ok":true}', "application/json")

    print(f"{len(todo)} candidates to judge · http://{a.host}:{a.port}", flush=True)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((a.host, a.port), H) as s:
        s.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
