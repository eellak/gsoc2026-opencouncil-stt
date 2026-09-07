"""The blind listening page: audio in, a transcript out, the witness only afterwards.

Spec: docs/specs/2026-08-26-training-round-2.md §4.1.
Protocol: docs/specs/listening-protocol.md, version 1, FROZEN 2026-08-09.
Record: exp-2026-08-26-training-round-2.

The task is not to judge who is right. It is to write what was said. So the listener
sees a player and an empty box, and nothing else: no adapter hypothesis, no Soniox text,
no left/right context, no "these are the words that went missing". Any of those would
anchor the label, and a label anchored on Soniox is a copy of Soniox rather than evidence
about it -- which would silently destroy the one thing this round is for.
`chooser/listen_server.py:29-31` shows all of that at once and cannot be reused.

§4.1 asks the page to ENFORCE the order rather than rely on discipline, so the
enforcement is structural in three layers:

  * the queue file carries no text at all (`cut_listening_clips.py` keeps the witness and
    the current training target in files this server never serves);
  * the browser receives an opaque token and a duration -- not the clip id, because a
    clip id names its meeting and the meeting has a published transcript;
  * `/reveal` refuses to answer until a committed answer exists for that presentation.

WHAT THE FROZEN PROTOCOL ADDS, AND WHY IT IS NOT OPTIONAL

The 2026-08-09 protocol governs every blind label this project has taken, including the
48 August windows. Labels collected under different rules do not belong to the same set
and cannot be separated afterwards. So:

  * write what was SAID, not what should have been said. Intelligible false starts and
    repetitions are written as spoken; only fillers (`ε`, `εμ`, `μμ`) are dropped, and
    numbers are written as spoken rather than as digits.
  * unintelligible speech is never guessed. It is marked, with one of three frozen
    reasons, and the share per reason is itself a finding.
  * after three listens the listener marks it and moves on; time is the scarce resource.
  * five clips are presented a second time at the end, without their earlier answers
    visible. The listener's disagreement with themselves is the NOISE FLOOR of the whole
    set, and a repeat is indistinguishable from a first hearing in the page -- a repeat
    the listener can spot measures memory instead. Repeats never become training rows.

NOTE ON A CONFLICT, resolved in favour of the frozen document. §4.1 of the round-2 spec
says to strip false starts as well as fillers. The frozen protocol says to keep them.
This module collects the RAW label under the frozen protocol, because a false start that
was never written cannot be recovered while one that was written can always be stripped.
Deriving the cleaned training target is a separate, versioned step and is not done here.

Serving: bound to one address on purpose. `0.0.0.0` would put council audio on every
network this machine joins, the same PII category as the 2026-07-21 history purge, so it
is refused rather than warned about.

    .venv-eval/bin/python -m eval.controlled_eval.blind_listen --host 100.x.y.z
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import http.server
import json
import os
import re
import socketserver
import subprocess
import urllib.parse
from pathlib import Path

SC = Path.home() / ".cache" / "oc-public"
ROOT_DIR = SC / "listen-2026-08"
# `current` is a symlink to one immutable generation under `generations/`. The labels
# sit OUTSIDE it, because they belong to the listener's session rather than to a cut.
CURRENT = ROOT_DIR / "current"
QUEUE_PATH = CURRENT / "queue.json"
CLIPS_DIR = CURRENT / "clips"
WITNESS_PATH = CURRENT / "witness.json"
TARGET_PATH = CURRENT / "current_target.json"
LABELS_PATH = ROOT_DIR / "labels.json"

# A label is a few hundred bytes. Anything larger is not a transcript, and reading it
# would block the single-threaded server for as long as the sender cared to take.
MAX_BODY_BYTES = 64 * 1024
SOCKET_TIMEOUT_SEC = 30

# --------------------------------------------------------------------- frozen
STATES = ("transcribed", "uncertain", "excluded")
EXCLUSION_REASONS = ("unintelligible", "overlap", "noise_or_music")
MAX_LISTENS = 3      # listening-protocol.md: mark uncertain and move on
N_REPEATS = 5        # listening-protocol.md: the listener's own noise floor

# USER DECISION 2026-08-28. Typing 721 clips from scratch is 9-17 hours and Harold said
# plainly he would not do it; the alternative he proposed -- correct Soniox instead --
# makes every label a derivative of the witness it is supposed to judge. The split keeps
# both: a BLIND CORE of 120 clips typed from nothing, which is the only material any
# measurement may use, and an ASSISTED remainder where Soniox is prefilled and edited,
# which is training data only. Every label records which condition produced it, and
# `training_rows(mode="blind")` is how a measurement asks for the clean set.
BLIND_CORE_N = 120
MODES = ("blind", "assisted")

_CLIP_ID_RE = re.compile(r"^[a-z0-9]+__[a-z0-9_]+__\d+$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{16}$")


class LabelError(ValueError):
    """The answer cannot be stored: empty, unknown state, or outside the session."""


class RevealDenied(PermissionError):
    """The witness was asked for before an answer existed. This is the whole point."""


def is_safe_clip_id(clip_id: str) -> bool:
    """Clip ids are `<city>__<meeting>__<offset_ms>` and nothing else.

    A whitelist, not a blacklist: the id is pasted straight into a filesystem path, and
    every blacklist of separators eventually misses one.
    """
    return bool(_CLIP_ID_RE.match(clip_id or ""))


def present_token(clip_id: str, occurrence: int) -> str:
    """The opaque handle the browser uses for one PRESENTATION of a clip.

    Keyed on the occurrence as well as the clip, so the second hearing of a clip is a
    different token and cannot be recognised as a repeat from the page. Deterministic,
    so a session can be resumed and the order reproduced. It is opacity, not secrecy:
    anyone holding the queue file can invert it. What it stops is the meeting -- and the
    fact that this one is a repeat -- arriving unasked in front of the listener.
    """
    return hashlib.sha256(f"{clip_id}#{occurrence}".encode("utf-8")).hexdigest()[:16]


def _even_indices(n_total: int, k: int) -> list[int]:
    """`k` indices spread evenly over `range(n_total)`, deterministic and sorted."""
    k = min(k, n_total)
    if k <= 0:
        return []
    step = (n_total - 1) / (k - 1) if k > 1 else 0
    return sorted({int(round(i * step)) for i in range(k)})


def blind_core(queue, n: int = BLIND_CORE_N) -> list[str]:
    """The clip ids Harold types from nothing, taken at an even step over the queue.

    Even step rather than random: the queue is ordered by city and meeting, so this
    inherits its city mix, and it is reproducible from the queue file alone -- a random
    core would have to be stored somewhere, and a core that drifted between sessions
    would silently mix the two conditions.
    """
    return [queue[i]["clip_id"] for i in _even_indices(len(queue), n)]


def session_order(queue, n_repeats: int = N_REPEATS,
                  n_blind: int = BLIND_CORE_N) -> list[dict]:
    """The blind core first, then its repeats, then the assisted remainder.

    Blind first because it is the material that decides: if the session stops early,
    what exists is the clean set rather than half of each. The repeats are drawn from
    the core alone -- a repeat of an assisted clip would measure how consistently he
    edits Soniox, not how consistently he hears.
    """
    core = set(blind_core(queue, n_blind))
    by_id = {c["clip_id"]: c for c in queue}

    def row(c, occ, repeat, mode):
        return {"token": present_token(c["clip_id"], occ), "clip_id": c["clip_id"],
                "clip_dur": c["clip_dur"], "repeat": repeat, "mode": mode}

    blind = [row(c, 0, False, "blind") for c in queue if c["clip_id"] in core]
    reps = [row(by_id[blind[i]["clip_id"]], 1, True, "blind")
            for i in _even_indices(len(blind), n_repeats)]
    assisted = [row(c, 0, False, "assisted") for c in queue
                if c["clip_id"] not in core]
    return blind + reps + assisted


def remaining(presentations, labels: dict) -> list[dict]:
    """The presentations still owed an answer, in session order.

    528 presentations is several sittings. Without this the page restarts at the first
    clip every time it is opened and the listener re-hears work they have already done;
    with it, reopening the page resumes. Order is preserved, so the repeats stay last.
    """
    return [p for p in presentations if not _committed(labels.get(p["token"]))]


def _script_json(obj) -> str:
    """JSON safe to substitute into an inline ``<script>`` block.

    ``json.dumps`` leaves ``<`` and ``>`` as themselves, so any payload field that ever
    carries free text could close the block with ``</script>``. Today the payload is
    tokens, durations and a mode, none of which can, but the substitution site is the
    same shape that was exploitable in ``blind_preference.py`` (CodeRabbit, 2026-09-06).
    ``\u003c`` and friends are ordinary JSON escapes, so the page reads back byte-identical
    strings.
    """
    return (json.dumps(obj, ensure_ascii=False)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def page_payload(presentations) -> list[dict]:
    """What the browser is allowed to know: that a clip exists and how long it is.

    Not the city, not the meeting, not the source URL, not the clip id, and not whether
    this is the second time it has been heard.
    """
    return [{"token": p["token"], "clip_dur": p["clip_dur"],
             "mode": p.get("mode", "blind")} for p in presentations]


def _committed(row) -> bool:
    if not row:
        return False
    if row.get("state") in ("excluded", "uncertain"):
        return True
    return bool((row.get("text") or "").strip())


def save_label(labels: dict, presentations, token: str, text: str, now: str,
               state: str = "transcribed", reason: str | None = None,
               listens: int | None = None) -> dict:
    """Store one answer, keeping any previous answer for the same presentation.

    A relabel is kept rather than overwritten: §4.2 asks for repeated corrections of the
    same span to be deduplicated, and that needs the history.
    """
    if state not in STATES:
        raise LabelError(f"unknown state {state!r}; frozen states are {STATES}")
    if state == "excluded" and reason not in EXCLUSION_REASONS:
        raise LabelError(
            f"an exclusion must name one of {EXCLUSION_REASONS}; got {reason!r}. The "
            "share of exclusions per reason is itself a finding.")
    if state == "transcribed" and not (text or "").strip():
        raise LabelError("an empty transcription is not a label; mark it excluded")
    by_token = {p["token"]: p for p in presentations}
    if token not in by_token:
        raise LabelError(f"{token} is not a presentation in this session")
    out = dict(labels)
    prev = out.get(token)
    row = {"text": (text or "").strip(), "state": state, "saved_at": now,
           "mode": by_token[token].get("mode", "blind")}
    if reason:
        row["reason"] = reason
    if listens is not None:
        row["listens"] = int(listens)
    if prev:
        history = list(prev.get("previous") or [])
        history.append({"text": prev.get("text"), "saved_at": prev.get("saved_at"),
                        "state": prev.get("state")})
        row["previous"] = history
    out[token] = row
    return out


def prefill(token: str, presentations, witness: dict) -> dict:
    """The witness text, BEFORE any answer, and only for an assisted presentation.

    This is the one place the blindness is deliberately spent, and it is spent on the
    clips whose labels may never be used for measurement. Asking for a blind clip's
    prefill is refused at this level rather than in the browser, because the browser is
    not the thing that has to be trustworthy.
    """
    p = {x["token"]: x for x in presentations}.get(token)
    if p is None:
        raise RevealDenied(f"{token} is not a presentation in this session")
    if p.get("mode", "blind") != "assisted":
        raise RevealDenied(
            f"{token} is a blind-core presentation; its label must be typed from the "
            "audio, not edited from the witness")
    return {"token": token, "text": witness.get(p["clip_id"], "")}


def reveal(token: str, presentations, labels: dict, witness: dict,
           target: dict) -> dict:
    """The witness and the current training target, only after an answer is committed."""
    clip = {p["token"]: p["clip_id"] for p in presentations}.get(token)
    if clip is None:
        raise RevealDenied(f"{token} is not a presentation in this session")
    row = labels.get(token)
    if not _committed(row):
        raise RevealDenied(f"no committed answer for {token} yet")
    if clip not in witness:
        raise RevealDenied(f"{clip} has no witness on file")
    return {"token": token, "label": row.get("text"), "state": row.get("state"),
            "witness_text": witness.get(clip), "current_target": target.get(clip)}


def training_rows(labels: dict, presentations, mode: str | None = None) -> list[dict]:
    """The answers that may become training data: first hearings, actually transcribed.

    A repeat is a consistency control. Letting it through would duplicate a row and
    quietly weight that clip twice in the mix.

    `mode` selects one condition. Any MEASUREMENT must pass `mode="blind"`: an assisted
    label was edited from the Soniox text and carries its errors, so pooling the two
    would make the round's own numbers a statement about Soniox. Training may use both,
    and then says so.
    """
    if mode is not None and mode not in MODES:
        raise LabelError(f"unknown mode {mode!r}; frozen modes are {MODES}")
    out = []
    for p in presentations:
        if p["repeat"]:
            continue
        if mode is not None and p.get("mode", "blind") != mode:
            continue
        row = labels.get(p["token"])
        if not row or row.get("state") != "transcribed":
            continue
        if not (row.get("text") or "").strip():
            continue
        out.append({"clip_id": p["clip_id"], "text": row["text"],
                    "mode": p.get("mode", "blind"),
                    "listens": row.get("listens"), "saved_at": row.get("saved_at")})
    return out


def self_consistency_pairs(labels: dict, presentations) -> list[tuple]:
    """(first answer, second answer, clip) for every clip heard twice.

    The disagreement between these two is the noise floor of the whole set, and every
    report that uses these labels states it. It is a cheap substitute for a second
    listener, not an equivalent: it measures how unstable one listener is, not how
    systematically they are biased.
    """
    first = {p["clip_id"]: labels.get(p["token"]) for p in presentations
             if not p["repeat"]}
    out = []
    for p in presentations:
        if not p["repeat"]:
            continue
        a, b = first.get(p["clip_id"]), labels.get(p["token"])
        if a and b:
            out.append((a.get("text"), b.get("text"), p["clip_id"]))
    return out


# ------------------------------------------------------------------ the server

PAGE = """<!doctype html><html lang="el"><meta charset="utf-8">
<title>listen-2026-08</title>
<style>
 body{font:16px/1.5 system-ui;margin:0;background:#111;color:#eee}
 main{max-width:46rem;margin:0 auto;padding:1.2rem}
 h1{font-size:1rem;font-weight:600;color:#9ab;margin:0 0 .3rem}
 audio{width:100%;margin:.4rem 0}
 textarea{width:100%;min-height:5.5rem;font:17px/1.5 system-ui;padding:.6rem;
   background:#1b1b1b;color:#eee;border:1px solid #444;border-radius:6px}
 button{font:15px system-ui;padding:.5rem .9rem;border-radius:6px;border:0;
   background:#2d6;color:#052;font-weight:600;cursor:pointer}
 button.ghost{background:#333;color:#ccc;font-weight:400}
 button kbd{opacity:.6;font-size:.75rem;margin-left:.35rem}
 .bar{display:flex;gap:.5rem;align-items:center;margin:.7rem 0;flex-wrap:wrap}
 .n{color:#789;font-variant-numeric:tabular-nums}
 .warn{color:#e94}
 .rev{margin-top:1rem;padding:.7rem;background:#181818;border-left:3px solid #46a;
   border-radius:0 6px 6px 0;white-space:pre-wrap;display:none}
 .rev h2{font-size:.72rem;color:#79b;margin:.3rem 0 .1rem;text-transform:uppercase}
 .hint{color:#667;font-size:.85rem}
 .mode{font-size:.75rem;padding:.1rem .5rem;border-radius:99px;font-weight:600}
 .mode.blind{background:#243;color:#7d9}
 .mode.assisted{background:#332a17;color:#db8}
 table.cheat{border-collapse:collapse;width:100%;font-size:.85rem;margin:.2rem 0 0}
 table.cheat td{padding:.25rem .5rem;border-top:1px solid #222;vertical-align:top}
 table.cheat td:first-child{color:#fd7;white-space:nowrap;font-weight:600}
 table.cheat td:last-child{color:#667}
 details{background:#161616;border-radius:6px;padding:.5rem .7rem;margin:.8rem 0 0}
 summary{cursor:pointer;color:#9ab;font-size:.9rem}
 label.chk{color:#667;font-size:.85rem;user-select:none;cursor:pointer}
</style>
<main>
 <h1>Γράψε ό,τι ακούς.</h1>
 <div class="bar"><span class="n" id="pos"></span>
   <span class="mode" id="mode"></span>
   <span class="hint" id="dur"></span><span class="warn" id="listens"></span>
   <span class="hint" id="eta"></span></div>
 <audio id="a" controls preload="none"></audio>
 <textarea id="t" placeholder="…" autofocus></textarea>
 <div class="bar">
   <button id="save">Αποθήκευση<kbd>Ctrl+Enter</kbd></button>
   <button class="ghost" id="unsure">Αβέβαιο<kbd>Alt+1</kbd></button>
   <button class="ghost" id="unclear">Ακατάληπτο<kbd>Alt+2</kbd></button>
   <button class="ghost" id="overlap">Επικάλυψη<kbd>Alt+3</kbd></button>
   <button class="ghost" id="noise">Θόρυβος<kbd>Alt+4</kbd></button>
   <span class="hint" id="msg"></span>
 </div>
 <div class="bar">
   <label class="chk"><input type="checkbox" id="showrev"> δείξε τον μάρτυρα μετά
     (σταματάει η αυτόματη ροή)</label>
   <span class="hint">Alt+R = ξανάκουσέ το</span>
 </div>
 <table class="cheat">
  <tr><td>Ακούω καθαρά</td><td>Γράψ' το όπως ειπώθηκε — ψευδοεκκινήσεις και επαναλήψεις
    ΜΕΣΑ, «ε/εμ/μμ» ΕΞΩ, αριθμοί με λέξεις. → γίνεται στόχος εκπαίδευσης.</td></tr>
  <tr><td>Ακούω τα μισά</td><td>Γράψε ό,τι έπιασες και πάτα <b>Αβέβαιο</b> — το κείμενο
    <b>κρατιέται</b>. → δεν εκπαιδεύει, μετράει το ταβάνι.</td></tr>
  <tr><td>Μιλάνε μαζί</td><td><b>Επικάλυψη</b>, χωρίς να γράψεις. → μετράει πόσο μας
    κοστίζει η ταυτόχρονη ομιλία.</td></tr>
  <tr><td>Δεν βγαίνει λέξη</td><td><b>Ακατάληπτο</b>. → δείχνει τι είναι όριο της
    ηχογράφησης και όχι του μοντέλου.</td></tr>
  <tr><td>Θόρυβος / μουσική / σιωπή</td><td><b>Θόρυβος</b>. → βγαίνει από τα δεδομένα
    πριν μολύνει την εκπαίδευση.</td></tr>
  <tr><td>Όνομα ή αριθμός που δεν είμαι σίγουρος</td><td>Γράψε ό,τι ακούς, μη μαντεύεις
    τη σωστή γραφή, και πάτα <b>Αβέβαιο</b>.</td></tr>
 </table>
 <details><summary>Γιατί έχει σημασία (30 δευτ.)</summary>
  <p class="hint">Το μοντέλο έμαθε να <b>σιωπά</b> στα δύσκολα, επειδή το δημοσιευμένο
  κείμενο σιωπούσε εκεί. Κάθε κλιπ που γράφεις είναι ένα σημείο όπου του λέμε «εδώ
  ειπώθηκε κάτι, γράψ' το». Τα <b>120 τυφλά</b> είναι και το μοναδικό υλικό με το οποίο
  κρίνουμε το Soniox — γι' αυτό εκεί δεν βλέπεις τίποτα πριν γράψεις. Στα υπόλοιπα 601
  διορθώνεις έτοιμο κείμενο και πάει μόνο για εκπαίδευση.</p>
  <p class="hint">Ποτέ μαντεψιά: μια μαντεμένη λέξη γίνεται λάθος στόχος και το μοντέλο
  τη μαθαίνει. Το «Αβέβαιο» δεν είναι αποτυχία — είναι μέτρηση.</p>
 </details>
 <div class="rev" id="rev"></div>
</main>
<script>
const Q = __DATA__, MAXL = __MAXL__; let i = 0, plays = 0, busy = false;
const $ = s => document.querySelector(s);
const BTNS = ['#save','#unsure','#unclear','#overlap','#noise'];
function lock(on){ busy = on; BTNS.forEach(b => $(b).disabled = on); }
function eta(){
  const left = Q.length - i;
  const secs = Q.slice(i).reduce((a,c) => a + c.clip_dur, 0) * 3.5;
  return left ? '≈' + Math.max(1, Math.round(secs/60)) + ' λ. ακόμα' : '';
}
function show(){
  const c = Q[i]; plays = 0;
  $('#pos').textContent = Q.length ? (i+1) + ' / ' + Q.length : 'τέλος';
  $('#dur').textContent = c ? c.clip_dur.toFixed(1) + ' δευτ.' : '';
  $('#eta').textContent = eta();
  $('#listens').textContent = '';
  $('#rev').style.display='none'; $('#rev').innerHTML='';
  $('#t').value=''; $('#msg').textContent='';
  $('#a').src = c ? '/clip/' + c.token + '.wav' : '';
  const badge = $('#mode');
  badge.className = 'mode ' + (c ? c.mode : '');
  badge.textContent = c ? (c.mode === 'assisted' ? 'διόρθωση Soniox' : 'τυφλό') : '';
  lock(!c);
  // Autoplay: the browser allows it once the page has had a user gesture, which the
  // first save provides. One less click on every clip after that.
  if(c) $('#a').play().catch(() => {});
  if(c && c.mode === 'assisted'){
    const tok = c.token;
    fetch('/prefill/' + tok).then(r => r.ok ? r.json() : null).then(d => {
      if(d && Q[i] && Q[i].token === tok && !$('#t').value) $('#t').value = d.text || '';
    }).catch(() => {});
  }
  $('#t').focus();
}
$('#a').addEventListener('play', () => {
  plays++;
  $('#listens').textContent = plays > MAXL
    ? 'ακρόαση ' + plays + ' — μετά από ' + MAXL + ' σήμανέ το αβέβαιο και προχώρα' : '';
});
async function commit(state, reason){
  if(busy) return;
  const c = Q[i]; if(!c) return;
  // An exclusion has no text by definition. `uncertain` KEEPS what was typed: half a
  // sentence heard is evidence, and throwing it away was costing exactly the hard clips.
  const tok = c.token, text = state === 'excluded' ? '' : $('#t').value;
  const wantRev = $('#showrev').checked;
  lock(true);
  try{
    const r = await fetch('/label', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({token: tok, text, state, reason, listens: plays})});
    if(!r.ok){ $('#msg').textContent = 'δεν αποθηκεύτηκε: ' + await r.text(); return; }
    if(!wantRev){ lock(false); next(); return; }
    const rev = await fetch('/reveal/' + tok);
    if(rev.ok && Q[i] && Q[i].token === tok){
      const d = await rev.json();
      $('#rev').innerHTML = '<h2>μάρτυρας (soniox)</h2>' + esc(d.witness_text||'—') +
        '<h2>τρέχων στόχος εκπαίδευσης</h2>' + esc(d.current_target||'—');
      $('#rev').style.display='block';
    }
    $('#msg').textContent = 'αποθηκεύτηκε · Enter για το επόμενο';
  } finally {
    lock(false);
    if(document.activeElement && document.activeElement.blur) document.activeElement.blur();
  }
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function next(){ if(i < Q.length-1){ i++; show(); } else { $('#msg').textContent='τέλος'; } }
$('#save').onclick = () => commit('transcribed', null);
$('#unsure').onclick = () => commit('uncertain', null);
$('#unclear').onclick = () => commit('excluded', 'unintelligible');
$('#overlap').onclick = () => commit('excluded', 'overlap');
$('#noise').onclick = () => commit('excluded', 'noise_or_music');
document.addEventListener('keydown', e => {
  if(e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { commit('transcribed', null); e.preventDefault(); return; }
  if(e.altKey){
    const map = {'1':['uncertain',null], '2':['excluded','unintelligible'],
                 '3':['excluded','overlap'], '4':['excluded','noise_or_music']};
    if(map[e.key]){ commit(map[e.key][0], map[e.key][1]); e.preventDefault(); return; }
    if(e.key === 'r' || e.key === 'ρ'){ $('#a').currentTime = 0; $('#a').play(); e.preventDefault(); return; }
  }
  if(busy) return;
  if(e.key === 'Enter' && document.activeElement !== $('#t')) next();
});
show();
</script></html>"""


def _read(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def make_handler(presentations, clips_dir: Path, labels_path: Path,
                 witness: dict, target: dict):
    state = {"labels": _read(labels_path, {})}
    by_token = {p["token"]: p["clip_id"] for p in presentations}

    def page() -> str:
        todo = remaining(presentations, state["labels"])
        return (PAGE
                .replace("__DATA__", _script_json(page_payload(todo)))
                .replace("__DONE__", str(len(presentations) - len(todo)))
                .replace("__TOTAL__", str(len(presentations)))
                .replace("__MAXL__", str(MAX_LISTENS)))

    class H(http.server.BaseHTTPRequestHandler):
        timeout = SOCKET_TIMEOUT_SEC     # a slow client may not hold the only thread

        def log_message(self, *a):
            pass                     # never log: the paths carry meeting ids

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")

        def do_GET(self):
            p = urllib.parse.urlparse(self.path).path
            if p in ("/", "/index.html"):
                return self._send(200, page().encode(), "text/html; charset=utf-8")
            m = re.fullmatch(r"/clip/([0-9a-f]{16})\.wav", p)
            if m:
                cid = by_token.get(m.group(1))
                if not cid or not is_safe_clip_id(cid):
                    return self.send_error(404)
                f = clips_dir / f"{cid}.wav"
                if not f.is_file():
                    return self.send_error(404)
                return self._send(200, f.read_bytes(), "audio/wav")
            m = re.fullmatch(r"/prefill/([0-9a-f]{16})", p)
            if m:
                try:
                    return self._json(200, prefill(m.group(1), presentations, witness))
                except RevealDenied as e:
                    return self._json(403, {"error": str(e)})
            m = re.fullmatch(r"/reveal/([0-9a-f]{16})", p)
            if m:
                try:
                    return self._json(200, reveal(m.group(1), presentations,
                                                  state["labels"], witness, target))
                except RevealDenied as e:
                    return self._json(403, {"error": str(e)})
            return self.send_error(404)

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/label":
                return self.send_error(404)
            body = {}
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                if not 0 < n <= MAX_BODY_BYTES:
                    raise LabelError(
                        f"Content-Length must be between 1 and {MAX_BODY_BYTES}")
                raw = self.rfile.read(n)
                if len(raw) != n:
                    raise LabelError("short body")
                body = json.loads(raw.decode("utf-8"))
                if not isinstance(body, dict):
                    raise LabelError("body must be a JSON object")
                now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                candidate = save_label(
                    state["labels"], presentations, body.get("token"),
                    body.get("text", ""), now,
                    state=body.get("state") or "transcribed",
                    reason=body.get("reason"), listens=body.get("listens"))
            except (LabelError, ValueError, TypeError, AttributeError,
                    UnicodeDecodeError) as e:
                return self._send(400, str(e).encode(), "text/plain; charset=utf-8")
            # The label reaches server state only after it is on disk. `/reveal` opens on
            # a committed answer, so a visible-but-unwritten label would spend the
            # blindness of a clip whose answer was lost -- and the next successful save
            # would carry the lost answer along in `previous`.
            tmp = labels_path.with_suffix(".json.tmp")
            try:
                tmp.write_text(json.dumps(candidate, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, labels_path)
            except OSError as e:
                tmp.unlink(missing_ok=True)
                return self._send(500, f"not saved: {e}".encode(),
                                  "text/plain; charset=utf-8")
            state["labels"] = candidate
            return self._json(200, {"saved": body.get("token"),
                                    "n": len(state["labels"])})

    return H


def _tailnet_addresses() -> frozenset[str]:
    """The addresses this machine actually holds on its tailscale interface.

    Asked of tailscale rather than inferred from the address, because 100.64.0.0/10 is
    carrier-grade NAT: an ISP-assigned address on a real interface falls inside it and
    would pass a range test while putting the audio on the carrier network. Fails closed
    -- no tailscale, no answer, nothing outside loopback is bindable.
    """
    try:
        out = subprocess.run(["tailscale", "ip"], capture_output=True, text=True,
                             timeout=5)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    if out.returncode != 0:
        return frozenset()
    return frozenset(line.strip() for line in out.stdout.splitlines() if line.strip())


def _bindable(host: str, tailnet: "frozenset[str] | set[str] | None" = None) -> bool:
    """Loopback, or an address this machine holds on the tailnet. Never anything else.

    There is no authentication on this server: the presentation tokens are opacity, not
    credentials. The binding is what keeps the audio off other networks, so it is
    checked rather than trusted -- and membership of the tailnet range is not the check,
    because that range is shared with carrier-grade NAT. The address must be one
    tailscale reports for this machine.
    """
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    if ip not in ipaddress.ip_network("100.64.0.0/10"):
        return False
    addrs = _tailnet_addresses() if tailnet is None else tailnet
    for a in addrs:
        try:
            if ipaddress.ip_address(a.strip()) == ip:
                return True
        except ValueError:
            continue
    return False


def main() -> None:  # pragma: no cover - operational entry point
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if not _bindable(a.host):
        raise SystemExit(
            f"refused: {a.host!r}. This server has no authentication and serves council "
            "audio, so it binds loopback, or an address `tailscale ip` reports for this "
            "machine, and nothing else. 0.0.0.0 would put the audio on every network "
            "this machine joins -- the same PII category as the 2026-07-21 history "
            "purge -- and a bare 100.64.0.0/10 range test would accept a carrier-grade "
            "NAT address on a real interface.")
    q = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))["clips"]
    pres = session_order(q)
    handler = make_handler(pres, CLIPS_DIR, LABELS_PATH,
                           _read(WITNESS_PATH, {}), _read(TARGET_PATH, {}))
    todo = len(remaining(pres, _read(LABELS_PATH, {})))
    print(f"{len(q)} clips + {len(pres) - len(q)} repeats, {todo} still to do · "
          f"http://{a.host}:{a.port}", flush=True)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((a.host, a.port), handler) as s:
        s.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
