"""gap2 audio-verification pass — pre-verify review-queue items with Soniox.

For each id in ui/src/lib/server/state/gap2-ids.json:
  1. fetch the review row (cached under ~/.cache/oc-public/gap-verify/api/)
  2. compute ADDED tokens (final_after_text minus initial_before_text) with the
     gap-mining normalization (NFD strip marks, lowercase, \\w+, minus fillers;
     difflib opcodes, added = inserts + net surplus of replaces)
  3. extract exact [start,end] and padded [start-3, end+3] clips (mono 16k wav)
  4. Soniox-transcribe the PADDED clip (file_transcribe.py --lang el); if the
     padded match passes (>=70%), a SECOND call on the exact clip localizes it
     (the wrapper emits no word timestamps)
  5. verdict: VERIFIED >=70% found / BOUNDARY (padded yes, exact no) /
     MIDDLE 40-70% / REJECT <40%. Fuzzy: exact normalized match, or
     Damerau-Levenshtein distance 1 for tokens of length>=5.

Resumable: manifest.jsonl is appended after every item; per-item caches under
api/ and asr/ make reruns free. Run:  python3 eval/gap2_verify_audio.py
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDS_PATH = ROOT / "ui/src/lib/server/state/gap2-ids.json"
WORK = Path.home() / ".cache/oc-public/gap-verify"
API = WORK / "api"
CLIPS = WORK / "clips"
ASR = WORK / "asr"
MANIFEST = WORK / "manifest.jsonl"
SONIOX_DIR = Path("/home/harold/projects/soniox-tools")
SONIOX_PY = SONIOX_DIR / ".venv" / "bin" / "python"
REVIEW_BASE = "https://79-76-114-184.sslip.io/api/review/group/"
PAD = 3.0
MARKER = "===== TRANSCRIPT ====="

# --- normalization: identical to gap-mining-script-2026-08-12.py ---
FILLER_RE = re.compile(r"^(ε{2,}|μ{2,}|α{2,}|ο{3,}|χμ+|εμ+|μχ+|χ{2,})$")
EXTRA_FILLERS = {"ε"}  # task spec lists single-ε too
TOK_RE = re.compile(r"\w+")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def ntoks(text: str) -> list[str]:
    out = []
    for m in TOK_RE.finditer(text or ""):
        nt = norm(m.group())
        if nt and not FILLER_RE.match(nt) and nt not in EXTRA_FILLERS:
            out.append(nt)
    return out


def added_tokens(before: str, after: str) -> list[str]:
    """Added word sequences: inserts + net surplus (tail) of replaces."""
    bn, an = ntoks(before), ntoks(after)
    sm = difflib.SequenceMatcher(a=bn, b=an, autojunk=False)
    added: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            added.extend(an[j1:j2])
        elif tag == "replace":
            net = (j2 - j1) - (i2 - i1)
            if net > 0:
                added.extend(an[j2 - net:j2])
    return added


# --- fuzzy match ---
def dl_leq1(a: str, b: str) -> bool:
    """Damerau-Levenshtein distance <= 1 (adjacent transposition counts as 1)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if a == b:
        return True
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        if len(diff) == 2 and diff[1] == diff[0] + 1:
            i = diff[0]
            return a[i] == b[i + 1] and a[i + 1] == b[i]
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    # b is a with one insertion
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def found_frac(added: list[str], transcript: str) -> tuple[float, int]:
    ttoks = set(ntoks(transcript))
    tlong = [t for t in ttoks if len(t) >= 4]  # candidates for fuzzy vs len>=5 added
    n_found = 0
    for tok in added:
        if tok in ttoks:
            n_found += 1
        elif len(tok) >= 5 and any(dl_leq1(tok, t) for t in tlong):
            n_found += 1
    return (n_found / len(added) if added else 0.0), n_found


# --- audio ---
def _ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, errors="replace", timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None


def extract_clip(url: str, start: float, end: float, out: Path) -> dict:
    """Same recipe as next_batch_step2_transcribe.extract_clip, explicit bounds."""
    if out.exists() and (_ffprobe_duration(out) or 0) > 0.2:
        return {"extract_ok": True, "error": None, "expected_dur": end - start,
                "cached": True}
    dur = end - start
    cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", url,
           "-t", f"{dur:.3f}", "-ac", "1", "-ar", "16000", "-f", "wav", str(out)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           timeout=180)
    except subprocess.TimeoutExpired:
        return {"extract_ok": False, "error": "ffmpeg_timeout", "expected_dur": dur}
    if p.returncode != 0 or not out.exists():
        return {"extract_ok": False, "expected_dur": dur,
                "error": f"ffmpeg_rc={p.returncode}: " + p.stderr.strip()[-300:]}
    decoded = _ffprobe_duration(out)
    ok = decoded is not None and (dur * 0.5 - 0.2) <= decoded <= (dur + 2.0)
    return {"extract_ok": bool(ok), "expected_dur": round(dur, 3),
            "decoded_dur": None if decoded is None else round(decoded, 3),
            "error": None if ok else
            f"duration_mismatch expected={dur:.2f} decoded={decoded}"}


def transcribe(wav: Path, expected_dur: float) -> dict:
    """Same as next_batch_step2_transcribe.transcribe (marker parse, temp key)."""
    timeout = max(90.0, expected_dur * 5 + 40)
    try:
        p = subprocess.run(
            [str(SONIOX_PY), "file_transcribe.py", str(wav), "--lang", "el"],
            capture_output=True, text=True, errors="replace", timeout=timeout,
            cwd=str(SONIOX_DIR))
    except subprocess.TimeoutExpired:
        return {"asr_ok": False, "soniox_text": "", "error": "asr_timeout"}
    if MARKER not in p.stdout:
        return {"asr_ok": False, "soniox_text": "",
                "error": f"no_marker rc={p.returncode}: "
                + (p.stderr.strip()[-300:] or p.stdout.strip()[-300:])}
    return {"asr_ok": True, "soniox_text": p.stdout.split(MARKER, 1)[1].strip(),
            "error": None}


def transcribe_cached(wav: Path, expected_dur: float, cache_key: str) -> dict:
    jp = ASR / f"{cache_key}.json"
    if jp.exists():
        try:
            r = json.loads(jp.read_text())
            if r.get("asr_ok"):
                return r
        except Exception:
            pass
    r = transcribe(wav, expected_dur)
    if not r.get("asr_ok"):          # retry once
        time.sleep(5)
        r = transcribe(wav, expected_dur)
    jp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return r


# --- review API ---
def fetch_row(uid: str) -> dict | None:
    jp = API / f"{uid}.json"
    if jp.exists():
        try:
            return json.loads(jp.read_text())
        except Exception:
            pass
    for attempt in range(2):
        try:
            with urllib.request.urlopen(REVIEW_BASE + uid, timeout=60) as r:
                d = json.loads(r.read())
            jp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            return d
        except Exception:
            if attempt == 0:
                time.sleep(3)
    return None


def trim_local(src: Path, offset: float, dur: float, out: Path) -> dict:
    """Cut the exact span out of the already-downloaded padded wav (no HTTP)."""
    if out.exists() and (_ffprobe_duration(out) or 0) > 0.2:
        return {"extract_ok": True, "error": None, "expected_dur": dur}
    cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{offset:.3f}", "-i", str(src),
           "-t", f"{dur:.3f}", "-ac", "1", "-ar", "16000", "-f", "wav", str(out)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           timeout=60)
    except subprocess.TimeoutExpired:
        return {"extract_ok": False, "error": "ffmpeg_timeout", "expected_dur": dur}
    if p.returncode != 0 or not out.exists():
        return {"extract_ok": False, "expected_dur": dur,
                "error": f"ffmpeg_rc={p.returncode}: " + p.stderr.strip()[-300:]}
    return {"extract_ok": True, "error": None, "expected_dur": round(dur, 3)}


def process_one(uid: str) -> tuple[dict, float]:
    """Full per-item job. Returns (manifest record, audio seconds sent to Soniox)."""
    rec: dict = {"id": uid, "tier": None, "found_frac": None,
                 "found_frac_exact": None, "added_tokens": None,
                 "n_added": None, "soniox_text": None,
                 "clip_exact": None, "clip_padded": None,
                 "error": None, "flags": []}
    audio_s = 0.0
    try:
        row = fetch_row(uid)
        if row is None:
            rec["error"] = "api_fetch_failed"
            raise StopIteration
        start, end = float(row["start"]), float(row["end"])
        url = row.get("audio_url") or row.get("audio_cdn_url")
        added = added_tokens(row.get("initial_before_text") or "",
                             row.get("final_after_text") or "")
        rec["added_tokens"] = added
        rec["n_added"] = len(added)
        if not added:
            rec["tier"] = "NO_ADDED"
            rec["flags"].append("no_added_tokens")
            raise StopIteration
        if not url or end <= start or (end - start) > 600:
            rec["error"] = f"bad_span_or_url start={start} end={end}"
            raise StopIteration

        pstart = max(0.0, start - PAD)
        wav_p = CLIPS / f"{uid}.pad.wav"
        wav_e = CLIPS / f"{uid}.exact.wav"
        rec["clip_padded"] = str(wav_p)
        rec["clip_exact"] = str(wav_e)
        ext_p = extract_clip(url, pstart, end + PAD, wav_p)
        if not ext_p["extract_ok"]:
            rec["error"] = "extract_padded: " + str(ext_p["error"])
            raise StopIteration
        audio_s += ext_p["expected_dur"]
        tr_p = transcribe_cached(wav_p, ext_p["expected_dur"], f"{uid}.pad")
        if not tr_p.get("asr_ok"):
            rec["error"] = "asr_padded: " + str(tr_p.get("error"))
            raise StopIteration
        rec["soniox_text"] = tr_p["soniox_text"]
        frac, n_found = found_frac(added, tr_p["soniox_text"])
        rec["found_frac"] = round(frac, 4)

        n_add = len(added)
        passed = (frac >= 0.70) or (n_add <= 2 and n_found >= 1)
        if passed:
            # localize: exact-span transcript, cut locally from the padded wav
            ext_e = trim_local(wav_p, start - pstart, end - start, wav_e)
            if ext_e["extract_ok"]:
                audio_s += ext_e["expected_dur"]
                tr_e = transcribe_cached(wav_e, ext_e["expected_dur"],
                                         f"{uid}.exact")
                if tr_e.get("asr_ok"):
                    fe, _ = found_frac(added, tr_e["soniox_text"])
                    rec["found_frac_exact"] = round(fe, 4)
                    rec["tier"] = "BOUNDARY" if fe < 0.40 else "VERIFIED"
                else:
                    rec["tier"] = "VERIFIED"
                    rec["flags"].append("needs_boundary_check")
            else:
                rec["tier"] = "VERIFIED"
                rec["flags"].append("needs_boundary_check")
        elif frac >= 0.40:
            rec["tier"] = "MIDDLE"
            rec["flags"].append("uncertain")
        else:
            rec["tier"] = "REJECT"
    except StopIteration:
        pass
    except Exception as e:  # never lose the run to one item
        rec["error"] = f"unexpected: {type(e).__name__}: {e}"
    return rec, audio_s


def main() -> None:
    for d in (API, CLIPS, ASR):
        d.mkdir(parents=True, exist_ok=True)
    ids_path, limit, workers = IDS_PATH, None, 3
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
        elif arg.startswith("--ids="):
            ids_path = Path(arg.split("=", 1)[1])
        elif arg.startswith("--workers="):
            workers = int(arg.split("=", 1)[1])
    ids = json.loads(ids_path.read_text())

    done: set[str] = set()
    if MANIFEST.exists():
        for l in MANIFEST.read_text().splitlines():
            try:
                r = json.loads(l)
                if r.get("tier") is not None or r.get("error"):
                    done.add(r["id"])
            except Exception:
                pass
    todo = [i for i in ids if i not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"[verify] {len(ids)} ids, {len(done)} done, {len(todo)} to go", flush=True)

    t0 = time.time()
    audio_s = 0.0
    n_done = 0
    lock = threading.Lock()
    with MANIFEST.open("a", encoding="utf-8") as mf, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, uid): uid for uid in todo}
        for fut in as_completed(futs):
            try:
                rec, a_s = fut.result()
            except Exception as e:
                rec, a_s = {"id": futs[fut], "tier": None,
                            "error": f"worker: {e}"}, 0.0
            with lock:
                n_done += 1
                audio_s += a_s
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
                if n_done % 10 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    print(f"[{n_done}/{len(todo)}] {el/60:.1f}m "
                          f"rate={n_done/el*60:.1f}/min "
                          f"ETA={(len(todo)-n_done)/(n_done/el)/60:.0f}m "
                          f"audio={audio_s/60:.0f}min", flush=True)
    print("[verify] done", flush=True)


if __name__ == "__main__":
    main()
