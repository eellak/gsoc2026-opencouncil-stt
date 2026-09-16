#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full deletion audit: all 203 clips, blind, resumable, local only.

This supersedes the 36-clip pilot (listen_server_small.py). Those files are left
untouched; nothing here reads or writes them.

WHY LOCAL. The benchmark's TEXT is published without auth, so quoting it costs
nothing. The AUDIO is a different act: re-hosting council speech under our own URL
is a new copy of the same PII category the 2026-07-21 purge removed from git. It
stays on this machine, and the verdicts -- which carry free-text comments that can
quote the transcript -- stay under $SC, never in the repo.

WHAT THE JUDGE SEES. One 18 s clip, the context either side, and the words the
fusion deleted. Nothing else. `oracle`, `kind`, `cat`, `rule`, `city`, `meeting`
and the clip FILENAME are never sent to the browser at all -- not hidden in the
DOM, not served and ignored. Blindness that depends on client-side discipline is
not blindness. Clips are addressed by an opaque id because the filename carries a
meeting id. Analysis joins the verdicts back to `oracle` by filename, server-side.

WHAT IT DOES NOT SEE. The deletion is HIGHLIGHTED, so this measures agreement with
a proposed deletion, not independent transcription. And the clip position is
approximate: cutclips.py centres an 18 s window on (column position / columns) x
window duration, which is a crude proxy because speakers do not talk at a constant
rate. The page says so.

ORDER. Fixed seed 20260909, and the realized order plus a fingerprint of the
presented text are written into the verdict file, so a run is reproducible and a
regenerated sample cannot be silently mixed into an old set of answers.

    python3 deletion_audit_server.py     ->  http://127.0.0.1:8772
"""
import fcntl
import hashlib
import http.server
import json
import os
import re
import secrets
import subprocess
import socketserver
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
DATA_DIR = Path(os.environ.get("AUDIT_DATA", SC / "chooser-2026-08-25"))
CLIPS = Path(os.environ.get("AUDIT_CLIPS", SC / "deletion_clips"))
SAMPLE = DATA_DIR / os.environ.get("AUDIT_SAMPLE", "listen_sample.json")
VERDICTS = DATA_DIR / os.environ.get("AUDIT_VERDICTS", "deletion_audit_verdicts.json")

SEED = int(os.environ.get("AUDIT_SEED", "20260909"))
SCHEMA = "deletion-audit-v1"
PORT = int(os.environ.get("PORT", "8772"))
HOST = os.environ.get("HOST", "127.0.0.1")

# An allowlist, not a wildcard veto. Refusing only 0.0.0.0 still leaves every LAN
# address of this machine bindable, and council audio on the LAN is the thing the
# rule exists to prevent. Loopback is the default; the tailnet address is the one
# non-loopback exception, because a tailnet is device-authenticated and encrypted
# and it is how Harold reaches this machine from the laptop.
ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost", "100.123.255.50"}

# MagicDNS names resolve to the tailnet address, so a browser that reaches the
# socket still sends `minipc` or `minipc.tail11f3d4.ts.net` as its Host header and
# the allowlist above would 403 it. Asked at runtime rather than hardcoded, because
# a tailnet rename would otherwise lock Harold out with a blank 403 and no clue why.
# Names only: this widens which Host headers are ACCEPTED, never what we bind to.
def _tailnet_names():
    try:
        out = subprocess.run(["tailscale", "status", "--json"], capture_output=True,
                             text=True, timeout=5)
        if out.returncode != 0:
            return set()
        st = json.loads(out.stdout)
        dns = (st.get("Self") or {}).get("DNSName") or ""
        fqdn = dns.rstrip(".")
        return {n for n in (fqdn, fqdn.split(".")[0]) if n}
    except Exception:
        return set()


ALLOWED_HOSTS |= _tailnet_names()

MAX_BODY = 8192
MAX_NOTE = 2000
TOKEN = secrets.token_urlsafe(24)

# ---------------------------------------------------------------- deck

def _load_deck():
    import random
    rows = [o for o in json.loads(SAMPLE.read_text(encoding="utf-8")) if o.get("clip")]
    if not rows:
        raise SystemExit(f"άρνηση: κανένα κλιπ στο {SAMPLE}")
    seen = {o["clip"] for o in rows}
    if len(seen) != len(rows):
        raise SystemExit("άρνηση: διπλά ονόματα κλιπ -- το filename δεν είναι κλειδί")
    random.Random(SEED).shuffle(rows)
    return rows


def _cid(clip):
    return hashlib.blake2s(clip.encode("utf-8"), digest_size=8).hexdigest()


def _fingerprint(deck):
    """Covers what the judge is shown, not just which files exist. A regenerated
    sample with the same filenames but different context is a different question."""
    h = hashlib.sha256()
    h.update(f"{SCHEMA}|{SEED}|{len(deck)}".encode("utf-8"))
    for o in deck:
        h.update(json.dumps([o["clip"], o["L"], o["R"], o["lost"]],
                            ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


DECK = _load_deck()
FINGERPRINT = _fingerprint(DECK)
BY_CID = {_cid(o["clip"]): o for o in DECK}
if len(BY_CID) != len(DECK):
    raise SystemExit("άρνηση: σύγκρουση opaque id")
ORDER = [o["clip"] for o in DECK]

# What the browser gets. Nothing else. Adding a field here is a decision to leak it.
CLIENT = [{"cid": _cid(o["clip"]), "L": o["L"], "R": o["R"], "lost": o["lost"]}
          for o in DECK]

# ---------------------------------------------------------------- state

def _refuse_if_in_a_repo(path):
    """`never in git` has to be enforced, not merely defaulted. AUDIT_VERDICTS is
    an env var, and an env var pointed at the repo would put comments that quote
    the transcript under version control."""
    d = path.resolve().parent
    for cand in [d, *d.parents]:
        if (cand / ".git").exists():
            raise SystemExit(f"άρνηση: {path} βρίσκεται μέσα στο git repo {cand}. "
                             f"Οι απαντήσεις περιέχουν κείμενο πρακτικών.")


_refuse_if_in_a_repo(VERDICTS)

_lock = threading.Lock()
_FLOCK = None


def _claim():
    """One writer per verdict file, process-wide. A threading.Lock spans threads,
    not processes: two servers on two ports over one file each hold their own
    STATE, and the second to save replaces the first one's answers with its stale
    snapshot. os.replace makes that atomic, not correct."""
    global _FLOCK
    _FLOCK = open(str(VERDICTS) + ".lock", "w")
    try:
        fcntl.flock(_FLOCK.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"άρνηση: άλλη διεργασία κρατά ήδη το {VERDICTS}. "
                         f"Κλείσε τον άλλο server.")


def _blank():
    return {"schema": SCHEMA, "seed": SEED, "sample": SAMPLE.name, "n": len(DECK),
            "fingerprint": FINGERPRINT, "order": ORDER, "verdicts": {}}


def _validate(d):
    """Refuse to start on anything we cannot prove belongs to this deck. Silently
    starting an empty audit on top of an unreadable file is how answers vanish."""
    if not isinstance(d, dict):
        return "δεν είναι αντικείμενο JSON"
    if d.get("schema") != SCHEMA:
        return f"schema {d.get('schema')!r} != {SCHEMA!r}"
    if d.get("seed") != SEED:
        return f"seed {d.get('seed')!r} != {SEED!r}"
    if d.get("fingerprint") != FINGERPRINT:
        return ("το fingerprint του δείγματος διαφέρει -- το listen_sample.json ή η "
                "σειρά άλλαξε από τότε που γράφτηκαν αυτές οι απαντήσεις")
    if d.get("order") != ORDER:
        return "η αποθηκευμένη σειρά διαφέρει από τη σειρά αυτού του deck"
    v = d.get("verdicts")
    if not isinstance(v, dict):
        return "το πεδίο verdicts δεν είναι αντικείμενο"
    known = set(ORDER)
    for clip, rec in v.items():
        if clip not in known:
            return f"άγνωστο κλιπ στις απαντήσεις: {clip!r}"
        if not isinstance(rec, dict):
            return f"η εγγραφή για {clip!r} δεν είναι αντικείμενο"
        a = rec.get("a")
        # `type(a) is int` on purpose: True == 1 in Python and a boolean is not an answer.
        if a is not None and (type(a) is not int or not 1 <= a <= 4):
            return f"μη έγκυρη απάντηση για {clip!r}: {a!r}"
        if not isinstance(rec.get("note", ""), str):
            return f"μη έγκυρο σχόλιο για {clip!r}"
    return None


def _load_state():
    if not VERDICTS.exists():
        return _blank()
    try:
        d = json.loads(VERDICTS.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"άρνηση: το {VERDICTS} δεν διαβάζεται ({e}). "
                         f"Μετακίνησέ το ή διόρθωσέ το -- δεν το αντικαθιστώ.")
    bad = _validate(d)
    if bad:
        raise SystemExit(f"άρνηση: το {VERDICTS} δεν ταιριάζει με αυτό το deck: {bad}. "
                         f"Μετακίνησέ το -- δεν το αντικαθιστώ.")
    return d


STATE = _load_state()


def _write(candidate):
    """Disk first, memory second. If /verdicts could report a save that never
    reached disk, 'resumable' is a claim rather than a property."""
    tmp = VERDICTS.with_name(VERDICTS.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    blob = json.dumps(candidate, ensure_ascii=False, indent=1)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, VERDICTS)
    try:
        dfd = os.open(str(VERDICTS.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def _save(clip, patch):
    with _lock:
        candidate = json.loads(json.dumps(STATE))
        rec = dict(candidate["verdicts"].get(clip, {}))
        rec.update(patch)
        rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        candidate["verdicts"][clip] = rec
        _write(candidate)
        STATE["verdicts"] = candidate["verdicts"]
        return sum(1 for r in STATE["verdicts"].values() if r.get("a"))


def _client_state():
    out = {}
    for clip, rec in STATE["verdicts"].items():
        e = {}
        if rec.get("a") is not None:
            e["a"] = rec["a"]
        if rec.get("note"):
            e["note"] = rec["note"]
        if e:
            out[_cid(clip)] = e
    return out


# ---------------------------------------------------------------- page

TEMPLATE = (HERE / "deletion_audit.html").read_text(encoding="utf-8")


def _js(obj):
    # `</script>` inside a JSON string literal ends the container element, no matter
    # what the JSON says. Escaping `<` is the whole fix.
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c")


def render():
    return (TEMPLATE
            .replace("__DATA__", _js(CLIENT))
            .replace("__STATE__", _js(_client_state()))
            .replace("__TOKEN__", TOKEN))


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "audit"
    sys_version = ""

    def log_message(self, *a):
        pass                      # never log; the paths carry meeting ids

    def log_error(self, *a):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            self.close_connection = True

    # A traceback prints the request line, and the request line names a meeting.
    def handle_error(self, *a):
        pass

    def _host_ok(self):
        h = self.headers.get("Host", "")
        name = h.rsplit(":", 1)[0].strip("[]") if h else ""
        return name in ALLOWED_HOSTS

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=(),
              close=False):
        """`close=True` for any rejection that returns BEFORE reading the request
        body. Keep-alive would leave those unread bytes in the socket and the next
        request on the same connection would start mid-JSON."""
        self.send_response(code)
        if close:
            self.close_connection = True
            self.send_header("Connection", "close")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if not self._host_ok():
            self._send(403, b"bad host"); return
        p = urllib.parse.urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._send(200, render().encode("utf-8"), "text/html; charset=utf-8")
            return
        m = re.fullmatch(r"/clip/([0-9a-f]{16})\.mp3", p)
        if m:
            row = BY_CID.get(m.group(1))
            if row is None:
                self._send(404, b"no"); return
            f = (CLIPS / row["clip"])
            # Belt and braces: the id is opaque and the name came from our own deck,
            # but a containment check costs one line and survives a future edit.
            if not f.is_file() or CLIPS.resolve() not in f.resolve().parents:
                self._send(404, b"no"); return
            self._send(200, f.read_bytes(), "audio/mpeg")
            return
        self._send(404, b"no")

    def do_POST(self):
        if not self._host_ok():
            self._send(403, b"bad host", close=True); return
        if urllib.parse.urlparse(self.path).path != "/save":
            self._send(404, b"no", close=True); return
        origin = self.headers.get("Origin")
        if origin is not None and urllib.parse.urlparse(origin).hostname not in ALLOWED_HOSTS:
            self._send(403, b"bad origin", close=True); return
        if "json" not in (self.headers.get("Content-Type") or ""):
            self._send(415, b"json only", close=True); return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, b"bad length", close=True); return
        if n <= 0 or n > MAX_BODY:
            self._send(413, b"bad length", close=True); return
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            self._send(400, b"bad json"); return
        if not isinstance(body, dict) or not secrets.compare_digest(
                str(body.get("token", "")), TOKEN):
            self._send(403, b"bad token"); return
        row = BY_CID.get(body.get("cid"))
        if row is None:
            self._send(400, b"bad cid"); return

        patch = {}
        if "a" in body:
            a = body["a"]
            if type(a) is not int or not 1 <= a <= 4:
                self._send(400, b"bad answer"); return
            patch["a"] = a
        if "note" in body:
            note = body["note"]
            if not isinstance(note, str) or len(note) > MAX_NOTE:
                self._send(400, b"bad note"); return
            patch["note"] = note
        if not patch:
            self._send(400, b"empty patch"); return
        try:
            answered = _save(row["clip"], patch)
        except OSError:
            self._send(500, b"write failed"); return
        self._send(200, json.dumps({"ok": True, "answered": answered}).encode(),
                   "application/json; charset=utf-8")


class Server(socketserver.ThreadingTCPServer):
    # Threaded on purpose: single-threaded, an 18 s mp3 transfer blocks the save
    # that the judge just made, and a save that looks slow gets pressed twice.
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    if HOST not in ALLOWED_HOSTS:
        raise SystemExit(
            f"άρνηση: HOST={HOST!r} δεν επιτρέπεται. Μόνο {sorted(ALLOWED_HOSTS)}. "
            f"Το 0.0.0.0 βγάζει τον ήχο του δημοτικού συμβουλίου στο LAN.")
    _claim()
    if HOST not in ("127.0.0.1", "::1", "localhost"):
        print(f"ΠΡΟΣΟΧΗ: bind σε {HOST} -- ο ήχος φεύγει από αυτό το μηχάνημα προς "
              f"το tailnet. Loopback είναι το default για λόγο.", flush=True)
    answered = sum(1 for r in STATE["verdicts"].values() if r.get("a"))
    print(f"{len(DECK)} κλιπ · {answered} απαντημένα · seed {SEED} · "
          f"http://{HOST}:{PORT}", flush=True)
    print(f"απαντήσεις -> {VERDICTS}", flush=True)
    with Server((HOST, PORT), H) as s:
        try:
            s.serve_forever()
        except KeyboardInterrupt:
            pass
