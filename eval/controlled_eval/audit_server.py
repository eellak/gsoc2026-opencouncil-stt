#!/usr/bin/env python3
"""Serve a listening-audit package on the tailnet, with working audio seeking.

Serves this directory only. The answer key lives beside it, outside the served tree, so
nothing here can leak it.

RANGE REQUESTS. `SimpleHTTPRequestHandler` answers every GET with the whole file and never
sends `Accept-Ranges`, so a browser cannot seek: stepping back three seconds makes it
refetch from byte zero and playback restarts from the beginning. That is unusable for
transcription, where you replay the same three seconds twenty times. This handler
implements single-range `bytes=` requests, which is all an audio element needs.

POST /save writes the current localStorage blob to answers.json next to this file, and
refuses to overwrite real work with an empty payload. It also merges rather than replaces:
a debounced save and a `sendBeacon` on unload can arrive out of order, and a plain
last-writer-wins would let the older of the two erase decisions. Per key, the entry with
the newer client timestamp survives. No audit ever deletes a key, so merging cannot
resurrect something the reviewer removed.
"""
import http.server
import json
import math
import os
import pathlib
import re
import socketserver

# AUDIT_DIR lets a package that lives outside the repo be served by this same handler.
# Audit packages hold council audio and transcripts, so they must not sit in git; the
# default stays the script's own directory so the older audits keep working unchanged.
ROOT = pathlib.Path(os.environ.get("AUDIT_DIR") or pathlib.Path(__file__).parent).resolve()
OUT = ROOT / "answers.json"
PORT = int(os.environ.get("PORT", "8780"))
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def merge_answers(prev, new):
    """Union of both, newer client timestamp wins. Older audits store bare strings with
    no timestamp; there the incoming value wins, which is what they did before."""
    out = dict(prev)
    for k, v in new.items():
        old = out.get(k)
        if isinstance(v, dict) and isinstance(old, dict):
            try:
                a, b = float(old.get("t") or 0), float(v.get("t") or 0)
                if math.isfinite(a) and math.isfinite(b) and a > b:
                    continue
            except (TypeError, ValueError, OverflowError):
                pass                                   # no usable timestamp: incoming wins
        out[k] = v
    return out


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        self._range_left = None
        super().__init__(*a, directory=str(ROOT), **kw)

    def send_head(self):
        """The parent's behaviour, plus single-range support so audio can seek."""
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or os.path.isdir(path) or not os.path.isfile(path):
            return super().send_head()
        m = RANGE_RE.match(rng.strip())
        if not m:
            self.send_error(400, "malformed Range")
            return None
        size = os.path.getsize(path)
        start_s, end_s = m.group(1), m.group(2)
        if start_s == "":                          # suffix form, bytes=-500
            if end_s == "":
                self.send_error(400, "malformed Range")
                return None
            start, end = max(0, size - int(end_s)), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self._range_left = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._range_left))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        left = self._range_left
        if left is None:
            return super().copyfile(source, outputfile)
        self._range_left = None
        while left > 0:
            chunk = source.read(min(64 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def send_response(self, code, message=None):
        super().send_response(code, message)
        if code == 200 and self.path.startswith("/clips/"):
            self.send_header("Accept-Ranges", "bytes")

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > 5_000_000:
            self.send_error(413)
            return
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return
        if not isinstance(data, dict):
            self.send_error(400)
            return
        prev = {}
        if OUT.exists():
            try:
                prev = json.loads(OUT.read_text())
            except json.JSONDecodeError:
                prev = {}
        # never let an empty autosave clobber real work
        if not data and prev:
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"skip")
            return
        data = merge_answers(prev, data)
        tmp = OUT.with_suffix(".part")
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(OUT)
        # The rename itself needs the directory synced, or a crash can leave neither name
        # pointing at the new file. An hour of clicking is worth two syscalls.
        d = os.open(str(ROOT), os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")
        print(f"saved {len(data)} entries -> {OUT}", flush=True)

    def log_message(self, fmt, *args):
        if "GET /clips" not in (fmt % args):
            super().log_message(fmt, *args)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        print(f"serving {ROOT} on 0.0.0.0:{PORT}", flush=True)
        httpd.serve_forever()
