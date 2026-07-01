#!/usr/bin/env python3
"""Check internal Markdown links (paths + heading anchors) in committed docs.

Scans tracked-ish Markdown (skips gitignored/local areas: archive/, data/,
docs/issues/, node_modules, .venv, .svelte-kit, .pytest_cache). Reports links whose
target file is missing, or whose #anchor does not match a heading in the target.

Usage: python3 scripts/check-doc-links.py   (exit 1 if any broken links)
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("/.git", "/node_modules", "/.venv", "/archive", "/data/",
        "/.svelte-kit", "/.pytest_cache", "/docs/issues", "/build/")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def slugify(heading: str) -> str:
    """GitHub-style anchor slug."""
    s = heading.strip().lower()
    s = re.sub(r"[`*_~]", "", s)                 # strip inline markdown
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.U)  # drop punctuation
    s = s.replace(" ", "-")
    return s


def anchors_of(path: str) -> set[str]:
    out: set[str] = set()
    try:
        for line in open(path, encoding="utf-8"):
            m = re.match(r"#{1,6}\s+(.*)", line)
            if m:
                out.add(slugify(m.group(1)))
    except OSError:
        pass
    return out


def md_files() -> list[str]:
    found = []
    for base, dirs, files in os.walk(ROOT):
        rel = base[len(ROOT):]
        if any(s in rel + "/" for s in SKIP):
            continue
        for f in files:
            if f.endswith(".md"):
                found.append(os.path.join(base, f))
    return found


def main() -> int:
    broken = []
    for md in md_files():
        d = os.path.dirname(md)
        try:
            txt = open(md, encoding="utf-8").read()
        except OSError:
            continue
        for m in LINK_RE.finditer(txt):
            target = m.group(2).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path, _, anchor = target.partition("#")
            if not path:
                continue
            full = os.path.normpath(os.path.join(d, path))
            if not os.path.exists(full):
                broken.append((md, target, "missing file"))
            elif anchor and full.endswith(".md") and slugify(anchor) not in anchors_of(full):
                broken.append((md, target, "missing anchor"))

    if broken:
        for md, target, why in broken:
            print(f"BROKEN [{why}] {os.path.relpath(md, ROOT)} -> {target}")
        print(f"\n{len(broken)} broken link(s)")
        return 1
    print("OK - no broken internal links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
