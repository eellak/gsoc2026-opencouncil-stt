"""Pool B, stage 1: mine acronym candidates from LEAK-FREE council transcripts.

Source restriction (this is the whole point of the file):
  * the meeting fed the 2026-06-20 glossary, i.e. it is in the eval-harness TRAIN
    fold — see research/glossary/glossary-2026-06-20.manifest.json; and
  * the meeting does not appear **anywhere** in the public benchmark run
    2026-08-10-corrected-adapter-label-prefix-fix-vs-ju.

The second condition is stricter than "train split only". A universal council term
selected on benchmark text would be a selection on the test set and would invalidate
issue #18 before it ran, so no benchmark meeting is read at all.

Output (local, gitignored): data/glossary/poolb_acronyms.raw.json
No thresholds are applied here. Counting is not selecting.

Run:  .venv-eval/bin/python scripts/glossary_poolb_mine.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MEETING_JSON = ROOT / "data/asr/meeting_json"
MANIFEST = ROOT / "research/glossary/glossary-2026-06-20.manifest.json"
BENCH_RUN = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
BENCH_CACHE = Path.home() / ".cache/oc-public" / f"bench_{BENCH_RUN}.json"
OUT = ROOT / "data/glossary/poolb_acronyms.raw.json"

_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΆΈΉΊΌΎΏΪΫ"
# An acronym token: >=3 Greek capitals, optionally with internal dots or a final .
_ACRO_RE = re.compile(rf"^[{_UPPER}](?:\.?[{_UPPER}]){{2,}}\.?$")


def bare(t: str) -> str:
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def source_meetings() -> list[tuple[str, str]]:
    man = json.loads(MANIFEST.read_text())
    train = {(d["city_id"], d["meeting_id"]) for d in man["source_meetings"]["pairs"]}
    rep = json.loads(BENCH_CACHE.read_text())
    bench = {(it["cityId"], it["meetingId"]) for it in rep["items"]}
    have = {}
    for f in MEETING_JSON.iterdir():
        if f.suffix != ".json" or "__" not in f.stem:
            continue
        c, m = f.stem.split("__", 1)
        have[(c, m)] = f
    return sorted((c, m) for (c, m) in have if (c, m) in train and (c, m) not in bench)


def meeting_text(city: str, meeting: str) -> str:
    d = json.loads((MEETING_JSON / f"{city}__{meeting}.json").read_text())
    out = []
    for seg in d.get("transcript") or []:
        for u in seg.get("utterances") or []:
            t = (u.get("text") or "").strip()
            if t:
                out.append(t)
    return "\n".join(out)


def main() -> None:
    pairs = source_meetings()
    acro_meetings: dict[str, set] = defaultdict(set)
    acro_cities: dict[str, set] = defaultdict(set)
    acro_count: dict[str, int] = defaultdict(int)
    variants: dict[str, set] = defaultdict(set)
    total_tokens = 0

    for city, meeting in pairs:
        for tok in meeting_text(city, meeting).split():
            total_tokens += 1
            t = tok.strip("«»\"'()[]{}·,;:!?…—–-")
            if not _ACRO_RE.match(t):
                continue
            key = bare(t.replace(".", "")).upper()
            acro_meetings[key].add(f"{city}/{meeting}")
            acro_cities[key].add(city)
            acro_count[key] += 1
            variants[key].add(t)

    rows = sorted(
        ({"acronym": k,
          "n_meetings": len(acro_meetings[k]),
          "n_cities": len(acro_cities[k]),
          "n_occurrences": acro_count[k],
          "cities": sorted(acro_cities[k]),
          "surface_variants": sorted(variants[k])}
         for k in acro_meetings),
        key=lambda r: (-r["n_meetings"], -r["n_occurrences"], r["acronym"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": {
            "rule": "eval-harness TRAIN fold AND absent from benchmark run "
                    + BENCH_RUN,
            "n_meetings": len(pairs),
            "n_cities": len({c for c, _ in pairs}),
            "n_tokens_read": total_tokens,
            "meetings": [{"city_id": c, "meeting_id": m} for c, m in pairs],
        },
        "n_distinct_acronyms": len(rows),
        "acronyms": rows,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(pairs)} meetings, {total_tokens:,} tokens, "
          f"{len(rows)} distinct acronym forms -> {OUT}")


if __name__ == "__main__":
    main()
