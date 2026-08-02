"""The frozen OpenCouncil benchmark, as a substrate for controlled experiments.

`bench.opencouncil.gr` publishes, for every run, a report.json that contains far more
than the leaderboard we normally read off it: for each of the 260 two-minute windows it
carries the **human-corrected reference** and the **verbatim hypothesis text of every
provider**, plus per-item error counts. Every provider transcribed identical audio.

That makes it the one place in this project where several independent ASR systems can be
compared, combined and ablated against a trustworthy reference at zero transcription
cost. `data/asr/export.jsonl` cannot do this: it has one system's output per utterance.

The report is public (no auth). It is cached under $SC rather than in the repo, because
`referenceText` and `hypothesisText` are verbatim council speech, i.e. the same category
of PII the 2026-07-21 purge removed from git history. Do not commit the cache.
"""
from __future__ import annotations

import csv
import json
import os
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

from eval.controlled_eval.scoring import wtoks

ROOT = Path("/home/harold/opencouncil-fine-tuning")
TRAIN_MANIFEST = ROOT / "data/asr/train_manifest.csv"
RUN_ID = "2026-06-10-oc-benchmark"
BASE = "https://bench.opencouncil.gr"

# The metric the benchmark skill calls the fairest single number: hesitation tokens
# ("εεε") are stripped from both sides, so a provider is not penalised for a
# transcription-style choice. Used only for the report's own per-item counts; our arms
# are scored with eval/controlled_eval/scoring.py.
METRIC = "wer-nofillers"


def cache_dir() -> Path:
    return Path(os.environ.get("SC", "/tmp"))


def load_report(run_id: str = RUN_ID) -> dict:
    """Fetch (once) and return the benchmark report."""
    p = cache_dir() / f"bench_{run_id}.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/api/runs/{run_id}/report.json"
        tmp = p.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(p)
    return json.loads(p.read_text())


def provider_ids(report: dict) -> list[str]:
    return [p["instanceId"] for p in report["manifest"]["providers"]]


def training_meetings() -> set[tuple[str, str]]:
    """(city_id, meeting_id) pairs the fine-tune was trained on.

    105 of the benchmark's 203 meetings are in here. Any claim about our own model on
    this benchmark has to be reported split on this, not pooled.
    """
    out = set()
    with open(TRAIN_MANIFEST) as f:
        for row in csv.DictReader(f):
            out.add((row["city_id"], row["meeting_id"]))
    return out


def common_items(report: dict, providers) -> list[dict]:
    """Windows where every requested provider returned a transcript.

    Providers fail independently (our own endpoint timed out on 10 windows), and an arm
    that silently scores on a different subset than the arm it is compared against is
    the exact defect that produced this project's earlier false results.
    """
    providers = list(providers)
    train = training_meetings()
    out = []
    for it in report["items"]:
        pp = it["perProvider"]
        if any(p not in pp or pp[p]["status"] != "ok" for p in providers):
            continue
        out.append({
            "item_id": it["itemId"],
            "city_id": it["cityId"],
            "meeting_id": it["meetingId"],
            "cluster": it["meetingId"],
            "ref": it["referenceText"],
            "hyp": {p: pp[p]["hypothesisText"] for p in providers},
            "app_counts": {p: (pp[p]["perMetric"][METRIC]["numerator"],
                               pp[p]["perMetric"][METRIC]["denominator"])
                           for p in providers},
            "in_training": (it["cityId"], it["meetingId"]) in train,
        })
    return out


# --------------------------------------------------------------- consensus selection
def _codes(texts: dict[str, str]) -> dict[str, str]:
    """Map word tokens to single characters so SequenceMatcher compares words, not
    letters. Distinct providers must share one vocabulary, hence one call per window."""
    enc: dict[str, str] = {}
    out = {}
    for p, t in texts.items():
        s = []
        for w in wtoks(t):
            if w not in enc:
                enc[w] = chr(0x3000 + len(enc))
            s.append(enc[w])
        out[p] = "".join(s)
    return out


def consensus_pick(item: dict, providers) -> str:
    """The reference-free selector: keep whichever hypothesis the others agree with most.

    Classic ROVER voting reduced to whole-window granularity. Needs at least three
    providers — with two there is no majority to be in, and the measured result is that
    a pair does worse than its better half alone.

    Similarity is difflib's ratio rather than a true edit distance. It is a ranking
    signal here, not a reported metric, and it is ~50x faster on 250-word windows.
    """
    providers = list(providers)
    if len(providers) < 3:
        raise ValueError("consensus needs >= 3 providers to have a majority")
    c = _codes({p: item["hyp"][p] for p in providers})
    return max(providers, key=lambda p: sum(
        SequenceMatcher(None, c[p], c[q], autojunk=False).ratio()
        for q in providers if q != p))
