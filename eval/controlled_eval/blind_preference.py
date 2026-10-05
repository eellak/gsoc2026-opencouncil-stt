"""The blind preference page, and the frozen rule that reads its answers.

Record: exp-2026-08-26-training-round-2. Design frozen 2026-09-06 in wayfinder #59,
built under #64. Terms: GLOSSARY.md ("Blind preference judgement", "Ordinary sample",
"Challenge sample").

ONE QUESTION PER CLIP

The listener hears a clip, sees two transcripts whose identity is hidden and whose order
is randomised, and answers **Α / Β / ισοπαλία**: which is closer to the audio. That is
the whole question. The 203-clip predecessor was abandoned after three judgements
because its question -- "was this said?" -- conflated fidelity-to-audio with whether
something belongs in the minutes, and those are different measurements that this project
has already paid for merging once. Nothing else is asked.

The tie is a first-class answer, not a fallback. Without it the listener is forced to
choose on clips where there is no difference, and the noise buries the signal.

THE ONE ANNOTATION, AND WHERE IT DEPARTS FROM THE TICKET

Beside the choice there is an optional "έχασε λόγια" (lost speech) flag -- **one per
candidate**, so two checkboxes on the page. The frozen guard is a comparison BETWEEN
arms ("the lost-speech count must not rise"), and a single unattributed flag cannot
produce one: it would only say that somebody dropped words, which is the same number for
both arms by construction. Attributing it is the minimum that makes the frozen guard
computable. It is not a second question -- it never asks whether something belongs in
the transcript, only whether that candidate dropped speech that is audible.

THE DECISION RULE IS IN THE CODE AND REFUSES TO BE COMPUTED ANY OTHER WAY

    the new arm wins  <=>  the LOWER bound of the win-rate CI is strictly above 50%
                           under a CITY-STRATIFIED, MEETING-CLUSTERED bootstrap,
                           on the ORDINARY SAMPLE,
                           over the COMPLETE set of that sample's clips,
                           and the lost-speech count does not rise.

`decide()` raises `SampleKindError` when handed anything but the ordinary sample,
`MixedSampleError` when the judgement set is not exactly that sample's clips, and
`JudgementError` when it is partial or duplicated. Requiring completeness is what stops
optional stopping: without it, "the CI excluded 50% at clip 54" is a decision made by
watching. The challenge sample goes through `challenge_table()`, which returns the same
descriptive numbers with `decides: False` and no verdict field at all. There is no flag
that turns one into the other.

THE TWO SAMPLES NOW SIT ON DIFFERENT FRAMES, AND THAT IS THE POINT

The ordinary sample was re-cut on 2026-09-06 (v2): 45 clips, 3 in each of 15 released
meetings that no training source touched and the exclusion manifest permits at both
levels. The challenge sample stayed on the listening queue, because `deletion_hard` spans
exist nowhere else -- which makes it a TRAINING-SIDE DIAGNOSTIC rather than an
evaluation, since its meetings are the ones the assisted clips train on. Every verdict
carries `ORD.PRECISION_NOTE`: **15 meeting clusters against v1's 67**, so only a large
effect is visible at all. That sentence rides on the number rather than sitting in a
report nobody re-reads.

Clustering is by `(city, meeting_id)` -- never the meeting slug, which repeats across
cities -- because two clips of one meeting are the same room, the same microphones and
often the same speaker; treating 80 clips as 80 independent trials would narrow the
interval by pretending to information the sample does not have. Resampling is STRATIFIED
BY CITY, drawing back each city's own number of meetings, because a pooled resample would
randomly disturb the 40/18/15/7 city mixture the design deliberately fixed, and the city
estimates are recombined with those same frozen clip weights (Codex review 0201f130).

TIES COUNT HALF. A clip on which neither transcript is closer to the audio is neutral
evidence about "which is better overall", so it enters the score as 0.5 rather than being
discarded. Discarding ties makes a handful of decisive judgements among mostly ties look
like a strong result, and it manufactures bootstrap replicates with no denominator at
all. The decisive-only rate is reported beside it as `decisive_win_rate`, descriptive and
unreachable from `decide()`.

BLINDNESS IS A PROPERTY OF THE PIPELINE

  * the browser receives a token, a duration and two texts -- never the clip id (which
    names its meeting), never the arm names, never which side is which;
  * the side assignment is derived server-side from a frozen seed and is not sent;
  * a judgement reaches server state only after the write to disk succeeded;
  * `/reveal` refuses until the WHOLE session is committed, not merely that clip.
    Revealing per clip would teach the listener each arm's textual style and unblind
    every clip after it (Codex review 0201f130).

SERVING. Bound to loopback or an address `tailscale ip` reports for this machine, and
nothing else -- there is no authentication and the audio is council speech, the same PII
category as the 2026-07-21 history purge.

    .venv-eval/bin/python -m eval.controlled_eval.blind_preference serve \\
        --sample ordinary --host 100.x.y.z
    .venv-eval/bin/python -m eval.controlled_eval.blind_preference report \\
        --sample ordinary
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import http.server
import json
import os
import random
import re
import socketserver
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.controlled_eval import cut_challenge_sample as CHL  # noqa: E402
from eval.controlled_eval import cut_ordinary_sample as ORD  # noqa: E402

SC = Path.home() / ".cache" / "oc-public"
PREF_DIR = SC / "pref-2026-09"

# --------------------------------------------------------------------- frozen
ORDINARY = ORD.SAMPLE_KIND          # "ordinary"
CHALLENGE = CHL.SAMPLE_KIND         # "challenge"
SAMPLES = (ORDINARY, CHALLENGE)

# The sample the decision rule may read. Not a default, not a parameter with a default:
# `decide` compares against this and raises otherwise.
DECIDING_SAMPLE = ORDINARY

SAMPLE_DIRS = {ORDINARY: ORD.CURRENT, CHALLENGE: CHL.CURRENT}

CHOICES = ("A", "B", "tie")
SIDES = ("A", "B")

CI_LEVEL = 0.95
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = "blind-preference-2026-09/bootstrap/v1"
SIDE_SEED = "blind-preference-2026-09/side/v1"
ORDER_SEED = "blind-preference-2026-09/order/v1"

# A tie scores half a win. The alternative -- dropping ties and scoring only the
# decisive judgements -- lives in `decisive_win_rate` as a named descriptive statistic
# that `decide` cannot reach: swapping between the two after seeing a number is how a
# frozen rule stops being frozen.
TIE_RULE = "half"
TIE_SCORE = 0.5

MAX_BODY_BYTES = 64 * 1024
SOCKET_TIMEOUT_SEC = 30

# The city segment allows a hyphen: `papagos-cholargos` is a real city id and entered
# the frame with the v2 sample. A regex that rejected it would have failed closed, but
# only at load time, after the clips were cut.
_CLIP_ID_RE = {
    ORDINARY: re.compile(r"^ord__[a-z0-9-]+__[a-z0-9_]+__\d+$"),
    CHALLENGE: re.compile(r"^chl__[a-z0-9-]+__[a-z0-9_]+__\d+$"),
}


class SampleKindError(RuntimeError):
    """The decision rule was asked for a sample that may not decide anything."""


class MixedSampleError(RuntimeError):
    """Judgements from two samples reached one score. They are never summed."""


class CandidateError(ValueError):
    """The two candidate transcripts are missing, malformed, or not two."""


class JudgementError(ValueError):
    """The answer cannot be stored: unknown choice, or outside the session."""


class RevealDenied(PermissionError):
    """Arm identity was asked for before the session was finished. That is the point."""


# ------------------------------------------------------------------- loading

def sample_dir(sample: str, root: Path | None = None) -> Path:
    if sample not in SAMPLES:
        raise SampleKindError(f"unknown sample {sample!r}; the two samples are {SAMPLES}")
    return Path(root) if root is not None else SAMPLE_DIRS[sample]


def load_sample(sample: str, root: Path | None = None) -> list[dict]:
    """The clips of one sample, checked to be that sample and only that sample."""
    d = json.loads((sample_dir(sample, root) / "sample.json").read_text(encoding="utf-8"))
    clips = d["clips"]
    rx = _CLIP_ID_RE[sample]
    for c in clips:
        if c.get("sample") != sample:
            raise MixedSampleError(
                f"{c.get('clip_id')} is labelled {c.get('sample')!r} inside the "
                f"{sample!r} manifest")
        if not rx.match(c.get("clip_id") or ""):
            raise MixedSampleError(
                f"{c.get('clip_id')!r} is not a {sample!r} clip id")
    return clips


def pref_dir(sample: str, root: Path | None = None) -> Path:
    return (Path(root) if root is not None else PREF_DIR) / sample


def load_candidates(sample: str, root: Path | None = None) -> dict:
    """The two candidate transcripts per clip. Never in git; never served with names.

    Shape:
        {"incumbent_arm": "artifact-...", "new_arm": "artifact-...",
         "clips": {"<clip_id>": {"<arm>": "<text>", "<arm>": "<text>"}}}
    """
    p = pref_dir(sample, root) / "candidates.json"
    if not p.exists():
        raise CandidateError(
            f"{p} does not exist. Write the two arms' transcripts there before serving; "
            "this module never produces them, because a page that could decode the "
            "audio itself would know which arm it was showing.")
    d = json.loads(p.read_text(encoding="utf-8"))
    inc, new = d.get("incumbent_arm"), d.get("new_arm")
    if not inc or not new or inc == new:
        raise CandidateError(
            f"incumbent_arm={inc!r} and new_arm={new!r} must both be set and differ; "
            "name the ledger artifact ids, not 'old' and 'new'")
    clips = d.get("clips")
    if not isinstance(clips, dict) or not clips:
        raise CandidateError(f"{p} carries no clips")
    for cid, texts in clips.items():
        if set(texts) != {inc, new}:
            raise CandidateError(
                f"{cid} carries arms {sorted(texts)}, expected exactly {sorted([inc, new])}")
        for arm, t in texts.items():
            if not isinstance(t, str):
                raise CandidateError(f"{cid}/{arm} is {type(t).__name__}, not a string")
    return {"incumbent_arm": inc, "new_arm": new, "clips": clips}


def check_coverage(clips, candidates: dict) -> None:
    missing = [c["clip_id"] for c in clips if c["clip_id"] not in candidates["clips"]]
    if missing:
        raise CandidateError(
            f"{len(missing)} of {len(clips)} clips have no candidate pair, e.g. "
            f"{missing[:3]}. A partly covered sample is a different sample.")


# ------------------------------------------------------------------- session

def _unit(*parts) -> float:
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") / 2 ** 64


def present_token(sample: str, clip_id: str) -> str:
    """The opaque handle for one presentation. Opacity, not secrecy.

    What it stops is the meeting -- which has a published transcript -- arriving unasked
    in front of the listener in a URL.
    """
    return hashlib.sha256(f"{sample}#{clip_id}".encode("utf-8")).hexdigest()[:16]


def side_assignment(sample: str, clip_id: str, incumbent_arm: str,
                    new_arm: str) -> dict[str, str]:
    """Which arm is shown as Α and which as Β, per clip, from a frozen seed.

    Per clip rather than per session, so the listener cannot learn "the new one is
    always on the left" halfway through; deterministic, so a resumed session shows the
    same sides and a stored judgement never changes meaning.
    """
    first = _unit(SIDE_SEED, sample, clip_id) < 0.5
    return {"A": incumbent_arm if first else new_arm,
            "B": new_arm if first else incumbent_arm}


def session_order(sample: str, clips, candidates: dict) -> list[dict]:
    """One presentation per clip, in an order that is not the meeting order.

    Meeting order would put a meeting's two clips back to back, and the second judgement
    would then be made in the shadow of the first.
    """
    check_coverage(clips, candidates)
    inc, new = candidates["incumbent_arm"], candidates["new_arm"]
    out = []
    for c in sorted(clips, key=lambda c: (_unit(ORDER_SEED, sample, c["clip_id"]),
                                          c["clip_id"])):
        sides = side_assignment(sample, c["clip_id"], inc, new)
        out.append({
            "token": present_token(sample, c["clip_id"]),
            "clip_id": c["clip_id"],
            "sample": sample,
            "meeting": f"{c['city']}/{c['meeting_id']}",
            "clip_dur": c["clip_dur"],
            "sides": sides,
            "texts": {s: candidates["clips"][c["clip_id"]][sides[s]] for s in SIDES},
        })
    toks = [p["token"] for p in out]
    if len(set(toks)) != len(toks):
        raise JudgementError("presentation token collision")
    return out


def page_payload(presentations) -> list[dict]:
    """What the browser is allowed to know: a duration and two anonymous texts."""
    return [{"token": p["token"], "clip_dur": p["clip_dur"],
             "a": p["texts"]["A"], "b": p["texts"]["B"]} for p in presentations]


def _committed(row) -> bool:
    return bool(row) and row.get("choice") in CHOICES


def remaining(presentations, judgements: dict) -> list[dict]:
    return [p for p in presentations if not _committed(judgements.get(p["token"]))]


def save_judgement(judgements: dict, presentations, token: str, choice: str,
                   lost_speech: dict | None, now: str) -> dict:
    """Store one judgement, resolved to arm identities, keeping any previous answer.

    The resolution happens here rather than at scoring time so the record on disk is
    self-contained: a later change to the side seed cannot retroactively reinterpret an
    answer that was already given.
    """
    if choice not in CHOICES:
        raise JudgementError(f"unknown choice {choice!r}; the frozen choices are {CHOICES}")
    by_token = {p["token"]: p for p in presentations}
    p = by_token.get(token)
    if p is None:
        raise JudgementError(f"{token} is not a presentation in this session")
    if not remaining(presentations, judgements):
        # Once the session is complete the identities are revealable, so any further
        # answer is an INFORMED one. Refusing is what keeps "the rule is applied once,
        # to the whole sample" true after the fact as well as before it.
        raise JudgementError(
            "this session is already complete, so the arm identities are revealable and "
            "a new answer would be an informed one. Move judgements.json aside and "
            "re-judge the whole sample if it has to be redone.")
    flags = dict(lost_speech or {})
    unknown = set(flags) - set(SIDES)
    if unknown:
        raise JudgementError(f"lost_speech names {sorted(unknown)}, not {list(SIDES)}")
    per_arm = {p["sides"][s]: bool(flags.get(s)) for s in SIDES}
    row = {
        "clip_id": p["clip_id"],
        "sample": p["sample"],
        "meeting": p["meeting"],
        "choice": choice,
        "winner_arm": None if choice == "tie" else p["sides"][choice],
        "loser_arm": None if choice == "tie" else p["sides"]["B" if choice == "A" else "A"],
        "lost_speech": per_arm,
        "saved_at": now,
    }
    out = dict(judgements)
    prev = out.get(token)
    if prev:
        history = list(prev.get("previous") or [])
        history.append({k: prev.get(k) for k in
                        ("choice", "winner_arm", "lost_speech", "saved_at")})
        row["previous"] = history
    out[token] = row
    return out


def reveal(token: str, presentations, judgements: dict) -> dict:
    """Which side was which arm -- only after the WHOLE session is committed.

    Not after that clip's judgement. Revealing per clip would let the listener learn each
    arm's textual style from the first few answers and carry it into every clip after,
    which unblinds the rest of the session without anybody deciding to (Codex review
    0201f130). The write-before-reveal ordering still holds: the judgement is on disk
    before it counts towards completion.
    """
    p = {x["token"]: x for x in presentations}.get(token)
    if p is None:
        raise RevealDenied(f"{token} is not a presentation in this session")
    todo = remaining(presentations, judgements)
    if todo:
        raise RevealDenied(
            f"{len(todo)} of {len(presentations)} clips are still unjudged. Identity is "
            "released when the session is finished, not clip by clip: knowing which arm "
            "won clip 3 is a hint about every clip after it.")
    row = judgements.get(token)
    if not _committed(row):
        raise RevealDenied(f"no committed judgement for {token} yet")
    return {"token": token, "sides": p["sides"], "choice": row["choice"],
            "winner_arm": row["winner_arm"], "lost_speech": row["lost_speech"]}


# ------------------------------------------------------------------- scoring

def records(judgements: dict, presentations=None) -> list[dict]:
    """The committed judgements as flat rows, in a stable order."""
    known = {p["token"] for p in presentations} if presentations is not None else None
    out = []
    for token, row in sorted(judgements.items()):
        if known is not None and token not in known:
            continue
        if not _committed(row):
            continue
        out.append({k: row[k] for k in
                    ("clip_id", "sample", "meeting", "choice", "winner_arm",
                     "lost_speech")})
    return out

def _one_sample(rows) -> str:
    """The sample every row belongs to, refusing a set that is not exactly one."""
    if not rows:
        raise JudgementError("no committed judgements")
    kinds = {r["sample"] for r in rows}
    if len(kinds) != 1:
        raise MixedSampleError(
            f"judgements from {sorted(kinds)} reached one score. The ordinary and the "
            "challenge sample answer different questions and are never summed.")
    sample = kinds.pop()
    rx = _CLIP_ID_RE.get(sample)
    if rx is None:
        raise SampleKindError(f"unknown sample {sample!r}")
    stray = [r["clip_id"] for r in rows if not rx.match(r["clip_id"] or "")]
    if stray:
        raise MixedSampleError(
            f"{len(stray)} clip ids do not belong to the {sample!r} sample, e.g. "
            f"{stray[:3]}")
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise JudgementError(f"{len(dup)} clips are judged twice in one score: {dup[:3]}")
    return sample


def check_complete(rows, clips) -> None:
    """Exactly the sample's clips, no more and no fewer.

    Completeness is not tidiness: a rule applied to whatever has been judged so far is a
    rule applied at a moment the listener chose, and "the interval excluded 50% at clip
    54" is optional stopping wearing a confidence interval (Codex review 0201f130).
    """
    want = {c["clip_id"] for c in clips}
    got = {r["clip_id"] for r in rows}
    if got != want:
        missing, extra = sorted(want - got), sorted(got - want)
        raise JudgementError(
            f"the judgement set is not the frozen sample: {len(missing)} clips are "
            f"unjudged (e.g. {missing[:3]}) and {len(extra)} are not in the manifest "
            f"(e.g. {extra[:3]}). The rule is applied once, to the whole sample.")


def _score(row, new_arm: str) -> float:
    """One judgement as a number for the new arm. Ties are half, by the frozen rule."""
    if row["choice"] == "tie":
        return TIE_SCORE
    return 1.0 if row["winner_arm"] == new_arm else 0.0


def _city(row) -> str:
    return row["meeting"].split("/", 1)[0]


def _weighted_point(by_city_rows: dict, weights: dict, new_arm: str) -> float | None:
    """Mean score per city, recombined with the frozen city clip weights."""
    num = 0.0
    den = 0.0
    for city, rows in by_city_rows.items():
        if not rows:
            continue
        w = weights.get(city)
        if not w:
            continue
        num += w * (sum(_score(r, new_arm) for r in rows) / len(rows))
        den += w
    return (num / den) if den else None


def city_weights(rows) -> dict[str, int]:
    """The frozen city mixture, taken from the judged sample's own clip counts.

    The ordinary sample is apportioned 40/18/15/7 and `check_complete` has already
    established that every one of those clips is present, so counting them here is
    reading the frozen quota rather than reacting to the data.
    """
    out: dict[str, int] = {}
    for r in rows:
        out[_city(r)] = out.get(_city(r), 0) + 1
    return out


def win_rate_ci(rows, new_arm: str, n_bootstrap: int = N_BOOTSTRAP,
                ci_level: float = CI_LEVEL, seed: str = BOOTSTRAP_SEED) -> dict:
    """The new arm's win rate, with a city-stratified, meeting-clustered percentile CI.

    Within each city, that city's meetings are resampled WITH REPLACEMENT to the same
    number of meetings it actually has, and every judgement of a drawn meeting comes with
    it. The city means are then recombined with the frozen city clip weights. Two things
    this buys: a replicate cannot silently change the city mixture the design fixed, and
    -- because ties score half rather than being dropped -- no replicate can have an
    empty denominator, so the interval is never conditioned on which replicates happened
    to be computable.
    """
    _one_sample(rows)
    weights = city_weights(rows)
    by_city_meetings: dict[str, dict[str, list]] = {}
    for r in rows:
        by_city_meetings.setdefault(_city(r), {}).setdefault(r["meeting"], []).append(r)

    point = _weighted_point(
        {c: [r for m in ms.values() for r in m] for c, ms in by_city_meetings.items()},
        weights, new_arm)
    ties = sum(1 for r in rows if r["choice"] == "tie")
    wins = sum(1 for r in rows if r["winner_arm"] == new_arm)
    losses = len(rows) - ties - wins

    out = {
        "n_judgements": len(rows),
        "n_meetings": sum(len(ms) for ms in by_city_meetings.values()),
        "n_cities": len(by_city_meetings),
        "cluster_key": "(city, meeting_id)",
        "city_weights": dict(sorted(weights.items())),
        "meetings_by_city": {c: len(ms) for c, ms in sorted(by_city_meetings.items())},
        "n_ties": ties,
        "tie_share": round(ties / len(rows), 4),
        "wins_new_arm": wins,
        "wins_incumbent": losses,
        "win_rate": None if point is None else round(point, 4),
        "decisive_win_rate": (round(wins / (wins + losses), 4)
                              if (wins + losses) else None),
        "n_decisive": wins + losses,
        "tie_rule": TIE_RULE,
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "bootstrap": "city-stratified, meeting-clustered percentile",
        "quantile_method": "nearest-rank on the sorted replicates",
        "bootstrap_seed": seed,
        "ci_lo": None,
        "ci_hi": None,
    }
    if point is None:
        return out

    rng = random.Random(f"{seed}|{new_arm}|{len(rows)}|{sorted(weights.items())}")
    # Sorted by city, not in row order: `rng.choice` is consumed city by city, so an
    # unsorted iteration would make the interval depend on the order the judgements
    # happened to arrive in.
    meetings_by_city = {c: sorted(ms) for c, ms in sorted(by_city_meetings.items())}
    reps = []
    for _ in range(n_bootstrap):
        drawn = {}
        for city, names in meetings_by_city.items():
            picked = [rng.choice(names) for _ in names]
            drawn[city] = [r for name in picked
                           for r in by_city_meetings[city][name]]
        rep = _weighted_point(drawn, weights, new_arm)
        if rep is None:                       # unreachable: ties score, so every city has a mean
            raise JudgementError("a bootstrap replicate had no score; refusing to drop it")
        reps.append(rep)
    reps.sort()
    alpha = (1 - ci_level) / 2

    def q(prob: float) -> float:
        idx = min(len(reps) - 1, max(0, int(round(prob * (len(reps) - 1)))))
        return reps[idx]

    out["ci_lo"] = round(q(alpha), 4)
    out["ci_hi"] = round(q(1 - alpha), 4)
    return out


def lost_speech_counts(rows, incumbent_arm: str, new_arm: str) -> dict:
    """How many clips each arm was marked as having dropped audible speech on.

    Both arms are counted from the SAME judgements, so the comparison is paired. The flag
    is per candidate; a single unattributed flag would give the same number twice and
    could not implement a guard that is a comparison between arms.
    """
    def flag(r, arm):
        flags = r.get("lost_speech") or {}
        if arm not in flags:
            raise JudgementError(
                f"{r['clip_id']} has no lost-speech flag for {arm!r}. The guard is a "
                "comparison between arms and cannot be computed from an unattributed "
                "flag; refusing to guess.")
        return bool(flags[arm])

    inc = sum(1 for r in rows if flag(r, incumbent_arm))
    new = sum(1 for r in rows if flag(r, new_arm))
    both = sum(1 for r in rows if flag(r, incumbent_arm) and flag(r, new_arm))
    return {"incumbent": inc, "new": new, "both": both, "n": len(rows),
            "new_only": new - both, "incumbent_only": inc - both,
            "rose": new > inc}


def decide(rows, incumbent_arm: str, new_arm: str, clips, **kw) -> dict:
    """THE frozen rule. Ordinary sample only, complete, clustered, deletion-guarded.

    Every clause is a conjunction and none of them is optional:
      * the sample must be the ordinary one;
      * the judgement set must be exactly that sample's clips;
      * the LOWER bound of the CI must be strictly above 50% -- an interval entirely
        below 50% also "excludes 50%", and that is a loss, not a win;
      * the lost-speech count must not rise.

    `clips` is the frozen sample manifest and is required, not optional: membership is
    checked against it rather than against a clip-id regex, so a judgement for a clip
    that is not in this sample cannot pass by looking well formed.
    """
    sample = _one_sample(rows)
    if sample != DECIDING_SAMPLE:
        raise SampleKindError(
            f"the {sample!r} sample may not decide anything. It is a training-side "
            "diagnostic: its positions were chosen by the incumbent's failures and its "
            "meetings are the ones the assisted clips train on, so both defects point "
            "towards flattering the arm being judged. It answers 'did the recipe fix the "
            "failures in meetings it trained on' and nothing else. Use challenge_table().")
    for c in clips:
        if c.get("sample") != DECIDING_SAMPLE:
            raise MixedSampleError(
                f"{c.get('clip_id')} in the manifest is labelled {c.get('sample')!r}")
    check_complete(rows, clips)
    ci = win_rate_ci(rows, new_arm, **kw)
    guard = lost_speech_counts(rows, incumbent_arm, new_arm)
    lo, hi = ci["ci_lo"], ci["ci_hi"]
    above = lo is not None and lo > 0.5
    excludes_50 = lo is not None and hi is not None and (lo > 0.5 or hi < 0.5)
    wins = bool(above and not guard["rose"])
    return {
        "sample": sample,
        "rule": ("lower bound of the city-stratified, meeting-clustered win-rate CI "
                 "strictly above 0.5 on the complete ordinary sample, and the "
                 "lost-speech count not rising"),
        "estimand": ORD.ESTIMAND,
        "incumbent_arm": incumbent_arm,
        "new_arm": new_arm,
        "verdict": "NEW ARM WINS" if wins else "NO WIN",
        "new_arm_wins": wins,
        "precision_note": ORD.PRECISION_NOTE,
        "ci_lower_above_50": above,
        "ci_excludes_50": excludes_50,
        "lost_speech_guard_passed": not guard["rose"],
        "lost_speech": guard,
        **ci,
    }


def challenge_table(rows, incumbent_arm: str, new_arm: str, clips=None, **kw) -> dict:
    """The challenge sample's own table. Descriptive, and it decides nothing."""
    sample = _one_sample(rows)
    if sample == DECIDING_SAMPLE:
        raise SampleKindError(
            "the ordinary sample goes through decide(), which applies the frozen rule")
    if clips is not None:
        check_complete(rows, clips)
    ci = win_rate_ci(rows, new_arm, **kw)
    return {
        "sample": sample,
        "incumbent_arm": incumbent_arm,
        "new_arm": new_arm,
        "decides": False,
        "role": "training-side diagnostic",
        "estimand": CHL.ESTIMAND,
        "may_support": ("did the recipe fix the failures in meetings it trained on; a "
                        "negative kills the recipe, a positive means little. Never a "
                        "claim about overall quality, and never summed with the "
                        "ordinary sample."),
        "complete": clips is not None,
        "lost_speech": lost_speech_counts(rows, incumbent_arm, new_arm),
        **ci,
    }


# ------------------------------------------------------------------ the server

PAGE = """<!doctype html><html lang="el"><meta charset="utf-8">
<title>__TITLE__</title>
<style>
 body{font:16px/1.5 system-ui;margin:0;background:#111;color:#eee}
 main{max-width:60rem;margin:0 auto;padding:1.2rem}
 h1{font-size:1rem;font-weight:600;color:#9ab;margin:0 0 .3rem}
 audio{width:100%;margin:.4rem 0}
 .bar{display:flex;gap:.6rem;align-items:center;margin:.7rem 0;flex-wrap:wrap}
 .n{color:#789;font-variant-numeric:tabular-nums}
 .hint{color:#667;font-size:.85rem}
 .pair{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin:.6rem 0}
 .cand{background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:.7rem .8rem}
 .cand h2{font-size:.8rem;color:#9ab;margin:0 0 .4rem;letter-spacing:.08em}
 .cand p{margin:0;white-space:pre-wrap;font-size:17px}
 .cand.empty p{color:#755;font-style:italic}
 .cand label{display:block;margin-top:.6rem;color:#c96;font-size:.85rem;
   user-select:none;cursor:pointer}
 button{font:15px system-ui;padding:.55rem 1.1rem;border-radius:6px;border:0;
   background:#2d6;color:#052;font-weight:600;cursor:pointer}
 button.ghost{background:#333;color:#ccc;font-weight:400}
 button kbd{opacity:.6;font-size:.75rem;margin-left:.35rem}
 .rev{margin-top:1rem;padding:.7rem;background:#181818;border-left:3px solid #46a;
   border-radius:0 6px 6px 0;display:none;font-size:.9rem;color:#9ab}
 details{background:#161616;border-radius:6px;padding:.5rem .7rem;margin:.8rem 0 0}
 summary{cursor:pointer;color:#9ab;font-size:.9rem}
 @media(max-width:44rem){.pair{grid-template-columns:1fr}}
</style>
<main>
 <h1>Ποιο κείμενο είναι πιο κοντά στον ήχο;</h1>
 <div class="bar"><span class="n" id="pos"></span>
   <span class="hint" id="dur"></span><span class="hint" id="eta"></span></div>
 <audio id="a" controls preload="none"></audio>
 <div class="pair">
   <div class="cand" id="ca"><h2>Α</h2><p id="ta"></p>
     <label><input type="checkbox" id="la"> έχασε λόγια</label></div>
   <div class="cand" id="cb"><h2>Β</h2><p id="tb"></p>
     <label><input type="checkbox" id="lb"> έχασε λόγια</label></div>
 </div>
 <div class="bar">
   <button id="pa">Α<kbd>1</kbd></button>
   <button id="pt" class="ghost">Ισοπαλία<kbd>2</kbd></button>
   <button id="pb">Β<kbd>3</kbd></button>
   <span class="hint" id="msg"></span>
   <span class="hint">Alt+R = ξανάκουσέ το</span>
 </div>
 <details><summary>Τι ρωτάει ακριβώς (20 δευτ.)</summary>
  <p class="hint">Μία ερώτηση: <b>ποιο από τα δύο είναι πιο κοντά σε ό,τι ακούς</b>.
  Όχι ποιο διαβάζεται καλύτερα, όχι ποιο ταιριάζει στα πρακτικά. Αν δεν ξεχωρίζουν,
  <b>ισοπαλία</b> — είναι κανονική απάντηση, όχι αποτυχία.</p>
  <p class="hint">Το «έχασε λόγια» μπαίνει σε όποιο κείμενο <b>παρέλειψε</b> λόγια που
  ακούγονται. Μπορεί να μπει και στα δύο, ή σε κανένα. Χαμηλότερο λάθος πληρωμένο με
  σιωπή δεν είναι βελτίωση.</p>
  <p class="hint">Ποιο κείμενο είναι ποιανού δεν φαίνεται πουθενά, και η σειρά Α/Β
  αλλάζει σε κάθε κλιπ.</p>
 </details>
 <div class="rev" id="rev"></div>
</main>
<script>
const Q = __DATA__; let i = 0, busy = false;
const $ = s => document.querySelector(s);
const BTNS = ['#pa','#pt','#pb'];
function lock(on){ busy = on; BTNS.forEach(b => $(b).disabled = on); }
function eta(){
  const left = Q.length - i;
  return left ? '≈' + Math.max(1, Math.round(left*25/60)) + ' λ. ακόμα' : '';
}
function show(){
  const c = Q[i];
  $('#pos').textContent = c ? (i+1) + ' / ' + Q.length : 'τέλος — τίποτα άλλο';
  $('#dur').textContent = c ? c.clip_dur.toFixed(1) + ' δευτ.' : '';
  $('#eta').textContent = eta();
  $('#rev').style.display='none'; $('#rev').textContent='';
  $('#msg').textContent='';
  $('#la').checked = false; $('#lb').checked = false;
  $('#ta').textContent = c ? (c.a || '— κενό —') : '';
  $('#tb').textContent = c ? (c.b || '— κενό —') : '';
  $('#ca').className = 'cand' + (c && !c.a ? ' empty' : '');
  $('#cb').className = 'cand' + (c && !c.b ? ' empty' : '');
  $('#a').src = c ? '/clip/' + c.token + '.wav' : '';
  lock(!c);
  if(c) $('#a').play().catch(() => {});
}
async function commit(choice){
  if(busy) return;
  const c = Q[i]; if(!c) return;
  lock(true);
  let r;
  try{
    r = await fetch('/judge', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({token: c.token, choice: choice,
        lost_speech: {A: $('#la').checked, B: $('#lb').checked}})});
  }catch(e){
    $('#msg').textContent = 'δεν αποθηκεύτηκε (δίκτυο) — δοκίμασε ξανά';
    lock(false); return;
  }
  if(!r.ok){
    $('#msg').textContent = 'δεν αποθηκεύτηκε: ' + await r.text();
    lock(false); return;
  }
  // The answer is on disk before the next clip appears. `show()` re-locks when the
  // queue runs out, so the buttons do not stay live on the end screen.
  lock(false);
  next();
}
function next(){ i++; show(); }
$('#pa').onclick = () => commit('A');
$('#pb').onclick = () => commit('B');
$('#pt').onclick = () => commit('tie');
document.addEventListener('keydown', e => {
  if(e.altKey && (e.key === 'r' || e.key === 'ρ')){
    $('#a').currentTime = 0; $('#a').play(); e.preventDefault(); return; }
  if(e.target && e.target.tagName === 'INPUT') return;
  const map = {'1':'A', '2':'tie', '3':'B'};
  if(map[e.key]){ commit(map[e.key]); e.preventDefault(); }
});
show();
</script></html>"""


def _read(path: Path, default):
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def script_json(obj) -> str:
    """JSON safe to substitute into an inline `<script>` block.

    `json.dumps` leaves `<` and `>` as themselves, so a candidate transcript containing
    `</script>` would close the block and turn the rest of the transcript into markup.
    The rendering side already uses `textContent`; this closes the injection on the way
    IN as well (CodeRabbit, 2026-09-06). `\u003c` and friends are ordinary JSON escapes,
    so the strings the page ends up with are byte-identical.
    """
    return (json.dumps(obj, ensure_ascii=False)
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))


def make_handler(presentations, clips_dir: Path, judgements_path: Path, sample: str):
    state = {"judgements": _read(judgements_path, {})}
    by_token = {p["token"]: p["clip_id"] for p in presentations}
    rx = _CLIP_ID_RE[sample]

    def page() -> str:
        # The payload is substituted INTO an inline <script>, so `json.dumps` alone is
        # not enough: it leaves `<` and `>` as themselves, and a candidate transcript
        # containing `</script>` would end the block and turn the rest of the transcript
        # into markup. The rendering side already uses `textContent`; this closes the
        # injection on the way in as well (CodeRabbit, 2026-09-06). Escaping to \u003c
        # is transparent to JSON, so the strings the page sees are unchanged.
        todo = remaining(presentations, state["judgements"])
        return (PAGE
                .replace("__DATA__", script_json(page_payload(todo)))
                .replace("__TITLE__", f"pref-2026-09 · {sample}"))

    class H(http.server.BaseHTTPRequestHandler):
        timeout = SOCKET_TIMEOUT_SEC

        def log_message(self, *a):
            pass                      # never log: the paths carry meeting ids

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
                if not cid or not rx.match(cid):
                    return self.send_error(404)
                f = Path(clips_dir) / f"{cid}.wav"
                if not f.is_file():
                    return self.send_error(404)
                return self._send(200, f.read_bytes(), "audio/wav")
            m = re.fullmatch(r"/reveal/([0-9a-f]{16})", p)
            if m:
                try:
                    return self._json(200, reveal(m.group(1), presentations,
                                                  state["judgements"]))
                except RevealDenied as e:
                    return self._json(403, {"error": str(e)})
            return self.send_error(404)

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/judge":
                return self.send_error(404)
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                if not 0 < n <= MAX_BODY_BYTES:
                    raise JudgementError(
                        f"Content-Length must be between 1 and {MAX_BODY_BYTES}")
                raw = self.rfile.read(n)
                if len(raw) != n:
                    raise JudgementError("short body")
                body = json.loads(raw.decode("utf-8"))
                if not isinstance(body, dict):
                    raise JudgementError("body must be a JSON object")
                now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                candidate = save_judgement(
                    state["judgements"], presentations, body.get("token"),
                    body.get("choice"), body.get("lost_speech"), now)
            except (JudgementError, ValueError, TypeError, AttributeError,
                    UnicodeDecodeError) as e:
                return self._send(400, str(e).encode(), "text/plain; charset=utf-8")
            # The judgement reaches server state only after it is on disk. `/reveal`
            # opens on a committed judgement, so a visible-but-unwritten answer would
            # spend the blindness of a clip whose answer was then lost.
            tmp = Path(judgements_path).with_suffix(".json.tmp")
            try:
                tmp.write_text(json.dumps(candidate, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, judgements_path)
            except OSError as e:
                tmp.unlink(missing_ok=True)
                return self._send(500, f"not saved: {e}".encode(),
                                  "text/plain; charset=utf-8")
            state["judgements"] = candidate
            return self._json(200, {"saved": body.get("token"),
                                    "n": len(state["judgements"])})

    return H


def _tailnet_addresses() -> frozenset:
    try:
        out = subprocess.run(["tailscale", "ip"], capture_output=True, text=True,
                             timeout=5)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    if out.returncode != 0:
        return frozenset()
    return frozenset(line.strip() for line in out.stdout.splitlines() if line.strip())


def _bindable(host: str, tailnet=None) -> bool:
    """Loopback, or an address this machine holds on the tailnet. Never anything else.

    Membership of 100.64.0.0/10 is not the check: that range is carrier-grade NAT, so an
    ISP-assigned address on a real interface would pass a range test while putting
    council audio on the carrier network. Fails closed.
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
    for a in (_tailnet_addresses() if tailnet is None else tailnet):
        try:
            if ipaddress.ip_address(a.strip()) == ip:
                return True
        except ValueError:
            continue
    return False


def _session(sample: str):
    clips = load_sample(sample)
    cands = load_candidates(sample)
    return clips, cands, session_order(sample, clips, cands)


def serve(sample: str, host: str, port: int) -> None:  # pragma: no cover
    if not _bindable(host):
        raise SystemExit(
            f"refused: {host!r}. This server has no authentication and serves council "
            "audio, so it binds loopback, or an address `tailscale ip` reports for this "
            "machine, and nothing else. 0.0.0.0 would put the audio on every network "
            "this machine joins -- the same PII category as the 2026-07-21 history "
            "purge -- and a bare 100.64.0.0/10 range test would accept a carrier-grade "
            "NAT address on a real interface.")
    _clips, _cands, pres = _session(sample)
    jpath = pref_dir(sample) / "judgements.json"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    handler = make_handler(pres, sample_dir(sample) / "clips", jpath, sample)
    todo = len(remaining(pres, _read(jpath, {})))
    print(f"{sample}: {len(pres)} clips, {todo} still to judge · "
          f"http://{host}:{port}", flush=True)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), handler) as s:
        s.serve_forever()


def report(sample: str) -> dict:  # pragma: no cover - operational
    clips, cands, pres = _session(sample)
    rows = records(_read(pref_dir(sample) / "judgements.json", {}), pres)
    if not rows:
        return {"sample": sample, "n_clips": len(clips), "n_judgements": 0,
                "note": "nothing judged yet"}
    inc, new = cands["incumbent_arm"], cands["new_arm"]
    fn = decide if sample == DECIDING_SAMPLE else challenge_table
    return fn(rows, inc, new, clips)


def main() -> None:  # pragma: no cover - operational entry point
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["serve", "report", "check"])
    ap.add_argument("--sample", choices=list(SAMPLES), required=True)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    try:
        if a.mode == "serve":
            return serve(a.sample, a.host, a.port)
        if a.mode == "check":
            clips, cands, pres = _session(a.sample)
            print(json.dumps({"sample": a.sample, "n_clips": len(clips),
                              "n_presentations": len(pres),
                              "n_meetings": len({p["meeting"] for p in pres}),
                              "incumbent_arm": cands["incumbent_arm"],
                              "new_arm": cands["new_arm"]},
                             ensure_ascii=False, indent=2))
            return
        print(json.dumps(report(a.sample), ensure_ascii=False, indent=2))
    except (CandidateError, FileNotFoundError) as e:
        # An operator state, not a defect: the candidate transcripts arrive after the
        # round-3 adapter exists. A traceback would read as a broken tool.
        raise SystemExit(f"not ready: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
