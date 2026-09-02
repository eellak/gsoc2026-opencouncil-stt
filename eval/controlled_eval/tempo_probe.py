"""Does feeding Soniox slower recover words lost to crosstalk?

Spec: docs/specs/2026-08-28-tempo-probe.md. Record: exp-2026-08-28-tempo-probe.
Plan reviewed by Codex db4d8ac8.

Five arms over the 15 gold cells that carry human-marked overlap, three interleaved
cycles each, scored as overlap EPISODES rather than serial WER, because two people
talking at once have no single correct word order.

Three things here are load-bearing and each fixes a mistake that was actually made:

1. A stretched timestamp maps back by MULTIPLICATION. At atempo=0.7 the clip gets
   longer, so an event at 10 s comes back near 14.29 s; dividing would have placed
   every word at twice its real time.
2. Unmatched hypothesis words are NOT charged as insertions. The human reference was
   written listening at 1.0x, so a word the slowdown genuinely recovers can be absent
   from it. Penalising that would reject the phenomenon under test; those words go to
   blind adjudication instead.
3. The noise floor is the spread of EACH arm across its own cycles, not one
   control-versus-control repeat, which is a single draw rather than an estimate.

Usage:
    python3 -m eval.controlled_eval.tempo_probe plan
    python3 -m eval.controlled_eval.tempo_probe run [--jobs 6] [--cells N]
    python3 -m eval.controlled_eval.tempo_probe score
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.scoring import greek_normalize  # noqa: E402
from eval.soniox_confidence_probe import group_words  # noqa: E402

SC = Path(os.environ.get("SC", Path.home() / ".cache" / "oc-public"))
GOLD = SC / "gold-set-2026-08"
CLIPS = SC / "gold-set" / "clips"
OUT = SC / "tempo-probe-2026-08"
COUNTS_PATH = ROOT / "eval" / "results_tempo_probe.json"

SONIOX_DIR = Path(os.environ.get("SONIOX_TOOLS_DIR",
                                 Path.home() / "projects" / "soniox-tools"))
SONIOX_PY = Path(os.environ.get("SONIOX_TOOLS_PYTHON", SONIOX_DIR / ".venv/bin/python"))

SEED = 20260828
CYCLES = 3
# Held to what the spec froze. `raw` exists to prove atempo=1.0 is a no-op; `pace070`
# changes only how fast the client pushes bytes, so it isolates streaming behaviour
# from acoustics.
DUR_TOLERANCE_S = 0.25


@dataclass(frozen=True)
class Arm:
    tempo: float          # atempo factor applied to the audio, 1.0 = untouched
    feed: float           # client playback multiplier into the realtime socket
    note: str


ARMS: dict[str, Arm] = {
    "raw":     Arm(1.0, 1.0, "untouched wav, real-time feed"),
    "proc10":  Arm(1.0, 1.0, "atempo=1.0, the control every arm is compared against"),
    "t085":    Arm(0.85, 1.0, "conservative stretch"),
    "t070":    Arm(0.70, 1.0, "aggressive stretch"),
    "pace070": Arm(1.0, 0.7, "acoustics untouched, fed at 0.7x wall clock"),
}
PROCESSED = ("proc10", "t085", "t070")   # arms whose audio goes through ffmpeg
CONTROL = "proc10"


# ----------------------------------------------------------------- geometry

def to_original_ms(returned_ms: float, tempo: float) -> float:
    """Map a timestamp in stretched audio back to real time.

    At tempo r the output is 1/r as long, so an event at t appears at t/r and comes
    back by multiplying. Dividing here is the bug this function exists to prevent.
    """
    return returned_ms * tempo


def expected_dur(orig_dur_s: float, tempo: float) -> float:
    return orig_dur_s / tempo


# ----------------------------------------------------------------- episodes

@dataclass(frozen=True)
class Episode:
    block_ids: tuple
    start: float
    end: float
    certain_words: list
    uncertain_words: list


@dataclass(frozen=True)
class Tok:
    text: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class Match:
    matched: int
    n_ref: int
    unmatched_hyp: list
    n_hyp: int


def _words(text: str) -> list:
    return [w for w in greek_normalize(text or "").split() if w]


def episodes(blocks: list) -> list:
    """Connected components of the human `ov_with` graph, as timed word bags.

    Components, not pairs: a block can overlap two different blocks, and scoring the
    pairs separately would let one hypothesis word satisfy both. The graph is made
    symmetric first, because the collection page recorded the link on one side only in
    some cells.
    """
    by_id = {b["id"]: b for b in blocks}
    adj = collections.defaultdict(set)
    for b in blocks:
        for other in b.get("ov_with") or []:
            if other in by_id:
                adj[b["id"]].add(other)
                adj[other].add(b["id"])

    seen: set = set()
    out = []
    for b in blocks:
        bid = b["id"]
        if bid in seen or bid not in adj:
            continue
        stack, comp = [bid], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj[cur] - seen)
        comp = [c for c in (bb["id"] for bb in blocks) if c in comp]  # stable order
        certain, uncertain = [], []
        for cid in comp:
            blk = by_id[cid]
            (uncertain if blk.get("text_unc") else certain).extend(_words(blk["text"]))
        out.append(Episode(
            block_ids=tuple(comp),
            start=min(by_id[c]["s"] for c in comp),
            end=max(by_id[c]["e"] for c in comp),
            certain_words=certain,
            uncertain_words=uncertain,
        ))
    return out


def score_episode(ep: Episode, hyp: list) -> Match:
    """One-to-one multiset matching of hypothesis tokens inside the episode span.

    One-to-one so a word said twice is not satisfied once, and so a single hypothesis
    token cannot be credited to two speakers.
    """
    want = collections.Counter(ep.certain_words)
    inside = [t for t in hyp
              if ep.start <= ((t.start_ms + t.end_ms) / 2000.0) <= ep.end]
    matched, unmatched = 0, []
    for t in inside:
        w = greek_normalize(t.text).strip()
        if not w:
            continue
        if want[w] > 0:
            want[w] -= 1
            matched += 1
        else:
            unmatched.append(w)
    return Match(matched=matched, n_ref=len(ep.certain_words),
                 unmatched_hyp=unmatched, n_hyp=len(inside))


# ----------------------------------------------------------------- run plan

@dataclass(frozen=True)
class Job:
    cell: str
    arm: str
    cycle: int


def run_plan(cells: list, cycles: int = CYCLES, seed: int = SEED) -> list:
    """Every arm, every cell, `cycles` times, interleaved and shuffled per cycle.

    Blocking by arm would confound the arm with whatever the server was doing that
    minute; the shuffle is seeded so a resumed run keeps the same order.
    """
    rng = random.Random(seed)
    jobs = []
    for cyc in range(cycles):
        block = [Job(c, a, cyc) for c in cells for a in ARMS]
        rng.shuffle(block)
        jobs.extend(block)
    return jobs


# ----------------------------------------------------------------- audio

def stretched_path(cell: str, arm: str) -> Path:
    return OUT / "audio" / f"{cell}__{arm}.wav"


def probe_duration(path: Path) -> float | None:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, timeout=120)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def make_audio(cell: str, arm: str) -> Path:
    """Materialise the arm's audio. Non-processed arms use the original file itself."""
    src = CLIPS / f"{cell}.wav"
    if not src.exists():
        raise FileNotFoundError(src)
    if arm not in PROCESSED:
        return src
    dst = stretched_path(cell, arm)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tempo = ARMS[arm].tempo
    tmp = dst.with_suffix(".part.wav")
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-filter:a", f"atempo={tempo}", "-ac", "1", "-ar", "16000", str(tmp)],
        capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ffmpeg failed for {cell} {arm}")
    want = expected_dur(probe_duration(src) or 0.0, tempo)
    got = probe_duration(tmp) or 0.0
    if abs(got - want) > DUR_TOLERANCE_S:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{cell} {arm}: stretched to {got:.2f}s, expected {want:.2f}s. A filter "
            "that silently did nothing would look like a null result.")
    tmp.replace(dst)
    return dst


# ----------------------------------------------------------------- transcribe

def token_path(cell: str, arm: str, cycle: int) -> Path:
    return OUT / "tokens" / f"{cell}__{arm}__c{cycle}.json"


def transcribe(wav: Path, feed: float, timeout: int = 900) -> dict:
    cmd = [str(SONIOX_PY), "file_transcribe.py", str(wav), "--lang", "el", "--json",
           "--realtime", "--speed", str(feed)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=str(SONIOX_DIR), errors="replace")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"rc={r.returncode} stderr={r.stderr[-300:]!r}")
    return json.loads(r.stdout)


def run_job(job: Job, attempts: int = 3) -> tuple:
    dst = token_path(job.cell, job.arm, job.cycle)
    if dst.exists():
        return job, True, "cached"
    wav = make_audio(job.cell, job.arm)
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            d = transcribe(wav, ARMS[job.arm].feed)
        except Exception as e:                                   # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(5 * attempt)
            continue
        d["arm"] = job.arm
        d["cycle"] = job.cycle
        d["cell"] = job.cell
        d["tempo"] = ARMS[job.arm].tempo
        d["feed"] = ARMS[job.arm].feed
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        tmp.write_text(json.dumps(d, ensure_ascii=False))
        tmp.replace(dst)
        return job, True, f"{len(d.get('tokens') or [])} tok"
    return job, False, last


# ----------------------------------------------------------------- scoring

def load_cells() -> dict:
    answers = json.loads((GOLD / "answers-user-2026-08-16.json").read_text())
    scored = set(json.loads((GOLD / "manifest.json").read_text())["scored_cell_ids"])
    out = {}
    for cid, rec in answers.items():
        if cid not in scored:
            continue
        blocks = (rec.get("b") or {}).get("blocks") or []
        if not any(b.get("ov_with") for b in blocks):
            continue
        if not (CLIPS / f"{cid}.wav").exists():
            continue
        out[cid] = blocks
    return out


def meeting_of(cell_id: str) -> str:
    """`gc_<city>_<meeting>_<offset>` -> `<city>__<meeting>`."""
    body = cell_id[3:].rsplit("_", 1)[0]
    city, meeting = body.split("_", 1)
    return f"{city}__{meeting}"


def words_from_tokens(tokens: list, tempo: float) -> list:
    """Soniox returns SUB-WORD pieces, not words: "Κ", "υ", "ρ", "ία", " Αγ".

    The grouping is NOT reimplemented here. `soniox_confidence_probe.group_words` already
    mirrors the production Go client (`soniox-core internal/soniox/soniox.go`): finals
    only, explode each piece to runes carrying its confidence, cut words on whitespace.
    The first version of this module wrote its own loop and treated each piece as a word,
    so gold WORDS were matched against ASR PIECES and every number was meaningless.
    Harold caught it by reading the adjudication page and saying the items were not
    words. Reusing the existing function is the actual fix; a second implementation of
    the same rule is how the two drift apart again.

    The only thing added here is the tempo mapping, which `group_words` cannot know
    about: a word's span is converted from stretched time back to original time.
    """
    words, stats = group_words(tokens)
    dropped = stats["words_without_timestamp"] + stats["words_without_confidence"]
    if dropped:
        # group_words skips a word with no confidence or no start time. That is right
        # for a confidence study and wrong for a recall study, where a silently missing
        # word is a false deletion charged to whichever arm produced it.
        raise ValueError(
            f"group_words dropped {dropped} words ({stats}); recall would count them as "
            "deletions. Handle them explicitly rather than scoring around them.")
    return [Tok(w["raw"],
                to_original_ms(w["start"] * 1000.0, tempo),
                to_original_ms(w["end"] * 1000.0, tempo))
            for w in words if w["raw"].strip()]


def tokens_of(cell: str, arm: str, cycle: int) -> list:
    """The arm's WORDS for one cell and cycle, in original time."""
    p = token_path(cell, arm, cycle)
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return words_from_tokens(d.get("tokens") or [],
                             d.get("tempo", ARMS[arm].tempo))


def noise_envelope(per_cycle: dict) -> float:
    """The widest within-arm spread across cycles -- every arm, not just the control."""
    spreads = [max(v) - min(v) for v in per_cycle.values() if len(v) > 1]
    return max(spreads) if spreads else 0.0


def verdict(gain: float, envelope: float, meetings_agreeing: int, n_meetings: int,
            non_overlap_loss: float, max_loss: float = 0.01) -> dict:
    why = []
    if gain <= envelope:
        why.append(f"gain {gain:.4f} is inside the noise envelope {envelope:.4f}")
    if meetings_agreeing < max(4, (n_meetings * 2) // 3):
        why.append(f"only {meetings_agreeing}/{n_meetings} meetings agree in direction")
    if non_overlap_loss > max_loss:
        why.append(f"non-overlap recall fell {non_overlap_loss:.4f} > {max_loss}")
    return {"advance": not why, "why": "; ".join(why) or "all three conditions met"}


def repo_summary(arms: dict, envelope: float, cells: int, meetings: int) -> dict:
    """Counts only. Council speech never enters git -- the words themselves stay in the
    cache and reach Harold only through the adjudication page."""
    return {
        "experiment": "exp-2026-08-28-tempo-probe",
        "spec": "docs/specs/2026-08-28-tempo-probe.md",
        "model_requested": "stt-rt-v5",
        "cells": cells, "meetings": meetings, "cycles": CYCLES,
        "noise_envelope": round(envelope, 5),
        "arms": {
            name: {
                "recall": round(v["recall"], 5),
                "n_ref": v["n_ref"],
                "matched": v["matched"],
                "n_unmatched_hyp": len(v.get("unmatched_hyp") or []),
            } for name, v in arms.items()
        },
    }


def score() -> dict:
    cells = load_cells()
    eps = {c: episodes(b) for c, b in cells.items()}
    per_arm_cycle = collections.defaultdict(list)
    totals = collections.defaultdict(lambda: {"matched": 0, "n_ref": 0,
                                              "unmatched_hyp": [], "unc_matched": 0,
                                              "unc_ref": 0, "out_matched": 0,
                                              "out_ref": 0})
    per_meeting = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"matched": 0, "n_ref": 0}))

    for arm in ARMS:
        for cycle in range(CYCLES):
            m = n = 0
            for cell, cell_eps in eps.items():
                hyp = tokens_of(cell, arm, cycle)
                if not hyp:
                    continue
                for ep in cell_eps:
                    r = score_episode(ep, hyp)
                    m += r.matched
                    n += r.n_ref
                    totals[arm]["matched"] += r.matched
                    totals[arm]["n_ref"] += r.n_ref
                    totals[arm]["unmatched_hyp"].extend(r.unmatched_hyp)
                    unc = Episode(ep.block_ids, ep.start, ep.end,
                                  ep.uncertain_words, [])
                    ru = score_episode(unc, hyp)
                    totals[arm]["unc_matched"] += ru.matched
                    totals[arm]["unc_ref"] += ru.n_ref
                    pm = per_meeting[meeting_of(cell)][arm]
                    pm["matched"] += r.matched
                    pm["n_ref"] += r.n_ref
                # collateral guard: the same cell's blocks that overlap nothing
                covered = {b for e in cell_eps for b in e.block_ids}
                for blk in cells[cell]:
                    if blk["id"] in covered or blk.get("text_unc"):
                        continue
                    solo = Episode((blk["id"],), blk["s"], blk["e"],
                                   _words(blk["text"]), [])
                    ro = score_episode(solo, hyp)
                    totals[arm]["out_matched"] += ro.matched
                    totals[arm]["out_ref"] += ro.n_ref
            if n:
                per_arm_cycle[arm].append(m / n)

    def rate(a, k_num, k_den):
        d = totals[a][k_den]
        return (totals[a][k_num] / d) if d else 0.0

    arms_out = {a: {"recall": rate(a, "matched", "n_ref"),
                    "n_ref": totals[a]["n_ref"], "matched": totals[a]["matched"],
                    "unmatched_hyp": totals[a]["unmatched_hyp"]} for a in ARMS}
    env = noise_envelope(per_arm_cycle)
    ctrl = arms_out[CONTROL]["recall"]
    ctrl_out = rate(CONTROL, "out_matched", "out_ref")

    detail = {}
    for a in ARMS:
        if a == CONTROL:
            continue
        agree = sum(1 for mt, d in per_meeting.items()
                    if d[a]["n_ref"] and d[CONTROL]["n_ref"]
                    and (d[a]["matched"] / d[a]["n_ref"])
                    > (d[CONTROL]["matched"] / d[CONTROL]["n_ref"]))
        gain = arms_out[a]["recall"] - ctrl
        loss = ctrl_out - rate(a, "out_matched", "out_ref")
        detail[a] = {
            "gain_vs_control": round(gain, 5),
            "non_overlap_loss": round(loss, 5),
            "meetings_better": agree,
            "uncertain_recall": round(rate(a, "unc_matched", "unc_ref"), 5),
            "per_cycle_recall": [round(x, 5) for x in per_arm_cycle.get(a, [])],
            **verdict(gain, env, agree, len(per_meeting), loss),
        }

    summary = repo_summary(arms_out, env, len(cells), len(per_meeting))
    summary["control"] = CONTROL
    summary["control_non_overlap_recall"] = round(ctrl_out, 5)
    summary["per_cycle_recall"] = {a: [round(x, 5) for x in v]
                                   for a, v in per_arm_cycle.items()}
    summary["arms_vs_control"] = detail
    COUNTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    (OUT / "score.json").write_text(json.dumps(
        {"summary": summary,
         "unmatched_by_arm": {a: totals[a]["unmatched_hyp"] for a in ARMS}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# ----------------------------------------------------------------- entry point

def main() -> None:  # pragma: no cover - operational
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "run", "score"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cells", type=int, default=None)
    a = ap.parse_args()

    cells = sorted(load_cells())
    if a.cells:
        cells = cells[:a.cells]
    plan = run_plan(cells)

    if a.mode == "plan":
        print(json.dumps({
            "cells": len(cells), "meetings": len({meeting_of(c) for c in cells}),
            "arms": {k: vars(v) for k, v in ARMS.items()},
            "cycles": CYCLES, "sessions": len(plan),
            "done": sum(1 for j in plan
                        if token_path(j.cell, j.arm, j.cycle).exists()),
        }, ensure_ascii=False, indent=2))
        return

    if a.mode == "score":
        print(json.dumps(score(), ensure_ascii=False, indent=2))
        return

    if not SONIOX_PY.exists():
        sys.exit(f"no soniox-tools interpreter at {SONIOX_PY}")
    from concurrent.futures import ThreadPoolExecutor
    todo = [j for j in plan if not token_path(j.cell, j.arm, j.cycle).exists()]
    print(f"{len(plan)} sessions, {len(todo)} to run, jobs={a.jobs}", flush=True)
    t0, failed, done = time.time(), [], 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for job, ok, msg in ex.map(run_job, todo):
            done += 1
            if not ok:
                failed.append((job.cell, job.arm, job.cycle, msg[:120]))
            if done % 10 == 0 or not ok:
                print(f"  [{done}/{len(todo)}] {job.arm} {job.cycle} "
                      f"{'ok' if ok else 'FAIL ' + msg[:80]}", flush=True)
    (OUT / "run_log.json").write_text(json.dumps({
        "sessions": len(plan), "ran": len(todo), "failed": failed,
        "wall_s": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s, {len(failed)} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
