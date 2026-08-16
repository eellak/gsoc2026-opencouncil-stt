#!/usr/bin/env python3
"""An idea-search harness with the multiplicity defence built into the API.

WHY THIS EXISTS. `fusion_lab.evaluate` made it cheap to score one idea. Cheap
scoring plus a loop is how a project ships noise: our promotion gate is a paired
clustered bootstrap CI excluding zero, and if a loop tries eighty ideas, several
clear a 95% CI by chance alone. Every expensive mistake on this project has been a
measurement mistake. This module exists to make the honest protocol the only path
through the code, not a discipline the operator has to remember.

THE PROTOCOL, frozen in `docs/specs/2026-08-16-autoresearch-partition-prereg.md`:

  1. SEARCH / CONFIRM CITY SPLIT. The ten benchmark cities are cut once, by a rule
     that reads only reference-token counts, into a 6-city search partition and a
     4-city confirmation partition. The loop iterates freely on search. The API
     ENFORCES the split: `run_search` refuses a substrate containing a confirmation
     city, and `run_confirmation` refuses anything that is not exactly the
     confirmation partition.

  2. CONFIRMATION IS ONE FROZEN BATCH PER CYCLE. Codex job 362e2a7b: "at most five"
     is not a protection if result 1 decides who gets slot 2 — the later hypothesis is
     then no longer independent of confirmation data. Codex job 59c9564 sharpened it:
     five SEQUENTIAL singleton batches, each Holm-corrected inside itself, give a
     familywise error of 1 - 0.95^5 = 22.6%. So a cycle freezes EXACTLY ONE batch, of
     at most CONFIRM_BUDGET ideas, before any confirmation number is computed; a
     second batch is refused outright and needs a new PROTOCOL_VERSION.

  3. FITTING. Confirmation parameters are fitted once on the ENTIRE search partition
     and frozen. Applying them to the confirmation cities is a locked-box run, not
     another cross-fitted estimate. Search itself is leave-one-search-city-out.

  4. INFERENCE. The p-value fed to the multiplicity correction is a NULL-IMPOSED,
     studentized wild cluster bootstrap-t over meetings (Cameron/Gelbach/Miller),
     not the percentile bootstrap's tail mass. The percentile CI of `fusion_lab` is
     still reported beside it, unchanged.

  5. MULTIPLICITY. Holm at familywise alpha over the frozen confirmation batch is the
     ship gate — it bounds the probability that ANY shipped idea is false, which is
     the question. Benjamini-Hochberg is reported alongside, over the confirmation
     batch and over the whole search family, the latter purely as a fishing
     diagnostic. BH's FDR guarantee needs independence or PRDS, which sharing
     bootstrap weights does NOT establish for arbitrary ideas scored on the same
     meetings, so BH here is descriptive and Holm is the gate. Both always carry
     their denominator.

  6. THE SHIP TEST IS A MINIMUM-EFFECT TEST. A monotone arm that fixes one token in
     each of four meetings gets a CI excluding zero regardless of magnitude —
     measured, see `docs/reports/2026-08-16-harness-coverage-mde.md`. Testing against
     zero is therefore the wrong test. What Holm corrects is a ONE-SIDED test of
     H0: delta >= -MIN_EFFECT, so an idea ships only when the data reject a
     smaller-than-useful effect. On top of that an idea must touch enough meetings
     with a NONZERO error delta and must not have half its effect come from one
     meeting. The two-sided test against zero is still reported, and it is what the
     search-family BH diagnostic uses.

  7. BEHAVIOURAL DEDUP. An idea is fingerprinted by what it DOES — the canonical edit
     events between its output and W's, keyed-hashed. A cosmetic variant of an idea
     already evaluated is refused before it can reach confirmation.

WHAT NONE OF THIS BUYS. The ideas are proposed by agents that have already seen five
passes over these same 247 windows, including the confirmation cities. Sample
splitting controls the multiplicity of TESTING; it cannot undo the adaptivity of
PROPOSING. And four cities are four cities: a confirmation pass licenses a claim
about meetings inside these four, not about the next Greek municipality.

Journal: `research/autoresearch/journal.jsonl`, append-only, hash-chained, counts
only, with a checkpoint sidecar so that deleting the TAIL is detected too. It is a
tamper-EVIDENCE device against accident and forgetfulness, not a security boundary:
the chain is unkeyed, so anyone willing to recompute it can rewrite the whole file.
Firing-set fingerprints live under $SC — they are derived from council speech and
never enter git.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import inspect
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.fusion_lab import (Idea, Substrate,         # noqa: E402
                                             evaluate, log, sc)

# --------------------------------------------------------------- frozen protocol
PROTOCOL_VERSION = "autoresearch-2026-08-16b"

#: The split, computed by `plan_partition` from reference-token counts alone and then
#: pinned here so it cannot drift. Both partitions contain a city that contributed
#: nothing to fine-tuning (orestiada in search, argos in confirm).
SEARCH_CITIES = ("athens", "chalandri", "chania", "orestiada", "vrilissia", "zografou")
CONFIRM_CITIES = ("argos", "samothraki", "sparta", "xylokastro")
CONFIRM_TOKEN_SHARE_TARGET = 0.35

ALPHA = 0.05                     # familywise, Holm, over the frozen confirm batch
MIN_EFFECT = 0.0010              # 0.10 WER points: below this we do not care
MIN_SUPPORT_MEETINGS = 8         # meetings with a NONZERO error delta, search
MIN_SUPPORT_MEETINGS_CONFIRM = 6
MAX_DOMINATION = 0.50            # max_b |d_b| / sum_b |d_b|
DEDUP_JACCARD = 0.90             # >= this vs any earlier idea is a cosmetic variant
CONFIRM_BUDGET = 5               # one-way doors available this cycle
R_WILD = int(os.environ.get("R_WILD", "9999"))
DEDUP_KEY = os.environ.get(
    "AUTORESEARCH_DEDUP_KEY", "opencouncil-autoresearch-firing-v1").encode()
#: Firing sets keyed with a different key are not comparable, so the key's identity
#: is journalled and a mismatch fails closed rather than silently letting a
#: previously refused variant back in.
DEDUP_KEY_ID = hashlib.sha256(DEDUP_KEY).hexdigest()[:12]

JOURNAL = ROOT / "research/autoresearch/journal.jsonl"

REGISTERED = "REGISTERED"
SEARCH_RESULT = "SEARCH_RESULT"
DUPLICATE_REFUSED = "DUPLICATE_REFUSED"
CONFIRM_BATCH_FROZEN = "CONFIRM_BATCH_FROZEN"
CONFIRM_RESULT = "CONFIRM_RESULT"


# ------------------------------------------------------------------- partition
def plan_partition(sub: Substrate, target: float = CONFIRM_TOKEN_SHARE_TARGET):
    """The split rule, replayable: greedy balance on reference tokens, nothing else.

    Cities are sorted by reference-token count descending and each is given to
    whichever side is furthest below its token target. No WER, no idea, no outcome
    enters this function, which is the only reason the confirmation partition is
    worth anything.
    """
    tot: dict[str, int] = {}
    for w in sub.windows:
        tot[w.city] = tot.get(w.city, 0) + len(w.ref)
    total = sum(tot.values())
    want = {"search": total * (1.0 - target), "confirm": total * target}
    have = {"search": 0, "confirm": 0}
    out: dict[str, list[str]] = {"search": [], "confirm": []}
    for city, n in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0])):
        side = max(("search", "confirm"), key=lambda s: (want[s] - have[s], s))
        out[side].append(city)
        have[side] += n
    return tuple(sorted(out["search"])), tuple(sorted(out["confirm"]))


def assert_partition(sub: Substrate) -> None:
    """The pinned split must be exactly what the rule produces on this substrate."""
    s, c = plan_partition(sub)
    if (s, c) != (SEARCH_CITIES, CONFIRM_CITIES):
        raise AssertionError(
            f"partition drift: rule gives search={s} confirm={c}, "
            f"pinned search={SEARCH_CITIES} confirm={CONFIRM_CITIES}")


@dataclass(frozen=True)
class Partitions:
    """The two city sets, injected so the runner can ENFORCE them, not trust callers.

    Codex 59c9564: with the split living only in the caller, nothing stopped a caller
    from searching on confirmation cities or passing the same substrate twice. The
    runner now validates every substrate against this object.
    """
    search: tuple[str, ...] = SEARCH_CITIES
    confirm: tuple[str, ...] = CONFIRM_CITIES

    def __post_init__(self):
        if set(self.search) & set(self.confirm):
            raise ValueError("search and confirm partitions overlap")
        if not self.search or not self.confirm:
            raise ValueError("both partitions must be non-empty")

    def check_search(self, sub: Substrate) -> None:
        got = {w.city for w in sub.windows}
        if got & set(self.confirm):
            raise ValueError(
                f"search substrate contains confirmation cities "
                f"{sorted(got & set(self.confirm))} — the split would be for nothing")
        if not got <= set(self.search):
            raise ValueError(f"unknown cities in search substrate: "
                             f"{sorted(got - set(self.search))}")

    def check_confirm(self, sub: Substrate) -> None:
        got = {w.city for w in sub.windows}
        if got != set(self.confirm):
            raise ValueError(
                f"confirmation substrate is {sorted(got)}, not the frozen "
                f"confirmation partition {sorted(self.confirm)} — a hand-picked "
                "subset is not a locked box")


DEFAULT_PARTITIONS = Partitions()


def _restrict(sub: Substrate, cities) -> Substrate:
    cities = set(cities)
    ws = [w for w in sub.windows if w.city in cities]
    if not ws:
        raise ValueError(f"no windows for cities {sorted(cities)}")
    meta = dict(sub.meta)
    meta.update({"partition_cities": sorted(cities), "n_windows": len(ws),
                 "n_meetings": len({w.meeting for w in ws}),
                 "n_cities": len({w.city for w in ws}),
                 "ref_tokens": sum(len(w.ref) for w in ws)})
    return Substrate(ws, meta=meta)


def search_partition(sub: Substrate) -> Substrate:
    assert_partition(sub)
    return _restrict(sub, SEARCH_CITIES)


def confirm_partition(sub: Substrate) -> Substrate:
    assert_partition(sub)
    return _restrict(sub, CONFIRM_CITIES)


# ------------------------------------------------------------------- inference
def cluster_contributions(rows_arm, rows_w, clusters):
    """Per meeting: (error delta vs W, reference tokens), in sorted cluster order."""
    d: dict[str, float] = {}
    n: dict[str, float] = {}
    for a, b, k in zip(rows_arm, rows_w, clusters):
        d[k] = d.get(k, 0.0) + (a[0] + a[1] + a[2]) - (b[0] + b[1] + b[2])
        n[k] = n.get(k, 0.0) + a[3]
    keys = sorted(d)
    return keys, np.array([d[k] for k in keys]), np.array([n[k] for k in keys])


def rademacher(n_clusters: int, r: int = R_WILD, seed: int = 20260816) -> np.ndarray:
    """One weight matrix, shared by every idea in a batch, so their tests stay joint."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=(r, n_clusters))


def wild_cluster_test(d, n, delta0: float = 0.0, weights: np.ndarray | None = None,
                      r: int = R_WILD, seed: int = 20260816,
                      alternative: str = "two-sided") -> dict:
    """Null-imposed studentized wild cluster bootstrap-t on dWER = sum(d)/sum(n).

    The percentile bootstrap of `scoring.cluster_bootstrap` resamples the observed
    data, so its distribution is centred on the observed effect and its tail mass is
    not a p-value under H0. Here the null is imposed: the meeting contributions are
    re-signed around `delta0`, the statistic is studentized with a cluster-robust
    standard error, and

        p = (1 + #{|T*| >= |T_obs|}) / (R + 1)

    which is the finite-Monte-Carlo-valid form. `delta0` non-zero tests a MINIMUM
    EFFECT rather than a bare zero, which is the test we actually care about;
    `alternative="less"` makes that test one-sided (H0: delta >= delta0), which is
    the form the ship gate uses.
    """
    if alternative not in ("two-sided", "less"):
        raise ValueError("alternative must be 'two-sided' or 'less'")
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    if d.shape != n.shape or d.ndim != 1 or d.size == 0:
        raise ValueError("d and n must be equal-length 1-d arrays")
    N = n.sum()
    if N <= 0:
        raise ValueError("no reference tokens")
    delta = float(d.sum() / N)
    resid = d - delta0 * n                        # null-imposed residuals
    se = float(np.sqrt(((d - delta * n) ** 2).sum()) / N)
    if se == 0.0:
        # Every meeting sits exactly on the fitted line. That is the A/A case when the
        # estimate equals the null — but it ALSO happens when every meeting carries the
        # same non-zero rate (Codex 59c9564), which is the opposite situation and must
        # not be handed back as p=1.
        if delta == delta0:
            p_deg = 1.0
        elif alternative == "less" and delta > delta0:
            p_deg = 1.0
        else:
            p_deg = 1.0 / (r + 1)
        return {"delta": delta, "delta0": delta0, "se": 0.0, "t_obs": 0.0,
                "p": p_deg, "alternative": alternative, "n_clusters": int(d.size),
                "r": int(r), "degenerate": True}
    t_obs = (delta - delta0) / se
    w = weights if weights is not None else rademacher(d.size, r, seed)
    if w.shape[1] != d.size:
        raise ValueError(f"weights have {w.shape[1]} clusters, data has {d.size}")
    r = w.shape[0]
    dstar = delta0 * n[None, :] + resid[None, :] * w          # (R, B)
    delta_star = dstar.sum(axis=1) / N
    se_star = np.sqrt(((dstar - delta_star[:, None] * n[None, :]) ** 2).sum(axis=1)) / N
    num = delta_star - delta0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.where(se_star > 0, num / np.where(se_star > 0, se_star, 1.0),
                          np.where(num == 0, 0.0, np.inf))
    if alternative == "less":
        hits = int((t_star <= t_obs).sum())
    else:
        hits = int((np.abs(t_star) >= abs(t_obs)).sum())
    p = float((1 + hits) / (r + 1))
    return {"delta": delta, "delta0": delta0, "se": se, "t_obs": float(t_obs),
            "p": p, "alternative": alternative, "n_clusters": int(d.size),
            "r": int(r), "degenerate": False}


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict:
    """Holm-Bonferroni. Controls P(any false rejection) — the ship question."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return {"m": m, "alpha": alpha, "adjusted": adj,
            "reject": {k: bool(v <= alpha) for k, v in adj.items()}}


def benjamini_hochberg(pvals: dict[str, float], q: float = ALPHA) -> dict:
    """BH-FDR. Reported beside Holm; it bounds the false FRACTION, not the risk."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    adj, running = {}, 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        running = min(running, min(1.0, m * p / (i + 1)))
        adj[k] = running
    return {"m": m, "q": q, "adjusted": adj,
            "reject": {k: bool(v <= q) for k, v in adj.items()}}


# ---------------------------------------------------------------- firing sets
def edit_events(base: list[str], arm: list[str]) -> list[str]:
    """Canonical edit events turning W's stream into the arm's, as opaque strings.

    Raw output indices are unstable — one insertion shifts every later position and
    makes two nearly identical ideas look unrelated (Codex 362e2a7b). Events are
    therefore anchored to BASE positions and carry an ordinal for repeated insertions
    at one anchor.
    """
    n, m = len(base), len(arm)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        bi = base[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1,
                         prev[j - 1] + (bi != arm[j - 1]))
    events: list[str] = []
    i, j, ins_ord = n, m, {}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (base[i - 1] != arm[j - 1]):
            if base[i - 1] != arm[j - 1]:
                events.append(f"S|{i - 1}|{arm[j - 1]}")
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            events.append(f"D|{i - 1}")
            i -= 1
        else:
            k = ins_ord.get(i, 0)
            ins_ord[i] = k + 1
            events.append(f"I|{i}|{k}|{arm[j - 1]}")
            j -= 1
    return events


def firing_set(out_tokens: dict[str, list[str]], w_tokens: dict[str, list[str]],
               key: bytes = DEDUP_KEY) -> list[str]:
    """Keyed hashes of every edit the idea makes, sorted. No token survives this."""
    hashes = set()
    for wid, arm in out_tokens.items():
        base = w_tokens[wid]
        if arm == base:
            continue
        for ev in edit_events(base, arm):
            hashes.add(hmac.new(key, f"{wid}|{ev}".encode(), hashlib.sha256)
                       .hexdigest()[:16])
    return sorted(hashes)


def minhash(hashes: list[str], k: int = 128) -> list[str]:
    """A cheap shortlist signature. Enforcement uses the exact sets, never this."""
    if not hashes:
        return []
    vals = np.array([int(h, 16) for h in hashes], dtype=object)
    sig = []
    for i in range(k):
        salt = (i * 0x9E3779B97F4A7C15 + 1) & 0xFFFFFFFFFFFFFFFF
        sig.append(format(min(int(v) ^ salt for v in vals), "016x"))
    return sig


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    u = len(sa | sb)
    return len(sa & sb) / u if u else 1.0


def _firing_path(idea_key: str) -> Path:
    return sc() / "autoresearch" / "firing" / f"{idea_key}.json"


def save_firing(idea_key: str, hashes: list[str]) -> Path:
    p = _firing_path(idea_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"hashes": hashes, "minhash": minhash(hashes)}))
    return p


def load_firing(idea_key: str) -> list[str] | None:
    p = _firing_path(idea_key)
    if not p.exists():
        return None
    return json.loads(p.read_text())["hashes"]


def set_digest(hashes: list[str]) -> str:
    return hashlib.sha256("".join(hashes).encode()).hexdigest()[:32]


# --------------------------------------------------------------------- journal
def _canon(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class JournalCorrupt(RuntimeError):
    pass


class Journal:
    """Append-only, hash-chained JSONL with a tail checkpoint.

    A leaderboard whose losing entries can be quietly deleted is a leaderboard
    without a denominator, which is the specific dishonesty this file prevents.

    The chain alone only protects records that have a SUCCESSOR — deleting the last
    line, or the last ten, leaves a perfectly valid prefix (Codex 59c9564). So a
    sidecar `.head` records the count and the hash of the last record, written under
    the same lock, and `records()` refuses a journal that is shorter than its
    checkpoint. It is tamper EVIDENCE, not a security boundary: the chain is unkeyed
    and both files are writable, so a determined actor can recompute both.

    Use `transaction()` when a decision depends on what is already in the journal.
    Reading, deciding and appending outside one lock is how a confirmation gets spent
    twice by two concurrent processes.
    """

    def __init__(self, path: Path = JOURNAL):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.head_path = self.path.with_suffix(".head")

    # -- checkpoint --------------------------------------------------------
    def _read_head(self) -> dict | None:
        if not self.head_path.exists():
            return None
        try:
            return json.loads(self.head_path.read_text())
        except json.JSONDecodeError as e:
            raise JournalCorrupt(f"checkpoint is not JSON: {e}") from e

    def _write_head(self, count: int, head: str) -> None:
        tmp = self.head_path.with_suffix(".head.tmp")
        tmp.write_text(json.dumps({"count": count, "head": head,
                                   "protocol": PROTOCOL_VERSION}))
        os.replace(tmp, self.head_path)

    def records(self) -> list[dict]:
        recs, prev = self._records_unchecked()
        ck = self._read_head()
        if ck is not None:
            if len(recs) < ck["count"]:
                raise JournalCorrupt(
                    f"journal has {len(recs)} records, checkpoint says {ck['count']} — "
                    "the tail was truncated")
            if len(recs) == ck["count"] and prev != ck["head"]:
                raise JournalCorrupt("last record does not match its checkpoint")
        elif recs:
            raise JournalCorrupt(
                f"{self.head_path.name} is missing but the journal has "
                f"{len(recs)} records — restore it or start a new cycle")
        return recs

    def _records_unchecked(self):
        if not self.path.exists():
            return [], "genesis"
        recs, prev = [], "genesis"
        for ln, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise JournalCorrupt(f"line {ln} is not JSON: {e}") from e
            if rec.get("seq") != len(recs) + 1:
                raise JournalCorrupt(
                    f"line {ln}: seq {rec.get('seq')} != {len(recs) + 1} — "
                    "a record was inserted, removed or reordered")
            if rec.get("prev") != prev:
                raise JournalCorrupt(
                    f"line {ln}: hash chain broken — an earlier record was edited")
            prev = _rec_hash(rec)
            recs.append(rec)
        return recs, prev

    @contextmanager
    def transaction(self):
        """Hold the journal lock across read-decide-append. Use it for invariants."""
        self.path.touch()
        with open(self.path, "r+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield _Txn(self, fh)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def append(self, rec: dict) -> dict:
        with self.transaction() as tx:
            return tx.append(rec)


def _rec_hash(rec: dict) -> str:
    return hashlib.sha256(_canon(rec).encode()).hexdigest()[:32]


class _Txn:
    """The journal, seen from inside its lock."""

    def __init__(self, journal: Journal, fh):
        self.j, self.fh = journal, fh

    def records(self) -> list[dict]:
        return self.j.records()

    def append(self, rec: dict) -> dict:
        recs = self.j.records()
        prev = _rec_hash(recs[-1]) if recs else "genesis"
        full = dict(rec)
        full["seq"] = len(recs) + 1
        full["prev"] = prev
        full.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.fh.seek(0, os.SEEK_END)
        self.fh.write(_canon(full) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.j._write_head(full["seq"], _rec_hash(full))
        return full


# -------------------------------------------------------------------- registry
def _git_tree() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _src(obj) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return repr(obj)


def impl_fingerprint(factory) -> dict:
    """More than the decision function: the transitive artifact it rides on.

    Codex 362e2a7b — a decision function can call a helper that changed underneath
    it, so the identity of an idea pins the factory source, the source of every class
    in its MRO (the shared `apply`/`fit` scaffolding lives there), this module,
    `fusion_lab`, the frozen partition and the commit.

    KNOWN LIMITS, stated rather than hidden:

    - A change to a free helper FUNCTION in the caller's module is not caught. Hashing
      the caller's whole module would catch it, but it would also change the identity
      of every idea in a library whenever one new idea is added to it, which would make
      the confirmation journal unusable across sessions. Ideas must therefore keep
      their decision logic inside the class.
    - This module's own bytes are NOT hashed; `PROTOCOL_VERSION` stands for them. A
      behavioural change to the harness MUST bump `PROTOCOL_VERSION`, which re-keys
      every idea on purpose. Hashing the bytes instead would re-key every idea for a
      comment fix.
    """
    parts = [PROTOCOL_VERSION.encode(), _src(factory).encode()]
    for klass in getattr(factory, "__mro__", ())[1:]:
        if klass is object:
            continue
        parts.append(_src(klass).encode())
    parts.append((ROOT / "eval/controlled_eval/fusion_lab.py").read_bytes())
    parts.append(",".join(SEARCH_CITIES).encode())
    parts.append(",".join(CONFIRM_CITIES).encode())
    h = hashlib.sha256()
    for part in parts:
        h.update(hashlib.sha256(part).digest())
    return {"impl_sha256": h.hexdigest()[:32],
            "factory_sha256": hashlib.sha256(_src(factory).encode()).hexdigest()[:16],
            "module": getattr(factory, "__module__", "?"),
            "git_head": _git_tree()}


@dataclass(frozen=True)
class Handle:
    """Proof that a hypothesis was written down before a number existed."""
    name: str
    hypothesis: str
    idea_key: str
    impl_sha256: str
    registered_at: str


class Registry:
    """Register first, run second. The runner will not accept anything else."""

    def __init__(self, journal: Journal | None = None,
                 partitions: Partitions = DEFAULT_PARTITIONS):
        self.journal = journal or Journal()
        self.partitions = partitions
        self._factories: dict[str, object] = {}

    # -- registration ------------------------------------------------------
    def register(self, name: str, hypothesis: str, factory,
                 gates: dict | None = None, poor_bet: str | None = None,
                 resume: bool = False) -> Handle:
        """Write the hypothesis down, get a handle. Nothing runs without one.

        `resume=True` re-derives the handle of an idea already in the journal. It is
        safe precisely because the key is a hash of the hypothesis AND the
        implementation: if either moved, the key moves and this is a new idea. It
        does not let a result be re-reported — `run_search` still refuses that.
        """
        if not hypothesis or len(hypothesis.split()) < 4:
            raise ValueError("an idea needs a real one-line hypothesis, written "
                             "before its result exists")
        fp = impl_fingerprint(factory)
        payload = {"name": name, "hypothesis": hypothesis, "gates": gates or {},
                   "protocol": PROTOCOL_VERSION, "impl": fp["impl_sha256"]}
        key = hashlib.sha256(_canon(payload).encode()).hexdigest()[:16]
        prior = [r for r in self.journal.records()
                 if r["type"] == REGISTERED and r["idea_key"] == key]
        if prior:
            if not resume:
                raise ValueError(f"{name}: already registered as {key}")
            self._factories[key] = factory
            return Handle(name=name, hypothesis=hypothesis, idea_key=key,
                          impl_sha256=fp["impl_sha256"], registered_at=prior[0]["ts"])
        rec = self.journal.append({
            "type": REGISTERED, "idea_key": key, "name": name,
            "hypothesis": hypothesis, "gates": gates or {},
            "poor_bet": poor_bet, "protocol": PROTOCOL_VERSION, **fp})
        self._factories[key] = factory
        return Handle(name=name, hypothesis=hypothesis, idea_key=key,
                      impl_sha256=fp["impl_sha256"], registered_at=rec["ts"])

    def searched(self) -> set[str]:
        return {r["idea_key"] for r in self.journal.records()
                if r["type"] == SEARCH_RESULT}

    def _idea(self, h: Handle) -> Idea:
        factory = self._factories.get(h.idea_key)
        if factory is None:
            raise ValueError(f"{h.name}: no factory in this process for {h.idea_key}")
        if impl_fingerprint(factory)["impl_sha256"] != h.impl_sha256:
            raise ValueError(
                f"{h.name}: implementation changed after registration — "
                "register the new behaviour as a new idea")
        return factory()

    # -- search ------------------------------------------------------------
    def run_search(self, h: Handle, sub: Substrate, n_boot: int = 2000,
                   weights: np.ndarray | None = None) -> dict:
        """Evaluate on the search partition. Free to repeat; never decides anything."""
        if not isinstance(h, Handle):
            raise TypeError("run_search takes the Handle that register() returned")
        self.partitions.check_search(sub)
        recs = self.journal.records()
        if not any(r["type"] == REGISTERED and r["idea_key"] == h.idea_key
                   for r in recs):
            raise ValueError(f"{h.name}: not registered")
        if any(r["type"] == SEARCH_RESULT and r["idea_key"] == h.idea_key
               for r in recs):
            raise ValueError(f"{h.name}: already searched; re-running would let a "
                             "result be reported twice")
        res = evaluate(self._idea(h), sub, fold="city", n_boot=n_boot,
                       return_detail=True)
        detail = res.pop("detail")
        keys, d, n = cluster_contributions(detail["rows_arm"], detail["rows_W"],
                                           detail["meetings"])
        wild = wild_cluster_test(d, n, 0.0, weights=weights)
        wild_min = wild_cluster_test(d, n, -MIN_EFFECT, weights=weights,
                                     alternative="less")
        hashes = firing_set(detail["out_tokens"], detail["w_tokens"])
        save_firing(h.idea_key, hashes)

        summary = self._summary(res, wild, wild_min, d, n, keys, hashes,
                                MIN_SUPPORT_MEETINGS)
        summary.update({"type": SEARCH_RESULT, "idea_key": h.idea_key,
                        "name": h.name, "partition": "search",
                        "n_windows": res["n_windows"], "n_folds": res["n_folds"],
                        "dedup_key_id": DEDUP_KEY_ID})
        # One transaction: the duplicate verdict is decided and committed inside the
        # SEARCH_RESULT record itself, so a crash cannot leave a known duplicate with
        # no refusal beside it (Codex 59c9564).
        with self.journal.transaction() as tx:
            dup = self._duplicate_of(h.idea_key, hashes, tx.records())
            summary["duplicate_of"] = dup
            if dup:
                summary["screen"] = {"pass": False,
                                     "reason": f"cosmetic variant of {dup}"}
            tx.append(summary)
            if dup:
                tx.append({"type": DUPLICATE_REFUSED, "idea_key": h.idea_key,
                           "name": h.name, "duplicate_of": dup,
                           "jaccard_threshold": DEDUP_JACCARD})
        return summary

    def _duplicate_of(self, key: str, hashes: list[str], recs) -> str | None:
        """Exact Jaccard against every earlier evaluated idea; MinHash only shortlists.

        FAILS CLOSED. If an earlier idea's firing set is not on disk, or was keyed
        with a different secret, the 0.90 threshold cannot be evaluated and this
        raises rather than waving the new idea through — purging $SC must not be a way
        to resubmit a refused variant (Codex 59c9564).
        """
        if not hashes:
            return None                      # the null arm is not anybody's variant
        for r in recs:
            if r["type"] != SEARCH_RESULT or r["idea_key"] == key:
                continue
            if r.get("firing_size", 0) == 0:
                continue
            if r.get("dedup_key_id") != DEDUP_KEY_ID:
                raise RuntimeError(
                    f"idea {r['idea_key']} was fingerprinted under dedup key "
                    f"{r.get('dedup_key_id')}, this process uses {DEDUP_KEY_ID}; "
                    "the sets are not comparable — restore AUTORESEARCH_DEDUP_KEY")
            if r.get("firing_digest") == set_digest(hashes):
                return r["idea_key"]
            other = load_firing(r["idea_key"])
            if other is None:
                raise RuntimeError(
                    f"firing set for {r['idea_key']} is missing from $SC; dedup "
                    "cannot be enforced. Restore the cache or start a new cycle "
                    "under a new PROTOCOL_VERSION.")
            if jaccard(hashes, other) >= DEDUP_JACCARD:
                return r["idea_key"]
        return None

    @staticmethod
    def _summary(res, wild, wild_min, d, n, keys, hashes, min_support) -> dict:
        touched = int((d != 0).sum())
        absd = float(np.abs(d).sum())
        domination = float(np.abs(d).max() / absd) if absd > 0 else 0.0
        worst = keys[int(np.argmax(np.abs(d)))] if len(keys) else None
        g = res["gates"]
        screen = {
            "wer_improves": bool(g["wer_improves"]),
            "del_rate_gate": bool(g["del_rate_gate"]["pass"]),
            "ins_rate_gate": bool(g["ins_rate_gate"]["pass"]),
            "effect_floor": bool(wild["delta"] <= -MIN_EFFECT),
            "support": bool(touched >= min_support),
            "no_domination": bool(domination < MAX_DOMINATION),
            "ci_excludes_zero": bool(g["wer_ci_excludes_zero"]),
            # The one that actually decides: a one-sided rejection of "the effect is
            # smaller than useful". The point-estimate floor above is noisy on its
            # own (Codex 59c9564) and is kept only as a cheap search screen.
            "min_effect_test": bool(wild_min["p"] <= ALPHA
                                    and wild_min["delta"] < -MIN_EFFECT),
        }
        screen["pass"] = all(screen.values())
        return {
            "wer": res["out_of_fold"]["wer"], "wer_W": res["baseline_W"]["wer"],
            "del_rate": res["out_of_fold"]["del_rate"],
            "ins_rate": res["out_of_fold"]["ins_rate"],
            "sub_rate": res["out_of_fold"]["sub_rate"],
            "dwer": wild["delta"],
            "percentile_ci95": res["vs_W"]["wer"]["ci95"],
            "wild_p": wild["p"], "wild_t": wild["t_obs"], "wild_se": wild["se"],
            "wild_p_min_effect": wild_min["p"],
            "wild_p_min_effect_alternative": wild_min.get("alternative"),
            "min_effect": MIN_EFFECT,
            "meetings_touched": touched, "n_meetings": len(keys),
            "domination": domination, "domination_meeting": worst,
            "windows_changed_vs_W": res["windows_changed_vs_W"],
            "firing_size": len(hashes), "firing_digest": set_digest(hashes),
            "loo_sign_flips": {k: res["loo_vs_W"][k]["sign_flips"]
                               for k in ("window", "meeting", "city")},
            "cities_better": res["per_city"]["cities_better"],
            "cities_worse": res["per_city"]["cities_worse"],
            "screen": screen,
        }

    # -- confirmation ------------------------------------------------------
    def freeze_confirmation_batch(self, handles: list[Handle], note: str = "") -> str:
        """Close the confirmation family BEFORE any confirmation number exists.

        Selecting who gets slot 2 after seeing slot 1's result would make slot 2's
        hypothesis dependent on confirmation data, and the whole split would be for
        nothing. So the batch is named once, journalled, and cannot be extended.

        EXACTLY ONE BATCH PER CYCLE. Five sequential singleton batches, each
        Holm-corrected inside itself, give a familywise error of 1 - 0.95^5 = 22.6%
        (Codex 59c9564). A second batch under the same PROTOCOL_VERSION is refused.

        The whole check-and-commit runs inside one journal transaction so two
        processes cannot both see an unspent budget.
        """
        if not handles:
            raise ValueError("an empty confirmation batch is not a batch")
        with self.journal.transaction() as tx:
            recs = tx.records()
            batches = [r for r in recs if r["type"] == CONFIRM_BATCH_FROZEN]
            if batches:
                raise ValueError(
                    f"a confirmation batch ({batches[0]['batch_id']}) was already "
                    "frozen in this cycle; a second family would break the familywise "
                    "guarantee. Bump PROTOCOL_VERSION to start a new cycle.")
            searched = {r["idea_key"] for r in recs if r["type"] == SEARCH_RESULT}
            refused = {r["idea_key"] for r in recs if r["type"] == DUPLICATE_REFUSED
                       } | {r["idea_key"] for r in recs
                            if r["type"] == SEARCH_RESULT and r.get("duplicate_of")}
            keys = []
            for h in handles:
                if h.idea_key not in searched:
                    raise ValueError(f"{h.name}: never evaluated on search")
                if h.idea_key in refused:
                    raise ValueError(f"{h.name}: refused as a cosmetic variant")
                keys.append(h.idea_key)
            if len(set(keys)) != len(keys):
                raise ValueError("duplicate handle in the batch")
            if len(keys) > CONFIRM_BUDGET:
                raise ValueError(
                    f"confirmation budget is {CONFIRM_BUDGET}, batch asks for "
                    f"{len(keys)}")
            batch_id = hashlib.sha256("|".join(sorted(keys)).encode()).hexdigest()[:12]
            tx.append({"type": CONFIRM_BATCH_FROZEN, "batch_id": batch_id,
                       "idea_keys": keys,
                       "names": [h.name for h in handles], "note": note,
                       "budget": CONFIRM_BUDGET,
                       "confirm_cities": list(self.partitions.confirm)})
        return batch_id

    def run_confirmation(self, batch_id: str, handles: list[Handle],
                         search_sub: Substrate, confirm_sub: Substrate,
                         n_boot: int = 10000) -> dict:
        """The one-way door. Params frozen on all of search, applied to confirm."""
        self.partitions.check_search(search_sub)
        self.partitions.check_confirm(confirm_sub)
        recs = self.journal.records()
        batch = [r for r in recs if r["type"] == CONFIRM_BATCH_FROZEN
                 and r["batch_id"] == batch_id]
        if not batch:
            raise ValueError(f"no frozen batch {batch_id}")
        allowed = set(batch[0]["idea_keys"])
        by_key = {h.idea_key: h for h in handles}
        if allowed - set(by_key):
            raise ValueError("handles do not cover the frozen batch")
        done = {r["idea_key"] for r in recs if r["type"] == CONFIRM_RESULT}
        # Checked for the WHOLE batch before any confirmation window is read, so a
        # refusal never happens halfway through and leaves the partition touched.
        for key in sorted(allowed & done):
            raise ValueError(f"{by_key[key].name}: confirmation already spent")
        weights = rademacher(len({w.meeting for w in confirm_sub.windows}))
        results = {}
        for key in sorted(allowed):
            h = by_key[key]
            idea = self._idea(h)
            params = idea.fit(list(search_sub.windows))       # fitted once, frozen
            frozen = _Frozen(idea, params, h.name)
            res = evaluate(frozen, confirm_sub, fold="city", n_boot=n_boot,
                           return_detail=True)
            detail = res.pop("detail")
            keys, d, n = cluster_contributions(detail["rows_arm"], detail["rows_W"],
                                               detail["meetings"])
            wild = wild_cluster_test(d, n, 0.0, weights=weights)
            wild_min = wild_cluster_test(d, n, -MIN_EFFECT, weights=weights,
                                         alternative="less")
            hashes = firing_set(detail["out_tokens"], detail["w_tokens"])
            s = self._summary(res, wild, wild_min, d, n, keys, hashes,
                              MIN_SUPPORT_MEETINGS_CONFIRM)
            s.update({"type": CONFIRM_RESULT, "idea_key": key, "name": h.name,
                      "batch_id": batch_id, "partition": "confirm",
                      "n_windows": res["n_windows"],
                      "fit_note": "parameters fitted once on the whole search "
                                  "partition and frozen; no refit on confirm"})
            results[key] = s
        # The ship family is the ONE-SIDED MINIMUM-EFFECT test, not the test against
        # zero: an idea ships only when the data reject "smaller than useful".
        pv_ship = {k: v["wild_p_min_effect"] for k, v in results.items()}
        pv_zero = {k: v["wild_p"] for k, v in results.items()}
        hm, bh = holm(pv_ship), benjamini_hochberg(pv_ship)
        bh_zero = benjamini_hochberg(pv_zero)
        for k, v in results.items():
            v["holm_adjusted_p"] = hm["adjusted"][k]
            v["bh_adjusted_p"] = bh["adjusted"][k]
            v["bh_adjusted_p_vs_zero"] = bh_zero["adjusted"][k]
            v["family_size"] = hm["m"]
            v["ship_test"] = f"one-sided H0: dWER >= {-MIN_EFFECT}, Holm at {ALPHA}"
            v["ship"] = bool(hm["reject"][k] and v["screen"]["pass"])
            self.journal.append(v)
        return {"batch_id": batch_id, "holm": hm, "bh": bh, "bh_vs_zero": bh_zero,
                "results": results}

    # -- reporting ---------------------------------------------------------
    def leaderboard(self) -> dict:
        """Every idea ever tried, with the denominator that makes it readable."""
        recs = self.journal.records()
        reg = [r for r in recs if r["type"] == REGISTERED]
        sr = [r for r in recs if r["type"] == SEARCH_RESULT]
        cr = [r for r in recs if r["type"] == CONFIRM_RESULT]
        dup = [r for r in recs if r["type"] == DUPLICATE_REFUSED]
        search_bh = benjamini_hochberg({r["idea_key"]: r["wild_p"] for r in sr}) \
            if sr else {"m": 0, "adjusted": {}, "reject": {}}
        return {
            "protocol": PROTOCOL_VERSION,
            "denominator": {
                "registered": len(reg), "searched": len(sr),
                "duplicate_refused": len(dup),
                "screen_passed": sum(1 for r in sr if r["screen"]["pass"]),
                "confirmed": len(cr),
                "confirmation_budget": CONFIRM_BUDGET,
                "confirmation_batches_frozen": sum(
                    1 for r in recs if r["type"] == CONFIRM_BATCH_FROZEN),
                "confirmations_remaining": CONFIRM_BUDGET - len(
                    {k for r in recs if r["type"] == CONFIRM_BATCH_FROZEN
                     for k in r["idea_keys"]}),
            },
            "search": sorted(
                ({"name": r["name"], "idea_key": r["idea_key"], "dwer": r["dwer"],
                  "wild_p": r["wild_p"],
                  "search_bh_adjusted_p": search_bh["adjusted"].get(r["idea_key"]),
                  "percentile_ci95": r["percentile_ci95"],
                  "screen": r["screen"]["pass"],
                  "duplicate_of": r.get("duplicate_of")} for r in sr),
                key=lambda x: x["dwer"]),
            "search_family_bh": {
                "m": search_bh["m"],
                "note": "reported as a fishing diagnostic only; the ship gate is "
                        "Holm over the frozen confirmation batch",
            },
            "confirm": [{"name": r["name"], "dwer": r["dwer"], "wild_p": r["wild_p"],
                         "holm_adjusted_p": r["holm_adjusted_p"],
                         "bh_adjusted_p": r["bh_adjusted_p"],
                         "family_size": r["family_size"],
                         "screen": r["screen"]["pass"], "ship": r["ship"]}
                        for r in cr],
        }


class _Frozen(Idea):
    """An idea with its parameters already fitted on search. `fit` is inert."""
    fitted = False

    def __init__(self, inner: Idea, params, name: str):
        self.inner, self.params, self.name = inner, params, name

    def fit(self, train):
        return self.params

    def apply(self, w, params):
        return self.inner.apply(w, self.params)


def summary_line(s: dict) -> str:
    sc_ = s["screen"]
    return (f"{s['name']:26s} dWER={s['dwer']:+.5f} p={s['wild_p']:.4f} "
            f"pmin={s['wild_p_min_effect']:.4f} "
            f"ci={[round(x, 5) for x in s['percentile_ci95']]} "
            f"meetings={s['meetings_touched']}/{s['n_meetings']} "
            f"dom={s['domination']:.2f} screen={'PASS' if sc_['pass'] else 'fail'}")


if __name__ == "__main__":
    from eval.controlled_eval.fusion_lab import load_substrate
    s = load_substrate()
    assert_partition(s)
    log(json.dumps({"search": search_partition(s).meta,
                    "confirm": confirm_partition(s).meta}, indent=1))
    log(json.dumps(Registry().leaderboard()["denominator"], indent=1))
