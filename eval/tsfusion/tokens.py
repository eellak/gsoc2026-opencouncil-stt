#!/usr/bin/env python3
"""Turn timestamped decoder output into the exact word units `msa.py` aligns.

The frozen scorer's word unit is `scoring.wtoks`: NFD, drop combining marks,
lowercase, split on `\\w+`. Neither timestamped source emits that unit:

  Soniox `stt-rt-v4` emits SUBWORD pieces. "Έχει" arrives as "Έ" + "χει";
  a leading space marks a word boundary. Punctuation rides along on a piece.
  Whisper `word_timestamps=True` emits whitespace-prefixed words with punctuation
  attached (" κριτική.").

Three things go wrong if you skip this step and pair timestamps to tokens by index:

  1. a raw word can normalise to ZERO tokens (a piece that is only "," or ".")
  2. a raw word can normalise to MORE than one token (a Greek apostrophe form like
     "απ'το", or a hyphenated form, splits on `\\w+`)
  3. a raw word can normalise to a DIFFERENT string (tonos and final sigma survive
     or vanish differently than a naive lowercase would suggest)

Every one of those shifts the whole downstream timestamp assignment by one, which is
invisible on a page and fatal to it. So: build raw words, normalise each, and when a
raw word yields several tokens, split its interval proportionally to token length —
a guess, but a declared one, and it never changes the token COUNT.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import re

from eval.controlled_eval.scoring import wtoks

_WORDCHAR = re.compile(r"\w", re.UNICODE)


@dataclass
class TimedToken:
    """One MSA-unit token with the interval its decoder gave it.

    When one raw decoder word produced several tokens, `start`/`end` are a
    character-proportional GUESS inside the raw word and `parent_start`/`parent_end`
    keep the whole raw word's interval as the uncertainty envelope. Codex's review of
    the plan was right that a proportional split must not be presented as observed:
    `provenance` says `derived_within_raw_word` in that case.
    """
    token: str                    # normalised, exactly what msa.py sees
    start: float                  # seconds, in whatever clock the caller fed in
    end: float
    raw: str                      # the raw decoder string this came from
    conf: float | None = None
    split_of: int = 1             # how many tokens the raw word produced
    split_index: int = 0
    source: str = ""
    parent_start: float | None = None
    parent_end: float | None = None

    @property
    def provenance(self) -> str:
        return "observed_word" if self.split_of == 1 else "derived_within_raw_word"

    @property
    def envelope(self) -> tuple[float, float]:
        """The interval this token is certainly inside, per its own decoder."""
        return (self.parent_start if self.parent_start is not None else self.start,
                self.parent_end if self.parent_end is not None else self.end)


def _split_interval(start: float, end: float, toks: list[str]):
    """Divide [start, end) among `toks` in proportion to their lengths."""
    if len(toks) == 1:
        return [(start, end)]
    total = sum(len(t) for t in toks) or len(toks)
    out, cur = [], start
    span = end - start
    for t in toks[:-1]:
        nxt = cur + span * (len(t) / total)
        out.append((cur, nxt))
        cur = nxt
    out.append((cur, end))
    return out


def raw_word_to_tokens(raw: str, start: float, end: float, conf: float | None = None,
                       source: str = "") -> list[TimedToken]:
    """Normalise one raw decoder word into zero or more `TimedToken`s."""
    toks = wtoks(raw)
    if not toks:
        return []
    out = []
    for i, ((s, e), t) in enumerate(zip(_split_interval(start, end, toks), toks)):
        out.append(TimedToken(token=t, start=s, end=e, raw=raw, conf=conf,
                              split_of=len(toks), split_index=i, source=source,
                              parent_start=start, parent_end=end))
    return out


# ------------------------------------------------------------------ Soniox
def soniox_words(pieces: list[dict]) -> list[dict]:
    """Group `stt-rt-v4` subword pieces into raw words.

    A piece whose text starts with a space opens a new word. Confidence of the word
    is the MINIMUM over its pieces: a word is only as trustworthy as its least
    trustworthy fragment, and taking the mean hides exactly the case (one bad piece
    in an otherwise confident word) the page is meant to expose.

    Pieces that carry no word character (a bare "," or ".") are excluded from that
    minimum. Codex flagged this: punctuation confidence is a different quantity, and
    a low-confidence comma would otherwise condemn the word it happens to follow.
    """
    words: list[dict] = []
    for p in pieces:
        txt = p.get("text", "")
        if not txt:
            continue
        opens = txt.startswith(" ") or not words
        c = float(p.get("confidence", 1.0))
        wordish = bool(_WORDCHAR.search(txt))
        if opens:
            words.append({"raw": txt.strip(), "start": p["start_ms"] / 1000.0,
                          "end": p["end_ms"] / 1000.0,
                          "conf": c if wordish else None,
                          "conf_all": c, "n_pieces": 1})
        else:
            w = words[-1]
            w["raw"] += txt
            w["start"] = min(w["start"], p["start_ms"] / 1000.0)
            w["end"] = max(w["end"], p["end_ms"] / 1000.0)
            if wordish:
                w["conf"] = c if w["conf"] is None else min(w["conf"], c)
            w["conf_all"] = min(w["conf_all"], c)
            w["n_pieces"] += 1
    for w in words:
        if w["conf"] is None:
            w["conf"] = w["conf_all"]
    return [w for w in words if w["raw"]]


def soniox_timed_tokens(pieces: list[dict], t_offset: float = 0.0) -> list[TimedToken]:
    out = []
    for w in soniox_words(pieces):
        out += raw_word_to_tokens(w["raw"], w["start"] + t_offset, w["end"] + t_offset,
                                  conf=w["conf"], source="soniox-rt")
    return out


# ------------------------------------------------------------------ Whisper
def whisper_timed_tokens(segments: list[dict], t_offset: float = 0.0) -> list[TimedToken]:
    """`decode-rw` segments[].words[] -> MSA-unit tokens."""
    out = []
    for seg in segments:
        for w in seg.get("words", []):
            out += raw_word_to_tokens(w["w"], w["s"] + t_offset, w["e"] + t_offset,
                                      conf=w.get("p"), source="whisper-rw")
    return out


# ------------------------------------------- transferring timestamps across decodes
@dataclass
class Transfer:
    """Result of moving intervals from a timestamped stream onto a target stream."""
    intervals: list[TimedToken | None]     # one slot per target token, None if unstable
    ops: list[str]                         # per target token, see `transfer_timestamps`
    candidates: list[list[TimedToken]]     # every interval an optimal path would allow
    n_target: int = 0
    n_stable: int = 0
    n_ambiguous: int = 0
    n_unmatched: int = 0


def _edit_dp(a: list[str], b: list[str]):
    n, m = len(a), len(b)
    f = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        f[i][0] = i
    for j in range(m + 1):
        f[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            f[i][j] = min(f[i - 1][j - 1] + (ai != b[j - 1]),
                          f[i - 1][j] + 1, f[i][j - 1] + 1)
    return f


def transfer_timestamps(timed: list[TimedToken], target: list[str]) -> Transfer:
    """Give each `target` token an interval only when EVERY optimal alignment agrees.

    The timestamped decode is not the decode whose text the benchmark scored: Soniox
    timestamps come from the free `stt-rt-v4` while the benchmark's soniox text is the
    paid `stt-async-v5`, and our adapter's timestamps come from a
    `word_timestamps=True` pass known to change ~7.7% of the transcript. Transferring
    an interval is therefore a PROXY, and a proxy is only worth showing when it is not
    an artefact of a tie-break.

    A single Levenshtein backtrace is not enough. With

        timed:  ναι  οχι  ναι
        target: ναι       ναι

    both `ναι` tokens have two equally optimal partners and any one backtrace picks
    one arbitrarily. So this walks the FULL optimal-edit lattice: an edge is optimal
    when `prefix[from] + cost + suffix[to] == total`. A target token gets an interval
    only when every optimal edge consuming it is a zero-cost diagonal from the SAME
    source token. Otherwise `ops[j]` is `ambiguous` (several candidate partners, all
    listed in `candidates[j]`) or `unmatched` (some optimal path substitutes or
    inserts it), and no interval is transferred.
    """
    a = [t.token for t in timed]
    b = list(target)
    n, m = len(a), len(b)
    if m == 0:
        return Transfer([], [], [], 0, 0, 0, 0)
    if n == 0:
        return Transfer([None] * m, ["unmatched"] * m, [[] for _ in range(m)],
                        m, 0, 0, m)
    fwd = _edit_dp(a, b)
    bwd = _edit_dp(a[::-1], b[::-1])          # bwd[i][j] = suffix distance a[n-i:], b[m-j:]
    total = fwd[n][m]

    def suffix(i, j):
        return bwd[n - i][m - j]

    intervals: list[TimedToken | None] = [None] * m
    ops = ["unmatched"] * m
    cands: list[list[TimedToken]] = [[] for _ in range(m)]
    for j in range(1, m + 1):
        matched_from: set[int] = set()
        other_edge = False
        for i in range(0, n + 1):
            # horizontal edge (i, j-1) -> (i, j): target token j is an insertion
            if fwd[i][j - 1] + 1 + suffix(i, j) == total:
                other_edge = True
            if i == 0:
                continue
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if fwd[i - 1][j - 1] + cost + suffix(i, j) == total:
                if cost == 0:
                    matched_from.add(i - 1)
                else:
                    other_edge = True
        cands[j - 1] = [timed[i] for i in sorted(matched_from)]
        if len(matched_from) == 1 and not other_edge:
            intervals[j - 1] = timed[next(iter(matched_from))]
            ops[j - 1] = "stable"
        elif matched_from:
            ops[j - 1] = "ambiguous"
        else:
            ops[j - 1] = "unmatched"
    return Transfer(intervals=intervals, ops=ops, candidates=cands, n_target=m,
                    n_stable=ops.count("stable"),
                    n_ambiguous=ops.count("ambiguous"),
                    n_unmatched=ops.count("unmatched"))
