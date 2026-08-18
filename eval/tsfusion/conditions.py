#!/usr/bin/env python3
"""Three conditions, so the Whisper upgrade cannot be mistaken for an anchoring win.

The established W was composed from the OLD benchmark hypotheses. This page uses the
`word_timestamps=True` decode, which is independently known to be about 1.0 WER point
and 1.0 deletion point better than the hypotheses W was built on. Any anchoring
variant built on the new decode therefore starts ahead for a reason that has nothing
to do with timestamps or diarization.

  C1  the established W: the frozen hierarchical vote over the benchmark's
      scribe / soniox / adapter hypotheses.
  C2  the SAME vote, with the adapter hypothesis replaced by the timestamped decode.
      Timestamps and diarization are not consulted. C2 - C1 is the Whisper upgrade.
  C3  a timestamp- and diarization-aware variant on EXACTLY C2's hypotheses.
      C3 - C2 is the only contrast that says anything about anchoring.

C1 and C2 do not share an MSA topology: changing one hypothesis moves column
boundaries. They are therefore never projected onto one another; each is aligned and
voted independently, and only their texts are compared.

THE C3 RULE, frozen before any number was computed
--------------------------------------------------
C3 starts from C2's output and changes columns matching every one of these:

  a) the column has exactly ONE non-epsilon entry, so the occupancy vote drops it
     (this is the deletion class, and it is where our measured damage is);
  b) the proposing system is Soniox or Whisper, i.e. a system that has timestamps
     (a Scribe-only column has no anchor to reason from and is never eligible);
  c) that system's interval for this column is STABLE and OBSERVED: transferred on a
     match invariant across all optimal alignments, from a raw word that produced
     exactly one token. Bracketed, extrapolated and derived intervals are NOT
     eligible, so inferred timing never becomes evidence for the rule consuming it;
  d) the interval lies wholly inside exactly one exclusive turn, and the regular
     diarization reports exactly one active turn across it, i.e. no overlap;
  e) the proposing decoder's confidence for the word is at least CONF_MIN.

  Action: keep the token instead of dropping it. Nothing else changes.
  Null behaviour: if no column qualifies, C3 is byte-identical to C2.

This rule was written on this data by someone who had already looked at it. It is a
post-hoc descriptive hypothesis and would need untouched audio to be a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CONF_MIN = 0.90


@dataclass
class Condition:
    name: str
    label: str
    tokens: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    n_eligible: int = 0
    n_changed: int = 0
    changes: list[dict] = field(default_factory=list)


def sdi(ref: list[str], hyp: list[str]) -> dict:
    """Substitutions, deletions, insertions and WER against the published text.

    The backtrace that splits the distance into S/D/I is one of possibly several
    optimal ones; the TOTAL is unique, the split is not. Reported as counts, with
    that caveat carried on the page.
    """
    n, m = len(ref), len(hyp)
    f = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        f[i][0] = i
    for j in range(m + 1):
        f[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            f[i][j] = min(f[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                          f[i - 1][j] + 1, f[i][j - 1] + 1)
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and f[i][j] == f[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and f[i][j] == f[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {"S": s, "D": d, "I": ins, "errors": f[n][m], "ref_tokens": n,
            "hyp_tokens": m, "wer": f[n][m] / n if n else None}


def c3_eligible(col, col_time, speaker_call, confs) -> tuple[bool, str, dict]:
    """Apply the frozen table to one C2 column. Returns (eligible, token, why)."""
    present = [(k, e) for k, e in enumerate(col) if e is not None]
    why = {}
    why["occupancy"] = len(present)
    if len(present) != 1:
        return False, "", {**why, "fail": "occupancy != 1"}
    idx, token = present[0]
    why["system_index"] = idx
    if idx == 0:
        return False, token, {**why, "fail": "Scribe-only column has no anchor"}
    system = ("scribe", "soniox", "whisper")[idx]
    src = (col_time.sources or {}).get(system)
    if col_time.time_method != "observed" or not src:
        return False, token, {**why, "fail": f"time_method={col_time.time_method}"}
    if src.get("provenance") != "observed_word":
        return False, token, {**why, "fail": f"provenance={src.get('provenance')}"}
    why["speaker_state"] = speaker_call.state
    why["multiplicity"] = speaker_call.multiplicity
    if speaker_call.state != "named" or speaker_call.overlap_fraction < 1.0 \
            or speaker_call.multiplicity != 1:
        return False, token, {**why, "fail": "not wholly inside exactly one turn"}
    conf = confs.get(system)
    why["conf"] = conf
    if conf is None or conf < CONF_MIN:
        return False, token, {**why, "fail": f"confidence {conf} < {CONF_MIN}"}
    return True, token, {**why, "fail": None}
