"""Retrieval-biased hotwords: a prompt conditioned on the model's own first pass.

Spec: docs/specs/2026-08-26-training-round-2.md section 3.4.
Record: exp-2026-08-27-retrieval-hotwords (OPEN).

The closed static arm (exp-2026-07-25-hotwords) put ~20 roster surnames, 160 tokens,
into every window unconditionally. It lifted correctly matched name mentions
126 -> 161 of 246 and cost +133 deletions, del_rate +0.0112 [+0.0020, +0.0207], so it
was closed as do-not-ship. That record stands; this module does not reopen it.

What changes here, adapted from Fun-ASR (arXiv 2509.12508) section 5.5: the prompt is
RETRIEVED. Pass 1 runs unbiased (or is read from cache), its tokens are folded to their
Greek phonetic skeleton, and only lexicon entries within a frozen edit distance of some
token are passed to pass 2 -- often none at all, in which case no second decode is
issued and the pass-1 text stands unchanged.

Two design rules the tests enforce, both of which the arm's validity depends on:

  * pass 2 differs from pass 1 by the `hotwords` kwarg and by nothing else, under the
    identical window-only seed, so the contrast is the prompt rather than the sampling;
  * `retrieve` sees the hypothesis and the lexicon and nothing else. It has no
    reference and no scorer argument, because a reference-informed retrieval would
    measure an oracle rather than a policy.

Every knob here is frozen in the spec before any decode runs. Do not tune one after
seeing a number: `include_exact`, `max_distance`, `min_skeleton_len`, `k_max`,
`token_budget` and the choice of the STRICT or LOOSE fold are all preregistered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from eval.controlled_eval.eval_freeze import FILLER_RE
from eval.controlled_eval.greek_phonetics import phon
from eval.controlled_eval.scoring import edist, norm

# faster-whisper silently truncates hotwords at `max_length // 2 - 1` = 223 tokens
# (transcribe.py:722, :1546-1547). The closed arm asserted below that rather than
# discovering the cut in a result file; so does this one.
HOTWORD_TOKEN_CAP = 223

# The preregistered ceiling on retrieved entries (spec section 3.4).
K_MAX_CAP = 3

# Surfaces are kept as written so diagnostics name the token the model actually
# emitted. `phon` normalises case and accents internally, so no folding happens here.
_WORD_RE = re.compile(r"\w+")


@dataclass
class RetrievalConfig:
    """Frozen retrieval parameters. Every field is required on purpose.

    `loose` has no default because the two folds disagree on exactly the cases this
    arm exists to catch -- `Μαυρίδης`/`Μαβρίδης` collide under LOOSE and not under
    STRICT -- so silently inheriting one would decide the experiment by accident.
    """

    loose: bool
    max_distance: int
    min_skeleton_len: int
    k_max: int
    token_budget: int
    include_exact: bool

    def __post_init__(self) -> None:
        # K <= 3 is a preregistered hard cap, not a default. Checking it here rather
        # than at the slice means a caller cannot widen the frozen parameter by
        # accident, and a negative value cannot quietly become a from-the-end slice.
        if not isinstance(self.k_max, int) or isinstance(self.k_max, bool):
            raise TypeError(f"k_max must be an int, got {self.k_max!r}")
        if not 0 <= self.k_max <= K_MAX_CAP:
            raise ValueError(
                f"k_max must be between 0 and {K_MAX_CAP} inclusive (spec section 3.4 "
                f"freezes the cap); got {self.k_max}"
            )


@dataclass
class Retrieved:
    """One lexicon entry admitted for the second pass, with its provenance."""

    surface: str
    skeleton: str
    distance: int
    rank: int
    matched_token: str


@dataclass
class PolicyResult:
    text: str
    retrieved: list[Retrieved] = field(default_factory=list)
    pass2_ran: bool = False
    pass1_from_cache: bool = False


def _tokens(text: str) -> list[str]:
    """Hypothesis tokens, surfaces preserved, hesitation fillers dropped.

    Mirrors `eval_freeze.ftoks` in what it removes, but keeps the written form rather
    than the normalised one so `matched_token` names what the model actually emitted.
    """
    return [t for t in _WORD_RE.findall(text or "") if not FILLER_RE.match(norm(t))]


def _entries(lexicon):
    """(canonical, surface_forms) pairs from either shape of lexicon.

    A bare sequence of strings means each entry has one surface form. A mapping means
    the entry carries its recorded inflections as well -- the `aliases` of the DS-WER
    term files. Matching against the inflections is what keeps an ordinary genitive
    (`Μαυρίδη` for `Μαυρίδης`) from looking like a one-edit error and firing the
    prompt on a name the model already wrote correctly.
    """
    if isinstance(lexicon, Mapping):
        return [(k, [k] + [f for f in v if f != k]) for k, v in lexicon.items()]
    return [(e, [e]) for e in lexicon]


def retrieve(pass1_text: str, lexicon, cfg: RetrievalConfig) -> list[Retrieved]:
    """Lexicon entries within `cfg.max_distance` of some pass-1 token, in skeleton space.

    `lexicon` is either a sequence of surface forms or a mapping canonical -> forms.
    Distance to an entry is the minimum over all of its forms, so a token matching a
    recorded inflection exactly is treated as exact and (unless `include_exact`) skipped.

    Deterministic and independent of the order of `lexicon`: candidates are ranked by
    (distance, normalised surface, surface), and the rank is recorded on each result.
    """
    toks = [
        (t, phon(t, loose=cfg.loose))
        for t in _tokens(pass1_text)
    ]
    toks = [(t, sk) for t, sk in toks if len(sk) >= cfg.min_skeleton_len]
    if not toks:
        return []

    best: dict[str, tuple[int, str, str]] = {}
    for entry, forms in _entries(lexicon):
        skels = [(f, phon(f, loose=cfg.loose)) for f in forms]
        skels = [(f, sk) for f, sk in skels if len(sk) >= cfg.min_skeleton_len]
        if not skels:
            continue
        for tok, tok_skel in toks:
            d, form, sk = min(
                ((edist(list(tok_skel), list(sk)), f, sk) for f, sk in skels),
                key=lambda t: (t[0], t[1]),
            )
            if d == 0 and not cfg.include_exact:
                # The token already matches a known surface form of this entry -- the
                # nominative or one of its recorded inflections. Prompting for it
                # spends prev-context on every 30-second frame and buys nothing.
                continue
            if d > cfg.max_distance:
                continue
            prev = best.get(entry)
            if prev is None or (d, tok) < prev[:2]:
                best[entry] = (d, tok, sk)

    ordered = sorted(best.items(), key=lambda kv: (kv[1][0], norm(kv[0]), kv[0]))
    return [
        Retrieved(
            surface=entry,
            skeleton=sk,
            distance=d,
            rank=i,
            matched_token=tok,
        )
        for i, (entry, (d, tok, sk)) in enumerate(ordered[: cfg.k_max])
    ]


def fit_budget(retrieved: Sequence[Retrieved], budget: int, tokenizer) -> list[Retrieved]:
    """Greedy whole-entry inclusion under a budget counted by the real tokenizer.

    Counted the way faster-whisper counts it -- the joined string, leading space, no
    special tokens -- so the number here is the number the decoder will see. An entry
    that does not fit is dropped whole; a truncated surname is worse than an absent one.
    """
    assert 0 <= budget < HOTWORD_TOKEN_CAP, (
        f"budget {budget} must stay under faster-whisper's silent cut at "
        f"{HOTWORD_TOKEN_CAP} tokens"
    )
    kept: list[Retrieved] = []
    for r in retrieved:
        trial = kept + [r]
        text = " ".join(x.surface for x in trial)
        if len(tokenizer.encode(" " + text.strip(), add_special_tokens=False).ids) > budget:
            continue
        kept = trial
    return [Retrieved(r.surface, r.skeleton, r.distance, i, r.matched_token)
            for i, r in enumerate(kept)]


def run_policy(
    *,
    window_id: str,
    audio,
    lexicon: Sequence[str],
    cfg: RetrievalConfig,
    decoder: Callable[..., str],
    set_seed: Callable[[int], None],
    seed_for: Callable[[str], int],
    base_kwargs: dict,
    tokenizer,
    pass1_text: str | None = None,
) -> PolicyResult:
    """Run the two-pass policy on one window.

    `pass1_text` is supplied by the arm from the cached frozen-config decode
    (~/.cache/oc-public/decode-ablation/eval-A.json). It is reused rather than
    re-derived because exp-2026-08-16-adapter-confidence withdrew the bit-exact
    reproducibility claim for this local CPU decode (16 of 18, not 18 of 18). Passing
    None makes the policy produce pass 1 itself, which is what the tests exercise.
    """
    for banned in ("hotwords", "initial_prompt", "prefix"):
        if base_kwargs.get(banned) is not None:
            raise ValueError(
                f"base_kwargs must be the unbiased control; {banned!r} is set. "
                "faster-whisper ignores hotwords entirely when prefix is set."
            )

    seed = seed_for(window_id)
    pass1_from_cache = pass1_text is not None

    if pass1_text is None:
        set_seed(seed)
        pass1_text = decoder(audio, **dict(base_kwargs))

    retrieved = fit_budget(retrieve(pass1_text, lexicon, cfg), cfg.token_budget, tokenizer)
    if not retrieved:
        return PolicyResult(
            text=pass1_text,
            retrieved=[],
            pass2_ran=False,
            pass1_from_cache=pass1_from_cache,
        )

    kwargs = dict(base_kwargs)
    kwargs["hotwords"] = " ".join(r.surface for r in retrieved)
    set_seed(seed)  # common random numbers: the same draw as pass 1
    return PolicyResult(
        text=decoder(audio, **kwargs),
        retrieved=retrieved,
        pass2_ran=True,
        pass1_from_cache=pass1_from_cache,
    )
