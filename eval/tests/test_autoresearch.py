"""What must hold before the autoresearch loop's leaderboard is worth reading.

The harness is dangerous in one specific way: it makes a 95% CI cheap, and a loop
that tries eighty ideas gets several by chance. These tests are the evaluation of the
defence, not of the ideas:

- the partition rule is outcome-blind, deterministic and a true partition;
- the wild cluster bootstrap p-value is null-imposed and calibrated, and an A/A arm
  returns exactly 1;
- Holm and BH agree with hand-computed values, and Holm is the stricter one;
- a rewritten copy of an evaluated idea is refused by the firing-set guard;
- the journal cannot be edited, truncated or reordered without failing on load;
- confirmation is a one-way door that survives a process restart, cannot be extended
  after the batch is frozen, and cannot exceed its budget;
- in an end-to-end simulation with pure-noise ideas, the unadjusted rate of "some
  idea is significant" is high and the shipped rate is not.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import autoresearch as A                      # noqa: E402
from eval.controlled_eval.fusion_lab import Idea, Substrate, Window     # noqa: E402


# ------------------------------------------------------------------ substrate
def synth(n_cities=2, meetings_per_city=4, windows_per_meeting=2, n_tok=12):
    """A tiny substrate with the shape `evaluate` needs and no Greek in it."""
    ws = []
    for c in range(n_cities):
        city = f"city{c}"
        for m in range(meetings_per_city):
            meeting = f"{city}/m{m}"
            for k in range(windows_per_meeting):
                toks = [f"w{c}{m}{k}{i}" for i in range(n_tok)]
                ws.append(Window(
                    item_id=f"{meeting}/{k}", city=city, meeting=meeting,
                    ref=list(toks), hyps=[list(toks)] * 3, pivot=0,
                    cols=[(t, t, t) for t in toks],
                    decisions=[{"col": i, "token": t, "reason": "agree"}
                               for i, t in enumerate(toks)],
                    w_tokens=list(toks), v_tokens=list(toks), in_training=False))
    return Substrate(ws, meta={"n_windows": len(ws)})


class BreakFirstToken(Idea):
    """Deterministically corrupts the first token of every window (always worse)."""
    fitted = False
    name = "break"

    def apply(self, w, params):
        out = list(w.w_tokens)
        out[0] = "XXX"
        return out


class BreakFirstTokenRestyled(BreakFirstToken):
    """Identical behaviour, different code — the dedup guard must catch it."""
    name = "break_restyled"

    def apply(self, w, params):
        return ["XXX"] + [t for t in w.w_tokens[1:]]


class BreakSecondToken(BreakFirstToken):
    name = "break2"

    def apply(self, w, params):
        out = list(w.w_tokens)
        out[1] = "YYY"
        return out


TEST_PARTS = A.Partitions(search=("city0", "city1"), confirm=("city2", "city3"))


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setenv("SC", str(tmp_path / "cache"))
    monkeypatch.setattr(A, "R_WILD", 199)
    return A.Registry(A.Journal(tmp_path / "journal.jsonl"), partitions=TEST_PARTS)


def search_sub():
    return synth(n_cities=2)


def confirm_sub():
    s = synth(n_cities=4)
    return A._restrict(s, TEST_PARTS.confirm)


# ------------------------------------------------------------------ partition
def test_partition_rule_is_outcome_blind_and_a_true_partition():
    sub = synth(n_cities=6, meetings_per_city=3)
    s, c = A.plan_partition(sub)
    assert set(s) & set(c) == set()
    assert set(s) | set(c) == {w.city for w in sub.windows}
    assert (s, c) == A.plan_partition(sub), "the rule must be deterministic"


def test_pinned_partition_matches_the_rule_on_the_real_substrate():
    """The pinned cities are exactly what the token-balance rule produces."""
    pytest.importorskip("numpy")
    from eval.controlled_eval.fusion_lab import load_substrate
    try:
        sub = load_substrate()
    except Exception as e:                       # benchmark cache not on this machine
        pytest.skip(f"substrate unavailable: {e}")
    A.assert_partition(sub)
    assert set(A.SEARCH_CITIES) & set(A.CONFIRM_CITIES) == set()
    assert A.search_partition(sub).meta["n_cities"] == len(A.SEARCH_CITIES)
    assert A.confirm_partition(sub).meta["n_cities"] == len(A.CONFIRM_CITIES)


# ------------------------------------------------------------------ inference
def test_all_zero_arm_is_never_significant():
    d = np.zeros(20)
    n = np.full(20, 100.0)
    r = A.wild_cluster_test(d, n, 0.0, r=199)
    assert r["p"] == 1.0 and r["delta"] == 0.0 and r["degenerate"]


def test_wild_cluster_p_is_not_anticonservative_under_the_null():
    """Simulated null: the rate of p<0.05 must sit near 0.05, not above it."""
    rng = np.random.default_rng(11)
    B_ = 60
    n = rng.integers(80, 400, B_).astype(float)
    hits, trials = 0, 300
    for _ in range(trials):
        d = rng.normal(0.0, 1.0, B_) * np.sqrt(n)      # mean zero, unequal clusters
        if A.wild_cluster_test(d, n, 0.0, r=299)["p"] < 0.05:
            hits += 1
    assert hits / trials < 0.10, f"null rejection rate {hits / trials:.3f} too high"


def test_monotone_sparse_arm_is_caught_by_the_effect_and_support_floors():
    """The artifact the floors exist for: tiny, one-sided, spread over few meetings."""
    n = np.full(100, 500.0)
    d = np.zeros(100)
    d[:5] = -1.0                                   # one token fixed in five meetings
    r = A.wild_cluster_test(d, n, 0.0, r=999)
    assert abs(r["delta"]) < A.MIN_EFFECT, "effect floor must reject this magnitude"
    assert int((d != 0).sum()) < A.MIN_SUPPORT_MEETINGS


def test_holm_and_bh_match_hand_computed_values():
    p = {"a": 0.01, "b": 0.02, "c": 0.04}
    h = A.holm(p)
    assert h["adjusted"] == {"a": 0.03, "b": pytest.approx(0.04), "c": pytest.approx(0.04)}
    b = A.benjamini_hochberg(p)
    assert b["adjusted"]["a"] == pytest.approx(0.03)
    assert b["adjusted"]["c"] == pytest.approx(0.04)
    for k in p:
        assert h["adjusted"][k] >= b["adjusted"][k], "Holm must never be looser than BH"


def test_holm_is_monotone_in_family_size():
    assert A.holm({"a": 0.02})["adjusted"]["a"] == pytest.approx(0.02)
    assert A.holm({f"i{i}": 0.02 for i in range(5)})["adjusted"]["i0"] == pytest.approx(0.1)


# ---------------------------------------------------------------- firing sets
def test_edit_events_are_anchored_to_base_positions():
    base = ["a", "b", "c", "d"]
    assert A.edit_events(base, base) == []
    subbed = A.edit_events(base, ["a", "X", "c", "d"])
    assert subbed == ["S|1|X"]
    # an insertion at the front must not renumber the later substitution
    both = A.edit_events(base, ["Z", "a", "X", "c", "d"])
    assert "S|1|X" in both and any(e.startswith("I|0|") for e in both)


def test_firing_set_is_behavioural_and_carries_no_text():
    w = {"i1": ["αλφα", "βητα"]}
    f1 = A.firing_set({"i1": ["αλφα", "ΞΞΞ"]}, w)
    f2 = A.firing_set({"i1": ["αλφα", "ΞΞΞ"]}, w)
    f3 = A.firing_set({"i1": ["αλφα", "ΨΨΨ"]}, w)
    assert f1 == f2 and f1 != f3
    assert A.firing_set({"i1": list(w["i1"])}, w) == []
    blob = json.dumps({"hashes": f1, "minhash": A.minhash(f1)}, ensure_ascii=False)
    assert not any(ord(ch) > 127 for ch in blob), "no Greek may survive into a journal"
    assert all(len(h) == 16 and all(ch in "0123456789abcdef" for ch in h) for h in f1)
    assert A.firing_set({"i1": ["αλφα", "ΞΞΞ"]}, w, key=b"other") != f1


def test_jaccard_and_dedup_threshold():
    assert A.jaccard(["a"], ["a"]) == 1.0
    assert A.jaccard(["a", "b"], ["a", "c"]) == pytest.approx(1 / 3)
    assert A.jaccard([], []) == 1.0


# -------------------------------------------------------------------- journal
def test_journal_chain_detects_edit_truncation_and_reorder(tmp_path):
    j = A.Journal(tmp_path / "j.jsonl")
    for i in range(3):
        j.append({"type": A.REGISTERED, "idea_key": f"k{i}", "name": f"n{i}",
                  "hypothesis": "x", "gates": {}})
    assert len(j.records()) == 3

    lines = (tmp_path / "j.jsonl").read_text().splitlines()
    edited = json.loads(lines[0])
    edited["name"] = "tampered"
    (tmp_path / "j.jsonl").write_text("\n".join([json.dumps(edited, sort_keys=True,
                                                            ensure_ascii=False,
                                                            separators=(",", ":"))]
                                                + lines[1:]) + "\n")
    with pytest.raises(A.JournalCorrupt):
        j.records()

    (tmp_path / "j.jsonl").write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(A.JournalCorrupt):
        j.records()

    (tmp_path / "j.jsonl").write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    with pytest.raises(A.JournalCorrupt):
        j.records()


def test_deleting_the_tail_is_detected(tmp_path):
    """The chain alone accepts any prefix; the checkpoint is what closes that hole."""
    j = A.Journal(tmp_path / "j.jsonl")
    for i in range(4):
        j.append({"type": A.REGISTERED, "idea_key": f"k{i}", "name": "n",
                  "hypothesis": "x", "gates": {}})
    lines = (tmp_path / "j.jsonl").read_text().splitlines()
    for k in range(1, len(lines)):
        (tmp_path / "j.jsonl").write_text("\n".join(lines[:k]) + "\n")
        with pytest.raises(A.JournalCorrupt, match="truncated"):
            j.records()
    (tmp_path / "j.jsonl").write_text("\n".join(lines) + "\n")
    assert len(j.records()) == 4


def test_a_journal_without_its_checkpoint_is_refused(tmp_path):
    j = A.Journal(tmp_path / "j.jsonl")
    j.append({"type": A.REGISTERED, "idea_key": "k", "name": "n",
              "hypothesis": "x", "gates": {}})
    j.head_path.unlink()
    with pytest.raises(A.JournalCorrupt, match="missing"):
        j.records()


# ------------------------------------------------------------------- registry
def test_run_search_requires_the_handle_register_returned(reg):
    sub = search_sub()
    with pytest.raises(TypeError):
        reg.run_search("break", sub)


def test_hypothesis_must_be_written_before_anything_runs(reg):
    with pytest.raises(ValueError):
        reg.register("x", "too short", BreakFirstToken)


def test_an_idea_cannot_be_searched_twice(reg):
    sub = search_sub()
    h = reg.register("break", "corrupting one token per window must raise WER",
                     BreakFirstToken)
    reg.run_search(h, sub, n_boot=50)
    with pytest.raises(ValueError, match="already searched"):
        reg.run_search(h, sub, n_boot=50)


def test_implementation_change_after_registration_is_refused(reg, monkeypatch):
    sub = search_sub()
    h = reg.register("break", "corrupting one token per window must raise WER",
                     BreakFirstToken)
    object.__setattr__(h, "impl_sha256", "0" * 32)
    with pytest.raises(ValueError, match="implementation changed"):
        reg.run_search(h, sub, n_boot=50)


def test_cosmetic_variant_is_refused_by_the_firing_set_guard(reg):
    sub = search_sub()
    h1 = reg.register("break", "corrupting one token per window must raise WER",
                      BreakFirstToken)
    reg.run_search(h1, sub, n_boot=50)
    h2 = reg.register("break_restyled",
                      "the same corruption written differently must be refused",
                      BreakFirstTokenRestyled)
    s2 = reg.run_search(h2, sub, n_boot=50)
    assert s2["duplicate_of"] == h1.idea_key
    assert s2["screen"]["pass"] is False
    assert any(r["type"] == A.DUPLICATE_REFUSED for r in reg.journal.records())
    with pytest.raises(ValueError, match="cosmetic variant"):
        reg.freeze_confirmation_batch([h2])


def test_a_genuinely_different_idea_is_not_refused(reg):
    sub = search_sub()
    h1 = reg.register("break", "corrupting one token per window must raise WER",
                      BreakFirstToken)
    reg.run_search(h1, sub, n_boot=50)
    h2 = reg.register("break2", "corrupting a different token is a different idea",
                      BreakSecondToken)
    assert reg.run_search(h2, sub, n_boot=50)["duplicate_of"] is None


def test_null_arm_is_never_a_duplicate_and_never_ships(reg):
    sub = search_sub()
    h = reg.register("null", "an idea that changes nothing must measure as nothing",
                     lambda: Idea())
    s = reg.run_search(h, sub, n_boot=50)
    assert s["firing_size"] == 0 and s["wild_p"] == 1.0 and s["dwer"] == 0.0
    assert s["screen"]["pass"] is False and s["duplicate_of"] is None


# --------------------------------------------------------------- confirmation
def test_confirmation_is_a_one_way_door_across_a_restart(reg, tmp_path):
    sub = search_sub()
    h = reg.register("break", "corrupting one token per window must raise WER",
                     BreakFirstToken)
    reg.run_search(h, sub, n_boot=50)
    bid = reg.freeze_confirmation_batch([h], note="smoke")
    reg.run_confirmation(bid, [h], sub, confirm_sub(), n_boot=50)

    fresh = A.Registry(A.Journal(reg.journal.path), partitions=TEST_PARTS)
    fresh._factories[h.idea_key] = BreakFirstToken
    with pytest.raises(ValueError, match="already frozen"):
        fresh.freeze_confirmation_batch([h])
    with pytest.raises(ValueError, match="already spent"):
        fresh.run_confirmation(bid, [h], sub, confirm_sub(), n_boot=50)


def test_confirmation_outside_a_frozen_batch_is_refused(reg):
    sub = search_sub()
    h = reg.register("break", "corrupting one token per window must raise WER",
                     BreakFirstToken)
    reg.run_search(h, sub, n_boot=50)
    with pytest.raises(ValueError, match="no frozen batch"):
        reg.run_confirmation("deadbeef", [h], sub, confirm_sub(), n_boot=50)


def test_batch_cannot_be_extended_or_exceed_budget(reg, monkeypatch):
    sub = search_sub()
    monkeypatch.setattr(A, "CONFIRM_BUDGET", 1)
    ideas = [("break", BreakFirstToken), ("break2", BreakSecondToken)]
    hs = []
    for name, cls in ideas:
        h = reg.register(name, f"{name} changes tokens and must raise WER", cls)
        reg.run_search(h, sub, n_boot=50)
        hs.append(h)
    with pytest.raises(ValueError, match="budget"):
        reg.freeze_confirmation_batch(hs)
    reg.freeze_confirmation_batch([hs[0]])
    # A SECOND batch is refused outright: five sequential singleton batches, each
    # Holm-corrected inside itself, would give a familywise error of 22.6%.
    with pytest.raises(ValueError, match="already frozen"):
        reg.freeze_confirmation_batch([hs[1]])


def test_unsearched_idea_cannot_be_confirmed(reg):
    h = reg.register("break", "corrupting one token per window must raise WER",
                     BreakFirstToken)
    with pytest.raises(ValueError, match="never evaluated"):
        reg.freeze_confirmation_batch([h])


def test_leaderboard_always_carries_its_denominator(reg):
    sub = search_sub()
    for name, cls in (("break", BreakFirstToken), ("break2", BreakSecondToken)):
        h = reg.register(name, f"{name} changes tokens and must raise WER", cls)
        reg.run_search(h, sub, n_boot=50)
    d = reg.leaderboard()["denominator"]
    assert d["registered"] == 2 and d["searched"] == 2
    assert d["confirmations_remaining"] == A.CONFIRM_BUDGET


# ------------------------------------------------------------- the multiplicity
def _noise_trial(rng, n_ideas, n_clusters, r):
    """One simulated cycle: n_ideas pure-noise arms, best few 'confirmed'."""
    n = rng.integers(80, 400, n_clusters).astype(float)
    weights = A.rademacher(n_clusters, r=r, seed=int(rng.integers(1, 10 ** 8)))
    ps = {}
    for i in range(n_ideas):
        d = rng.normal(0.0, 1.0, n_clusters) * np.sqrt(n)      # exactly null
        ps[f"i{i}"] = A.wild_cluster_test(d, n, 0.0, weights=weights)["p"]
    return ps


@pytest.mark.parametrize("n_ideas", [5])
def test_multiplicity_control_actually_bites(n_ideas):
    """With pure-noise ideas, 'any p<0.05' is common and 'any Holm ship' is not."""
    rng = np.random.default_rng(3)
    trials, r = 200, 299
    any_raw = any_holm = 0
    for _ in range(trials):
        ps = _noise_trial(rng, n_ideas, 40, r)
        if min(ps.values()) < 0.05:
            any_raw += 1
        if any(A.holm(ps)["reject"].values()):
            any_holm += 1
    raw, shipped = any_raw / trials, any_holm / trials
    assert raw > shipped, "the correction must remove something"
    assert shipped <= 0.12, f"familywise ship rate {shipped:.3f} is not controlled"


# ------------------------------------------------------- enforced partitions
def test_search_refuses_a_substrate_holding_confirmation_cities(reg):
    h = reg.register("break", "corrupting one token per window must raise WER",
                     BreakFirstToken)
    with pytest.raises(ValueError, match="confirmation cities"):
        reg.run_search(h, synth(n_cities=4), n_boot=50)


def test_confirmation_refuses_anything_but_the_whole_confirm_partition(reg):
    sub = search_sub()
    h = reg.register("break", "corrupting one token per window must raise WER",
                     BreakFirstToken)
    reg.run_search(h, sub, n_boot=50)
    bid = reg.freeze_confirmation_batch([h])
    with pytest.raises(ValueError, match="not the frozen"):
        reg.run_confirmation(bid, [h], sub, sub, n_boot=50)     # same data twice
    part = A._restrict(synth(n_cities=4), ("city2",))           # hand-picked subset
    with pytest.raises(ValueError, match="not the frozen"):
        reg.run_confirmation(bid, [h], sub, part, n_boot=50)


def test_partitions_must_be_disjoint():
    with pytest.raises(ValueError, match="overlap"):
        A.Partitions(search=("a", "b"), confirm=("b",))


# ------------------------------------------------------------ fail-closed dedup
def test_dedup_fails_closed_when_the_firing_cache_is_gone(reg, tmp_path):
    sub = search_sub()
    h1 = reg.register("break", "corrupting one token per window must raise WER",
                      BreakFirstToken)
    reg.run_search(h1, sub, n_boot=50)
    A._firing_path(h1.idea_key).unlink()
    h2 = reg.register("break2", "corrupting a different token is a different idea",
                      BreakSecondToken)
    with pytest.raises(RuntimeError, match="missing from"):
        reg.run_search(h2, sub, n_boot=50)


# -------------------------------------------------- the minimum-effect ship test
def test_ship_test_is_one_sided_against_the_effect_floor():
    n = np.full(60, 500.0)
    # a real but far-too-small effect: the zero-null test sees it, the floor does not
    d = np.full(60, -0.05)
    zero = A.wild_cluster_test(d, n, 0.0, r=999)
    floor = A.wild_cluster_test(d, n, -A.MIN_EFFECT, r=999, alternative="less")
    assert zero["p"] < 0.05, "a consistent tiny effect is significant against zero"
    assert floor["p"] > 0.05, "but must not reject 'smaller than useful'"


def test_a_uniform_nonzero_effect_is_not_reported_as_a_null_arm():
    n = np.full(30, 100.0)
    d = -0.01 * n                      # zero variance, clearly non-null
    r = A.wild_cluster_test(d, n, 0.0, r=199)
    assert r["degenerate"] and r["delta"] == pytest.approx(-0.01)
    assert r["p"] < 0.05, "se==0 with delta!=delta0 must not return p=1"


def test_one_sided_test_ignores_effects_in_the_wrong_direction():
    n = np.full(40, 200.0)
    rng = np.random.default_rng(5)
    d = 0.02 * n + rng.normal(0, 1, 40)       # a large effect the WRONG way
    r = A.wild_cluster_test(d, n, -A.MIN_EFFECT, r=999, alternative="less")
    assert r["p"] > 0.5
