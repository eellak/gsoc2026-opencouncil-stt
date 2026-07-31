"""The A/B harness must actually reproduce the bug it claims to measure.

A legacy arm that quietly behaves like the fixed one would report "the bug cost
nothing" — the most expensive possible way to be wrong here. These tests pin the
two arms apart on CPU, before any GPU time is spent.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "ab_label_bug"))
sys.path.insert(0, str(ROOT / "notebooks"))

import run_ab                    # noqa: E402
import train_runpod as tr        # noqa: E402

MODEL_ID = "openai/whisper-large-v3"


@pytest.fixture(scope="module")
def processor():
    from transformers import WhisperProcessor
    return WhisperProcessor.from_pretrained(MODEL_ID, language="greek", task="transcribe")


def _feats(processor, texts):
    """Minimal collator input: real label ids, dummy features."""
    n_mel = processor.feature_extractor.feature_size
    return [{"input_features": np.zeros((n_mel, 3000), dtype=np.float32),
             "labels": processor.tokenizer(t).input_ids} for t in texts]


def test_legacy_arm_reproduces_the_bug(processor):
    """The legacy collator must leave <|startoftranscript|> in the labels."""
    sot = 50258
    out = run_ab.LegacyCollator(processor)(_feats(processor, ["Καλημέρα σας", "Ευχαριστώ"]))
    assert out["labels"][:, 0].tolist() == [sot, sot]


def test_fixed_arm_strips_the_prefix(processor):
    sot = 50258
    out = tr.Collator(processor, sot)(_feats(processor, ["Καλημέρα σας", "Ευχαριστώ"]))
    assert sot not in out["labels"][:, 0].tolist()
    # what remains is the language token, i.e. the canonical target start
    assert out["labels"][0, 0].item() == processor.tokenizer.convert_tokens_to_ids("<|el|>")


def test_arms_differ_by_exactly_one_leading_token(processor):
    """The two arms must differ in the label prefix and nothing else."""
    feats = _feats(processor, ["Καλημέρα σας", "Ευχαριστώ πολύ για τον χρόνο σας"])
    legacy = run_ab.LegacyCollator(processor)(feats)["labels"]
    fixed = tr.Collator(processor, 50258)(feats)["labels"]
    assert legacy.shape[1] == fixed.shape[1] + 1
    assert legacy[:, 1:].tolist() == fixed.tolist()


def test_bos_is_not_decoder_start(processor):
    """The trap itself: the id the old code compared against is the wrong one."""
    assert processor.tokenizer.bos_token_id == 50257
    assert processor.tokenizer("Καλημέρα").input_ids[0] == 50258


def test_wer_counting_matches_hand_computation():
    counts = run_ab.per_utt_counts(["το δημοτικό συμβούλιο", "καλημέρα"],
                                   ["το δημοτικο συμβουλιο", "καλησπέρα"])
    # accents are normalized away -> first utterance is exact; second is 1 sub of 1
    assert counts[0][0] == 0
    assert counts[1][0] == 1
    assert run_ab.agg_wer(counts) == pytest.approx(1 / 4)


def test_cluster_bootstrap_flags_a_real_gap_and_ignores_a_null_one():
    worse = [(2, 10, 0, 0)] * 40      # 20% WER
    better = [(1, 10, 0, 0)] * 40     # 10% WER
    mtgs = [f"city/m{i // 4}" for i in range(40)]   # 10 meetings, 4 utterances each
    res = run_ab.cluster_bootstrap(worse, better, mtgs, 500)
    assert res["delta_wer"] == pytest.approx(0.1)
    assert res["excludes_zero"]
    assert res["n_clusters"] == 10
    assert res["meetings_a_worse"] == 10

    same = run_ab.cluster_bootstrap(better, list(better), mtgs, 500)
    assert same["delta_wer"] == pytest.approx(0.0)
    assert not same["excludes_zero"]


def test_cluster_bootstrap_is_wider_than_ignoring_clustering():
    """A meeting-level interval must not be narrower than an utterance-level one.

    The whole reason for clustering: 20 correlated utterances from one meeting are
    not 20 independent observations, and pretending otherwise reports confidence
    the data does not have.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    a, b, mtgs = [], [], []
    for m in range(6):                       # 6 meetings, whole-meeting offsets
        shift = rng.normal(0, 1.5)
        for _ in range(25):
            err_b = max(0.0, rng.normal(2, 0.5))
            a.append((err_b + max(0.0, shift), 10, 0, 0))
            b.append((err_b, 10, 0, 0))
            mtgs.append(f"c/m{m}")
    clustered = run_ab.cluster_bootstrap(a, b, mtgs, 2000)
    unclustered = run_ab.cluster_bootstrap(a, b, [f"c/u{i}" for i in range(len(a))], 2000)
    width = lambda r: r["ci95_cluster"][1] - r["ci95_cluster"][0]
    assert width(clustered) > width(unclustered)


def test_cluster_bootstrap_rejects_mismatched_references():
    """Different reference lengths per arm mean the pairing is broken, not a result."""
    a = [(1, 10, 0, 0)] * 5
    b = [(1, 9, 0, 0)] * 5
    with pytest.raises(AssertionError):
        run_ab.cluster_bootstrap(a, b, ["c/m0"] * 5, 100)
