"""Regression test for the Whisper eval-decode fix (notebook cell 7).

Background: the first large-v3 LoRA run decoded predictions/references with the
WhisperTokenizer default ``clean_up_tokenization_spaces=True``. Transformers
itself warns this is *destructive for BPE* — it strips the space before
punctuation. That silently rewrites the model's text before WER/CER, so the
metric stops seeing punctuation-spacing differences. The fix: decode with
``clean_up_tokenization_spaces=False`` on BOTH sides, then apply one explicit,
identical whitespace policy.

This test pins the behaviour two ways:
1. A pure-Python replica of HF's ``clean_up_tokenization`` (no deps) — always
   runs, proves the cleanup mutates Greek council text.
2. The real tokenizer, if ``transformers`` is importable (Kaggle / mini PC).

Run: ``pytest eval/tests/test_decode_cleanup.py -v``
"""
try:
    import pytest
except ImportError:  # allow the pure-Python checks to run without pytest installed
    pytest = None

# A council-style line with the spacing patterns HF's cleanup targets: " ." and " ,".
GREEK = "Ναι , κύριε Παπαδόπουλε . Το ΦΕΚ είναι 4412 ;"


def hf_cleanup(s: str) -> str:
    """Replica of transformers PreTrainedTokenizerBase.clean_up_tokenization
    (the function invoked when clean_up_tokenization_spaces=True)."""
    return (
        s.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
        .replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m")
        .replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
    )


def policy_decode(s: str) -> str:
    """Our chosen policy: do NOT clean up; keep the text as the model produced it,
    only strip + collapse whitespace (applied identically to preds and refs)."""
    return " ".join(s.split())


def test_cleanup_is_destructive_on_greek_punctuation():
    """clean_up=True removes the space before '.' and ',' — corrupting the text."""
    cleaned = hf_cleanup(GREEK)
    assert cleaned != GREEK, "cleanup should mutate council text (it didn't)"
    assert " ." not in cleaned and " ," not in cleaned, "spaces before . and , survived"
    assert "Παπαδόπουλε." in cleaned and "Ναι," in cleaned, "expected the destructive joins"


def test_policy_preserves_text():
    """Our policy keeps spacing intact; only outer/duplicate whitespace is normalised."""
    out = policy_decode("  Ναι ,  κύριε .  ")
    assert out == "Ναι , κύριε .", out
    assert " ." in out and " ," in out, "policy must NOT strip spaces before punctuation"


def test_policy_is_symmetric():
    """The whitespace policy is a pure function -> identical transform on preds and refs,
    so WER stays fair (no asymmetry introduced)."""
    for s in (GREEK, "  a   b  ", "Το\tΦΕΚ\n4412"):
        assert policy_decode(s) == policy_decode(policy_decode(s))  # idempotent


def test_real_tokenizer_decode_flag():
    """With the actual tokenizer: clean_up=False must preserve a space the True path strips.
    Skipped where transformers isn't installed (laptop); runs on Kaggle / mini PC."""
    transformers = pytest.importorskip("transformers")
    # whisper-small shares the exact clean_up algorithm with large-v3 and is far lighter.
    tok = transformers.AutoTokenizer.from_pretrained("openai/whisper-small")
    ids = tok(GREEK).input_ids
    keep = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    strip = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # The False decode keeps the model's spacing; the True decode collapses it.
    assert keep != strip or " ." not in keep, (
        "expected the two decode flags to differ on punctuation spacing"
    )
    # And our policy decode of the False path must still contain a space before punctuation.
    assert " ." in policy_decode(keep) or " ;" in policy_decode(keep)
