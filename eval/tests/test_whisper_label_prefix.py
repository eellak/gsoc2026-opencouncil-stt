"""Regression test for the Whisper label-prefix bug (fixed 2026-07-31).

Background: every GPU fine-tune from the first Kaggle run through the published
`opencouncil/whisper-large-v3-el-council-lora` adapter used a collator that
stripped the label prefix only when

    labels[:, 0] == tokenizer.bos_token_id

For whisper-large-v3 that is 50257 (`<|endoftext|>`), while
`tokenizer(text).input_ids` starts with 50258 (`<|startoftranscript|>`, i.e.
`model.config.decoder_start_token_id`). The condition was therefore *always
False*, the strip never ran, and `shift_tokens_right` prepended a second
`<|startoftranscript|>`:

    labels:            SOT  <|el|> <|transcribe|> <|notimestamps|>  t1 t2 ...
    decoder_input_ids: SOT  SOT    <|el|>         <|transcribe|>    ... t1 ...

So the model was trained to emit SOT as its first token, and every content token
was trained one learned decoder position later than it appears at inference.

Two layers of protection, because they fail differently:

1. `test_no_bos_strip_in_training_code` — pure source scan, no deps, always
   runs. Catches the bug being reintroduced by copy-paste into *any* of the four
   training entry points, including the two notebooks a unit test cannot import.
2. `test_collator_*` — imports the REAL `Collator` from `notebooks/train_runpod.py`
   and asserts the actual tensors. Skipped where torch/transformers are absent
   (dev laptop); runs on the pod and the mini PC.

Run: ``pytest eval/tests/test_whisper_label_prefix.py -v``
"""
import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TRAIN_RUNPOD = REPO / "notebooks" / "train_runpod.py"
TRAINING_FILES = [
    TRAIN_RUNPOD,
    REPO / "notebooks" / "train_smoke.py",
    REPO / "notebooks" / "whisper_finetune_kaggle.ipynb",
    REPO / "notebooks" / "whisper_sweep_kaggle.ipynb",
]

# `labels[:, 0] == <something>.bos_token_id` — the exact shape of the bug.
BUGGY = re.compile(r"labels\[:,\s*0\]\s*==\s*[\w.]*\bbos_token_id\b")

GREEK = "Καλημέρα σας, ξεκινάει η συνεδρίαση του δημοτικού συμβουλίου."
GREEK2 = "Ευχαριστώ."


# ---------------------------------------------------------------- source guard

@pytest.mark.parametrize("path", TRAINING_FILES, ids=lambda p: p.name)
def test_no_bos_strip_in_training_code(path):
    """No training entry point may gate the prefix strip on bos_token_id."""
    assert path.exists(), f"{path} missing — update TRAINING_FILES"
    hit = BUGGY.search(path.read_text())
    assert hit is None, (
        f"{path.name} compares labels[:, 0] against bos_token_id: {hit.group(0)!r}. "
        "For Whisper that is <|endoftext|> (50257), not <|startoftranscript|> "
        "(50258 = decoder_start_token_id), so the strip silently never fires."
    )


# ------------------------------------------------------------- real collator

def _load_train_runpod():
    """Import notebooks/train_runpod.py as a module (stdlib-only at import time)."""
    spec = importlib.util.spec_from_file_location("train_runpod", TRAIN_RUNPOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def real():
    torch = pytest.importorskip("torch", reason="collator asserts need torch")
    transformers = pytest.importorskip("transformers")
    import numpy as np
    from transformers import WhisperProcessor
    from transformers.models.whisper.modeling_whisper import shift_tokens_right

    proc = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3", language="greek", task="transcribe")
    # The id the model itself will prepend. Pinned to the value in the base
    # model's config; if HF ever changes it the test must fail, not adapt.
    decoder_start = 50258
    assert proc.tokenizer.convert_ids_to_tokens(decoder_start) == "<|startoftranscript|>"
    assert proc.tokenizer.bos_token_id != decoder_start, (
        "premise of this whole test: bos_token_id and decoder_start_token_id differ")

    mod = _load_train_runpod()
    collator = mod.Collator(proc, decoder_start)

    def batch(texts):
        feats = [{"input_features": np.zeros((128, 3000), dtype=np.float32),
                  "labels": proc.tokenizer(t).input_ids} for t in texts]
        return collator(feats)

    return {"proc": proc, "torch": torch, "np": np, "sot": decoder_start,
            "collator": collator, "batch": batch, "shift": shift_tokens_right,
            "transformers": transformers}


def test_raw_tokenization_starts_with_sot(real):
    ids = real["proc"].tokenizer(GREEK).input_ids
    assert ids[0] == real["sot"]


def test_collator_strips_exactly_one_prefix_token(real):
    ids = real["proc"].tokenizer(GREEK).input_ids
    out = real["batch"]([GREEK])
    assert out["labels"][0].tolist() == ids[1:], "labels must equal raw_ids[1:]"
    assert out["labels"][0][0].item() != real["sot"], "SOT still present in labels"


def test_padding_is_masked_to_minus_100(real):
    out = real["batch"]([GREEK, GREEK2])          # different lengths -> padding
    short = out["labels"][1]
    assert (short == -100).any().item(), "short sequence should carry -100 padding"
    assert short[0].item() != -100


def test_shifted_decoder_input_has_a_single_sot(real):
    """The real check: what the model actually consumes."""
    proc, sot = real["proc"], real["sot"]
    out = real["batch"]([GREEK])
    labels = out["labels"].masked_fill(out["labels"] == -100, proc.tokenizer.pad_token_id)
    dec_in = real["shift"](labels, proc.tokenizer.pad_token_id, sot)[0].tolist()
    assert dec_in[0] == sot
    assert dec_in[1] != sot, f"duplicated <|startoftranscript|>: {dec_in[:5]}"
    assert proc.tokenizer.convert_ids_to_tokens(dec_in[:4]) == [
        "<|startoftranscript|>", "<|el|>", "<|transcribe|>", "<|notimestamps|>"]


def test_missing_prefix_fails_loudly(real):
    """A batch whose labels lost the prefix must raise, not train silently."""
    np, proc = real["np"], real["proc"]
    feats = [{"input_features": np.zeros((128, 3000), dtype=np.float32),
              "labels": proc.tokenizer(GREEK).input_ids[1:]}]   # prefix already gone
    with pytest.raises(ValueError, match="label prefix invariant"):
        real["collator"](feats)


def test_mixed_batch_fails_loudly(real):
    """One bad sequence among good ones must still stop the run."""
    np, proc = real["np"], real["proc"]
    feats = [{"input_features": np.zeros((128, 3000), dtype=np.float32), "labels": ids}
             for ids in (proc.tokenizer(GREEK).input_ids,
                         proc.tokenizer(GREEK2).input_ids[1:])]
    with pytest.raises(ValueError, match="1/2"):
        real["collator"](feats)


def test_empty_transcript_is_not_a_false_failure(real):
    """Empty text still tokenizes to prefix + EOS, so it must pass, not crash."""
    out = real["batch"](["", GREEK])
    assert out["labels"].shape[0] == 2


def test_left_padding_is_rejected_not_miscollated(real):
    """Left padding would make labels[:, 1:] strip a pad token instead of the
    prefix. The collator must refuse rather than silently corrupt targets."""
    proc = real["proc"]
    original = proc.tokenizer.padding_side
    proc.tokenizer.padding_side = "left"
    try:
        with pytest.raises(ValueError, match="right padding"):
            real["batch"]([GREEK, GREEK2])
    finally:
        proc.tokenizer.padding_side = original


def test_sot_literal_matches_model_config(real):
    """The two Kaggle notebooks derive the id from the "<|startoftranscript|>"
    literal instead of model.config.decoder_start_token_id. That is only safe
    while the two agree — pin it here so a divergence fails loudly."""
    from transformers import WhisperConfig
    cfg = WhisperConfig.from_pretrained("openai/whisper-large-v3")
    literal = real["proc"].tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    assert cfg.decoder_start_token_id == literal == real["sot"]
