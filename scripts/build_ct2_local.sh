#!/usr/bin/env bash
# Merge a local LoRA adapter into whisper-large-v3 and convert to CTranslate2 int8,
# into an arbitrary output directory. Same recipe as serve/oc-asr/build_model.sh
# (PEFT merge_and_unload -> ct2-transformers-converter --quantization int8_float32),
# but parameterised so screen adapters can be built outside the serving tree.
#
#   ADAPTER=/path/to/adapter OUT=/home/harold/oc-run2-stage2 ./scripts/build_ct2_local.sh
#
# Writes $OUT/ct2 and $OUT/ct2.sha256, and deletes the ~6GB $OUT/merged afterwards.
set -euo pipefail

VENV="${OC_ASR_VENV:-/home/harold/opencouncil-fine-tuning/.venv-eval}"
BASE="${BASE:-openai/whisper-large-v3}"
ADAPTER="${ADAPTER:?set ADAPTER to the adapter directory}"
OUT="${OUT:?set OUT to the build directory}"

mkdir -p "$OUT"

"$VENV/bin/python" - "$BASE" "$ADAPTER" "$OUT/merged" <<'PY'
import sys, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
from peft import PeftModel
base_id, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
model.generation_config.language = "greek"; model.generation_config.task = "transcribe"
model.save_pretrained(out)
WhisperProcessor.from_pretrained(base_id, language="greek", task="transcribe").save_pretrained(out)
WhisperFeatureExtractor.from_pretrained(base_id).save_pretrained(out)
print("merged ->", out)
PY

"$VENV/bin/ct2-transformers-converter" \
  --model "$OUT/merged" \
  --output_dir "$OUT/ct2" \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization int8_float32
echo BUILD_DONE

sha256sum "$OUT"/ct2/* > "$OUT/ct2.sha256"
rm -rf "$OUT/merged"
echo MERGED_REMOVED
cat "$OUT/ct2.sha256"
