#!/usr/bin/env bash
# Build the serving model: merge the LoRA adapter into whisper-large-v3, then
# convert to CTranslate2 int8. Rebuildable from the public adapter if lost.
#
#   ADAPTER=opencouncil/whisper-large-v3-el-council-lora ./build_model.sh
#
# Outputs ./merged (full HF model, ~6GB) and ./ct2 (CTranslate2 int8, ~1.5GB).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${OC_ASR_VENV:-/home/harold/opencouncil-fine-tuning/.venv-eval}"
ADAPTER="${ADAPTER:-opencouncil/whisper-large-v3-el-council-lora}"
BASE="${BASE:-openai/whisper-large-v3}"

"$VENV/bin/python" - "$BASE" "$ADAPTER" "$HERE/merged" <<'PY'
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
  --model "$HERE/merged" \
  --output_dir "$HERE/ct2" \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization int8_float32
echo "ct2 build ready -> $HERE/ct2"
