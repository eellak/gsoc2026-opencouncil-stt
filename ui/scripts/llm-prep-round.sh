#!/usr/bin/env bash
# Prepare N batches of 50 unlabeled utterances per round, with correct --skip
# offsets so they don't overlap. Use before kicking off a parallel-subagent round.
#
# Usage:
#   ./scripts/llm-prep-round.sh <start-num> <count>
# Example:
#   ./scripts/llm-prep-round.sh 17 12   # prepares batch-017 through batch-028

set -euo pipefail
START="${1:?start batch number, e.g. 17}"
COUNT="${2:?how many batches to prepare}"

cd "$(dirname "$0")/.."

mkdir -p .state/llm-judgments

for ((i=0; i<COUNT; i++)); do
  num=$((START + i))
  padded=$(printf "%03d" "$num")
  skip=$((i * 50))
  bun scripts/llm-batch-prep.ts 50 --skip "$skip" --out ".state/llm-judgments/batch-${padded}.json" 2>&1 | tail -1
done

echo "---"
echo "Prepared batches batch-$(printf '%03d' "$START") through batch-$(printf '%03d' "$((START+COUNT-1))")"
echo "Each contains 50 unlabeled utterances. Now spawn 2 subagents per batch (judge-1, judge-2)."
echo "After they write outputs, merge with:"
echo "  for b in \$(seq -f '%03g' $START $((START+COUNT-1))); do"
echo "    bun scripts/llm-merge-batch.ts .state/llm-judgments/batch-\$b.judge-1.json .state/llm-judgments/batch-\$b.judge-2.json --apply"
echo "  done"
