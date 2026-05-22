#!/usr/bin/env python3
import json
import sys

start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 400
end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 449

with open('/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-440.review-chunk3.json') as f:
    data = json.load(f)

for item in data:
    idx = item['index']
    if start_idx <= idx <= end_idx:
        print(f"[{idx}] {item['utterance_id']}")
        print(f"  B: {item['before']}")
        print(f"  A: {item['after']}")
        print(f"  C: {item['categories']}")
        print("-" * 60)
