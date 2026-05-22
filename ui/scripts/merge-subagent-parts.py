#!/usr/bin/env python3
import sys
import os
import json

VALID_CATEGORIES = {
    "homophone", "accent_tonos", "final_sigma", "word_boundary", "substitution_phonetic",
    "insertion", "deletion", "verb_inflection", "noun_case", "article_pronoun", "person_name",
    "place_name", "org_party_name", "acronym_abbreviation", "legal_admin_term", "number_date",
    "punctuation_capitalization", "disfluency_cleanup", "semantic_rewrite",
    "timestamp_misalignment", "unusable"
}

def main():
    if len(sys.argv) < 4:
        print("Usage: merge-subagent-parts.py <input_batch.json> <output_target.json> <part1.json> <part2.json> ...", file=sys.stderr)
        sys.exit(1)
        
    input_batch_path = sys.argv[1]
    output_target_path = sys.argv[2]
    part_paths = sys.argv[3:]
    
    # Load input batch to know expected IDs
    try:
        with open(input_batch_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        expected_ids = [item['utterance_id'] for item in input_data]
    except Exception as e:
        print(f"Error loading input batch {input_batch_path}: {e}", file=sys.stderr)
        sys.exit(1)
        
    merged = []
    seen_ids = set()
    
    for p in part_paths:
        if not os.path.exists(p):
            print(f"Error: Part file {p} does not exist", file=sys.stderr)
            sys.exit(1)
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Part file {p} is not a JSON list", file=sys.stderr)
                sys.exit(1)
            for item in data:
                if not isinstance(item, dict) or 'utterance_id' not in item or 'categories' not in item:
                    print(f"Warning: Invalid item format in {p}: {item}", file=sys.stderr)
                    continue
                uid = item['utterance_id']
                if uid in seen_ids:
                    print(f"Warning: Duplicate utterance_id {uid} found", file=sys.stderr)
                    continue
                # Validate categories
                cats = item['categories']
                if not isinstance(cats, list):
                    cats = []
                cleaned_cats = [c for c in cats if c in VALID_CATEGORIES]
                merged.append({
                    "utterance_id": uid,
                    "categories": cleaned_cats
                })
                seen_ids.add(uid)
        except Exception as e:
            print(f"Error parsing part file {p}: {e}", file=sys.stderr)
            sys.exit(1)
            
    # Verify we got everything in the correct order or at least all expected IDs are present
    missing = [uid for uid in expected_ids if uid not in seen_ids]
    if missing:
        print(f"Warning: {len(missing)} utterance_ids from input batch are missing in the merged judgment", file=sys.stderr)
        # Pad missing ones with empty categories so we preserve the structure
        for uid in missing:
            merged.append({
                "utterance_id": uid,
                "categories": []
            })
            
    # Re-order the merged judgments to match the exact order of the input batch for maximum cleanliness
    id_to_judgment = {item['utterance_id']: item for item in merged}
    final_merged = [id_to_judgment[uid] for uid in expected_ids if uid in id_to_judgment]
    
    # Double check length matches expected
    if len(final_merged) != len(expected_ids):
        print(f"Error: Final merged count ({len(final_merged)}) does not match expected input count ({len(expected_ids)})", file=sys.stderr)
        sys.exit(1)
        
    try:
        os.makedirs(os.path.dirname(output_target_path), exist_ok=True)
        with open(output_target_path, 'w', encoding='utf-8') as f:
            json.dump(final_merged, f, indent=2, ensure_ascii=False)
        print(f"Successfully merged and validated {len(final_merged)} items into {output_target_path}")
    except Exception as e:
        print(f"Error writing output {output_target_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
