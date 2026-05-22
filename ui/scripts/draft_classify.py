import json
import re
import unicodedata

# Category IDs
# homophone, accent_tonos, final_sigma, word_boundary, substitution_phonetic, insertion, deletion, verb_inflection, noun_case, article_pronoun, person_name, place_name, org_party_name, acronym_abbreviation, legal_admin_term, number_date, punctuation_capitalization, disfluency_cleanup, semantic_rewrite, timestamp_misalignment, unusable

VALID_CATEGORIES = {
    "homophone", "accent_tonos", "final_sigma", "word_boundary", "substitution_phonetic",
    "insertion", "deletion", "verb_inflection", "noun_case", "article_pronoun", "person_name",
    "place_name", "org_party_name", "acronym_abbreviation", "legal_admin_term", "number_date",
    "punctuation_capitalization", "disfluency_cleanup", "semantic_rewrite",
    "timestamp_misalignment", "unusable"
}

def clean_text(s):
    return unicodedata.normalize('NFC', s).strip()

def has_accents_diff(a, b):
    # Strip tones/accents
    def strip_tonos(s):
        # decompose and remove combining acute/tonos (U+0301)
        decomposed = unicodedata.normalize('NFD', s)
        filtered = "".join([c for c in decomposed if ord(c) != 0x0301])
        return unicodedata.normalize('NFC', filtered)
    
    a_clean = strip_tonos(a.lower())
    b_clean = strip_tonos(b.lower())
    
    # If ignoring accents they are equal, then there is an accent diff
    # But wait, sigma differences should be ignored or separated.
    # Let's normalize sigma first:
    def norm_sigma(s):
        return s.replace('ς', 'σ')
    
    a_sig = norm_sigma(a_clean)
    b_sig = norm_sigma(b_clean)
    
    return a_sig == b_sig and norm_sigma(a.lower()) != norm_sigma(b.lower())

def has_sigma_diff(a, b):
    def norm_sigma(s):
        return s.replace('ς', 'σ')
    def strip_tonos(s):
        decomposed = unicodedata.normalize('NFD', s)
        filtered = "".join([c for c in decomposed if ord(c) != 0x0301])
        return unicodedata.normalize('NFC', filtered)
        
    a_clean = strip_tonos(a.lower())
    b_clean = strip_tonos(b.lower())
    
    return norm_sigma(a_clean) == norm_sigma(b_clean) and a_clean != b_clean

def is_word_boundary_diff(a, b):
    return a.replace(" ", "") == b.replace(" ", "") and a != b

def main():
    input_path = '/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-438.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} items.")
    
    # Just print the first 20 items to inspect
    for i in range(20):
        item = data[i]
        print(f"{i}: {item['utterance_id']}")
        print(f"  Before: {item['before']}")
        print(f"  After:  {item['after']}")
        
if __name__ == '__main__':
    main()
