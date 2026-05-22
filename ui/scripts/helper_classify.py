import json
import unicodedata
import sys

VALID_CATEGORIES = {
    "homophone", "accent_tonos", "final_sigma", "word_boundary", "substitution_phonetic",
    "insertion", "deletion", "verb_inflection", "noun_case", "article_pronoun", "person_name",
    "place_name", "org_party_name", "acronym_abbreviation", "legal_admin_term", "number_date",
    "punctuation_capitalization", "disfluency_cleanup", "semantic_rewrite",
    "timestamp_misalignment", "unusable"
}

def clean_text(s):
    return unicodedata.normalize('NFC', s).strip()

def strip_tonos(s):
    decomposed = unicodedata.normalize('NFD', s)
    filtered = "".join([c for c in decomposed if ord(c) != 0x0301])
    return unicodedata.normalize('NFC', filtered)

def norm_sigma(s):
    return s.replace('ς', 'σ')

def core(s):
    return strip_tonos(norm_sigma(s.lower()))

def is_punctuation_capitalization(a, b):
    # Strip everything except letters, digits, and spaces
    def strip_non_space_punct(s):
        out = []
        for c in s:
            if c.isalnum() or c.isspace():
                out.append(c)
        return " ".join("".join(out).split())
    
    return strip_non_space_punct(a).lower() == strip_non_space_punct(b).lower()

def is_accent_tonos(a, b):
    if core(a) != core(b):
        return False
    return norm_sigma(a.lower()) != norm_sigma(b.lower())

def is_final_sigma(a, b):
    if core(a) != core(b):
        return False
    return strip_tonos(a.lower()) != strip_tonos(b.lower())

def is_word_boundary(a, b):
    return a.replace(" ", "") == b.replace(" ", "") and a != b

# Homophones
SIMPLE_VOWELS = {'η', 'ι', 'υ', 'ω', 'ο', 'ε', 'α'}
SIMPLE_VOWEL_CLASS = {
    'η': 'I', 'ι': 'I', 'υ': 'I',
    'ω': 'O', 'ο': 'O',
    'ε': 'E', 'α': 'A'
}

def homophone_skeleton(token):
    base = norm_sigma(strip_tonos(token.lower()))
    out = []
    i = 0
    while i < len(base):
        pair = base[i:i+2]
        if pair in ('ει', 'οι'):
            out.append('I')
            i += 2
            continue
        if pair == 'αι':
            out.append('E')
            i += 2
            continue
        if pair == 'ου':
            out.append('U')
            i += 2
            continue
        ch = base[i]
        out.append(SIMPLE_VOWEL_CLASS.get(ch, ch))
        i += 1
    return "".join(out)

def is_simple_vowel_swap(x, y):
    I_set = {'η', 'ι', 'υ'}
    O_set = {'ω', 'ο'}
    return (x in I_set and y in I_set) or (x in O_set and y in O_set)

def is_homophone(a, b):
    ta = a.split()
    tb = b.split()
    if len(ta) != len(tb) or len(ta) == 0:
        return False
    any_diff = False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if core(x) == core(y):
            continue  # accent / case / sigma only
        stripX = strip_tonos(x).lower()
        stripY = strip_tonos(y).lower()
        if len(stripX) != len(stripY):
            return False
        if homophone_skeleton(x) != homophone_skeleton(y):
            return False
        for c1, c2 in zip(stripX, stripY):
            if c1 == c2:
                continue
            if not is_simple_vowel_swap(c1, c2):
                return False
        any_diff = True
    return any_diff

def is_disfluency(a, b):
    ta = a.split()
    tb = b.split()
    if len(ta) == 0 or len(tb) == 0 or len(tb) >= len(ta):
        return False
    
    def norm(t):
        out = []
        for c in strip_tonos(t.lower()):
            if c.isalnum():
                out.append(c)
        return "".join(out)
    
    na = [norm(t) for t in ta]
    nb = [norm(t) for t in tb]
    
    fillers = {'ε', 'εε', 'εεε', 'εμ', 'εμμ', 'αα', 'μμ', 'χμ'}
    
    j = 0
    removed = []
    for i in range(len(na)):
        if j < len(nb) and na[i] == nb[j]:
            j += 1
        else:
            prev = na[i-1] if i > 0 else None
            removed.append((na[i], prev))
            
    if j != len(nb) or not removed:
        return False
        
    for tok, prev in removed:
        is_filler = tok in fillers
        is_dup = prev is not None and prev == tok
        if not is_filler and not is_dup:
            return False
    return True

def auto_classify(before, after):
    a = clean_text(before)
    b = clean_text(after)
    if not a or not b or a == b:
        return []
    
    cats = []
    if is_accent_tonos(a, b):
        cats.append("accent_tonos")
    if is_final_sigma(a, b):
        cats.append("final_sigma")
    if is_word_boundary(a, b):
        cats.append("word_boundary")
    if is_homophone(a, b):
        cats.append("homophone")
    if is_disfluency(a, b):
        cats.append("disfluency_cleanup")
    if is_punctuation_capitalization(a, b):
        # Only add punctuation_capitalization if there isn't other core textual differences
        # But actually, rulePunctuationCapitalization in Svelte doesn't check core(a) == core(b), 
        # it just compares text with only alphanumeric characters.
        # Let's check: if it is word boundary, final_sigma, or accent_tonos or homophone, 
        # we still co-fire or do not co-fire?
        # Let's add it if no other categories are added, or if it is purely punctuation/capitalization.
        if not cats:
            cats.append("punctuation_capitalization")
            
    return sorted(list(set(cats)))

def main():
    batch_path = "/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-440.json"
    with open(batch_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    chunk = data[0:200]
    print(f"Loaded {len(data)} items, processing first {len(chunk)} items...")
    
    unclassified_count = 0
    for idx, item in enumerate(chunk):
        before = item["before"]
        after = item["after"]
        cats = auto_classify(before, after)
        
        if not cats:
            unclassified_count += 1
            print(f"Index: {idx} | ID: {item['utterance_id']} | UNCLASSIFIED")
            print(f"  Before: {before}")
            print(f"  After:  {after}")
            print("-" * 60)
        else:
            # print(f"Index: {idx} | ID: {item['utterance_id']} | Auto: {cats}")
            pass
            
    print(f"Total unclassified: {unclassified_count} / {len(chunk)}")

if __name__ == '__main__':
    main()
