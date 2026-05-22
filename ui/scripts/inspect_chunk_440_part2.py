import json
import difflib

def get_diff(before, after):
    s = difflib.SequenceMatcher(None, before, after)
    result = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            result.append(before[i1:i2])
        elif tag == 'replace':
            result.append(f"[-{before[i1:i2]}-]{{+{after[j1:j2]}+}}")
        elif tag == 'delete':
            result.append(f"[-{before[i1:i2]}-]")
        elif tag == 'insert':
            result.append(f"{{+{after[j1:j2]}+}}")
    return "".join(result)

def main():
    batch_path = '/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-440.json'
    with open(batch_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    out_path = '/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-440.diffs-part2.txt'
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for idx in range(200, 400):
            if idx >= len(data):
                break
            item = data[idx]
            before = item['before']
            after = item['after']
            diff = get_diff(before, after)
            f_out.write(f"Index: {idx} | ID: {item['utterance_id']}\n")
            f_out.write(f"  Before: {before}\n")
            f_out.write(f"  After:  {after}\n")
            f_out.write(f"  Diff:   {diff}\n")
            f_out.write("-" * 80 + "\n")
            
    print(f"Wrote diffs to {out_path}")

if __name__ == '__main__':
    main()
