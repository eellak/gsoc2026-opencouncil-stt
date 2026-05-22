import json

def main():
    batch_path = "/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-440.json"
    out_path = "/Users/harold/.gemini/antigravity/scratch/batch-440-chunk1-raw.json"
    
    with open(batch_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    chunk = data[0:200]
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully sliced {len(chunk)} items to {out_path}")

if __name__ == "__main__":
    main()
