import json

def main():
    batch_path = "/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-438.json"
    with open(batch_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunk = data[600:800]
    print(f"Total items sliced: {len(chunk)}")
    for i, item in enumerate(chunk[:30]):
        print(f"Index {600+i}: before='{item['before']}' -> after='{item['after']}'")

if __name__ == "__main__":
    main()
