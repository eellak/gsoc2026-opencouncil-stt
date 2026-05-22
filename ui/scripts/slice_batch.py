import json
import sys

def main():
    batch_file = "/Users/harold/projects/opencouncil-fine-tuning/ui/.state/llm-judgments/batch-439.json"
    with open(batch_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunk_num = int(sys.argv[1])
    start = (chunk_num - 1) * 200
    end = chunk_num * 200
    chunk = data[start:end]
    
    print(json.dumps(chunk, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
