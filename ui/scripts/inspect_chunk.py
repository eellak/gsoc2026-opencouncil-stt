import json
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: inspect_chunk.py <batch.json> <start_idx> <end_idx>")
        sys.exit(1)
    
    batch_file = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])
    
    with open(batch_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    chunk = data[start_idx:end_idx+1]
    
    for idx, item in enumerate(chunk):
        real_idx = start_idx + idx
        print(f"[{real_idx}] ID: {item['utterance_id']}")
        print(f"  Before: {item['before']}")
        print(f"  After : {item['after']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
