# Load results a and b from json files from args
import argparse
import json
import os
import sys

def load_json(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as file:
        return json.load(file)
    
def main():
    parser = argparse.ArgumentParser(description='Load and compare JSON files.')
    parser.add_argument('file_a', type=str, help='Path to the first JSON file')
    parser.add_argument('file_b', type=str, help='Path to the second JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_a):
        print(f"Error: The file {args.file_a} does not exist.")
        return
    
    if not os.path.exists(args.file_b):
        print(f"Error: The file {args.file_b} does not exist.")
        return
    
    data_a = load_json(args.file_a)
    data_b = load_json(args.file_b)

    keys = set(data_a.keys()).union(set(data_b.keys()))

    same = 0
    for key in keys:
        value_a = data_a.get(key, {})
        value_b = data_b.get(key, {})
        
        if value_a is not None and value_b is not None and value_a == value_b:
            same += 1

    print(f"File A: {args.file_a}", file=sys.stderr)
    print(f"File B: {args.file_b}", file=sys.stderr)
    print(f"{same} / {len(keys)} = {same / len(keys) * 100:.2f}%", file=sys.stderr)
    print(same / len(keys))

# Run the main function
if __name__ == "__main__":
    main()