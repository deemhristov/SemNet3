# Load results a and b from json files from args
import argparse
import json
import os

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

    for key in keys:
        value_a = data_a.get(key, {})
        value_b = data_b.get(key, {})
        
        if value_a != value_b:
            print(f"Difference found for key '{key}':")
            print(f"  File A: {value_a}")
            print(f"  File B: {value_b}")
        else:
            print(f"No difference for key '{key}': {value_a}")

# Run the main function
if __name__ == "__main__":
    main()