# Load JSON file and extract multiples
import json
import sys

def count_hypernyms(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        counts = {}
        
        for key, hypernyms in data.items():
            print(f"Processing item with ID: {key}", file=sys.stderr)

            l = len(hypernyms)
            if l not in counts:
                counts[l] = 0
            counts[l] += 1

        return counts
    
    except Exception as e:
        print(f"Error reading JSON file: {e}", file=sys.stderr)
        sys.exit(1)

# run transform_data if this script is executed directly
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transform_data.py <json_file>", file=sys.stderr)
        sys.exit(1)
    
    json_file = sys.argv[1]
    count = count_hypernyms(json_file)

    # Dump the result on stdout
    json.dump(count, sys.stdout, indent=4)

    print("Counting completed successfully.", file=sys.stderr)
else:
    print("This script is intended to be run directly, not imported as a module.", file=sys.stderr)
