# Load JSON file and extract multiples
import json
import sys

def transform_data(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        new_data = {}
        
        for key, hypernyms in data.items():
            print(f"Processing item with ID: {key}", file=sys.stderr)

            # if there is a hypernym with old = None, skip it
            if any(hypernym.get('old') is None for hypernym in hypernyms.values()):
                print(f"Skipping item {key} due to hypernym with old type None", file=sys.stderr)
                continue

            new_data[key] = {}
            for hypernym_id in hypernyms:
                new_data[key][hypernym_id] = hypernyms[hypernym_id].get("new")

        return new_data
    
    except Exception as e:
        print(f"Error reading JSON file: {e}", file=sys.stderr)
        sys.exit(1)

# run transform_data if this script is executed directly
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transform_data.py <json_file>", file=sys.stderr)
        sys.exit(1)
    
    json_file = sys.argv[1]
    transformed_data = transform_data(json_file)

    # Dump the result on stdout
    json.dump(transformed_data, sys.stdout, indent=4)

    print("Data transformation completed successfully.", file=sys.stderr)
else:
    print("This script is intended to be run directly, not imported as a module.", file=sys.stderr)
