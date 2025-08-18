# Load JSON file and extract multiples
import json
import sys

def extract_multiples(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        multiples = {}
        for item in data.values():
            print(f"Processing item with ID: {item.get('id', 'unknown')}", file=sys.stderr)
            # count hypernyms that are not instances
            if 'hypernyms' in item and isinstance(item['hypernyms'], list):
                count = sum(1 for hypernym in item['hypernyms'] if not hypernym.get('type', 'instance') == 'instance')
                if count > 1:
                    hypernyms = {}
                    for hypernym in sorted(item['hypernyms'], key=lambda h: h['id']):
                        if not hypernym['type'] == 'instance':
                            hypernyms[hypernym['id']] = {"old": "hypernym"}
                    multiples[item['id']] = hypernyms

        
        return multiples
    
    except Exception as e:
        print(f"Error reading JSON file: {e}", file=sys.stderr)
        sys.exit(1)

# run extract_multiples if this script is executed directly
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_multiples.py <json_file>", file=sys.stderr)
        sys.exit(1)
    
    json_file = sys.argv[1]
    multiples = extract_multiples(json_file)
    
    # Print the result
    json.dump(multiples, sys.stdout, indent=4)
    print()
    print("Number of items with multiple hypernyms:", len(multiples), file=sys.stderr)
else:
    print("This script is intended to be run directly, not imported as a module.", file=sys.stderr)