import re
import sys
import os
import json

# Gwt directory path from command line arguments
directory = sys.argv[1] if len(sys.argv) > 1 else '.'

# Get all JSON files in a directory
def get_json_files(directory):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_files.append(os.path.join(directory, filename))
    return json_files

# Load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
# Main function to process JSON files
def main():
    json_files = get_json_files(directory)
    if not json_files:
        print("No JSON files found in the directory.")
        return
    
    data = {}
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        try:
            match = re.search(r'([^-]+).*-0\.7-(\d)-\d{8}-\d{6}\.json', file_name)
            if match and len(match.groups()) == 2:
                model = match.group(1)
                shot = match.group(2)
                if model not in data:
                    data[model] = {}
                data[model][shot] = load_json(file_path)
            elif file_name == "a-multiples-simple.json":
                data['manual'] = {}
                data['manual']['0'] = load_json(file_path)
                data['manual']['1'] = load_json(file_path)
                data['manual']['5'] = load_json(file_path)
            else:
                raise ValueError(f"Invalid file name format: {file_name}")
            print(f"Loaded data from {file_name}")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    synsets = data['manual']['0'].keys()
    if not synsets:
        print("No synsets found in the manual data.")
        return
    
    tsv_header = f"synset\t{"\t".join(data.keys())}\n"
    tsv = {}
    for shot in ['0', '1', '5']:
        tsv[shot] = tsv_header + "\n".join(
            f"{synset}\t" + "\t".join(
                str(data[model].get(shot, {}).get(synset, '')) for model in data
            ) for synset in synsets
        )
    
    # Write TSV files
    for shot, content in tsv.items():
        tsv_file_path = os.path.join(directory, f"agreement-{shot}.tsv")
        with open(tsv_file_path, 'w', encoding='utf-8') as tsv_file:
            tsv_file.write(content)
        print(f"Wrote {tsv_file_path}")

if __name__ == "__main__":
    main()