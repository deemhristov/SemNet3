import random
import json
import sys

# This script converts the raw WordNet 3.0 and 3.1 noun data into a JSON format.
synsets = {}

with (open(sys.argv[1], "r") if len(sys.argv) > 1 else sys.stdin) as file:
    lines = [line for line in file.readlines() if not line.startswith("  ")]
    for line_id, line in enumerate(lines):
        print(f"\rProcessing line {line_id + 1}/{len(lines)}", end="", file=sys.stderr)

        line_it = iter(line.split(" "))

        synset_id = next(line_it) + "-n"  # Append "-n" to the synset ID to indicate noun
        next(line_it)  # Skip lexicographer file number
        pos = next(line_it)

        synset = {
            "id": synset_id,
            "pos": pos,
            "words": [],
            "hypernyms": [],
            "hyponyms": [],
            "holonyms": [],
            "meronyms": [],
            "domains": [],
            "domain_members": [],
            "other_relations": [],
            "gloss": line.split("|")[-1].strip()
        }
        synsets[synset_id] = synset

        # Read words
        num_words = int(next(line_it), 16)
        for _ in range(num_words):
            word = next(line_it).replace("_", " ")
            lex_id = int(next(line_it), 16)
            synset["words"].append({"word": word, "lex_id": lex_id})

        # Read relations
        num_relations = int(next(line_it))
        for _ in range(num_relations):
            rel_type = next(line_it)
            rel_target = next(line_it) + "-n"  # Append "-n" to the target ID
            rel_pos = next(line_it)
            rel_src_trg = next(line_it)

            if (rel_pos != 'n' or rel_src_trg != '0000'):
                # Skip relations that are not nouns or have a source target other than whole synset
                continue

            match rel_type:
                case "@": synset["hypernyms"].append({"id": rel_target, "type": "regular"})
                case "@i": synset["hypernyms"].append({"id": rel_target, "type": "instance"})
                case "~": synset["hyponyms"].append({"id": rel_target, "type": "regular"})
                case "~i": synset["hyponyms"].append({"id": rel_target, "type": "instance"})
                case "#m": synset["holonyms"].append({"id": rel_target, "type": "member"})
                case "#s": synset["holonyms"].append({"id": rel_target, "type": "substance"})
                case "#p": synset["holonyms"].append({"id": rel_target, "type": "part"})
                case "%m": synset["meronyms"].append({"id": rel_target, "type": "member"})
                case "%s": synset["meronyms"].append({"id": rel_target, "type": "substance"})
                case "%p": synset["meronyms"].append({"id": rel_target, "type": "part"})
                case ";c": synset["domains"].append({"id": rel_target, "type": "topic"})
                case ";r": synset["domains"].append({"id": rel_target, "type": "region"})
                case ";u": synset["domains"].append({"id": rel_target, "type": "usage"})
                case "-c": synset["domain_members"].append({"id": rel_target, "type": "topic"})
                case "-r": synset["domain_members"].append({"id": rel_target, "type": "region"})
                case "-u": synset["domain_members"].append({"id": rel_target, "type": "usage"})
                case "!": synset["other_relations"].append({"id": rel_target, "type": "antonym"})
                case "=": synset["other_relations"].append({"id": rel_target, "type": "attribute"})

    print(file=sys.stderr)  # Print a newline after processing all lines


# Save the synsets to a JSON file
print("Saving synsets to JSON file...", file=sys.stderr)

if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as json_file:
        json.dump(synsets, json_file, indent=4)
else:  # Output to stdout
    json.dump(synsets, sys.stdout, indent=4)
    print(file=sys.stdout)