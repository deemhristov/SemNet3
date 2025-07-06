import sys
from lc.ranlp_hypernym_resolver import RanlpHypernymResolver
import json
import time
import os

SN3_ROOT = os.environ.get("SN3_ROOT")
if SN3_ROOT is None:
    raise EnvironmentError("SN3_ROOT environment variable is not set.")

# Load multiples data
with open(f'{SN3_ROOT}/Data/multiples.json', 'r') as f:
    m_data = json.load(f)

# Load the WordNet 3.0 JSON data
with open(f'{SN3_ROOT}/Data/wn-3.0-json/noun.json', 'r') as f:
    noun_data = json.load(f)

# Find all synsets with 2 or more hypernyms
synsets_with_multiple_hypernyms = [
    synset for synset, details in noun_data.items()
    if len([hn for hn in details.get('hypernyms', []) if hn['type'] == "regular"]) >= 2
]

if not synsets_with_multiple_hypernyms:
    raise ValueError("No synsets with 2 or more hypernyms found in wn-3.0-json/noun.json")

print(f"Found {len(synsets_with_multiple_hypernyms)} synsets with 2 or more hypernyms.", file=sys.stderr)

def sd_hypernyms(synset_data):
    """Get the direct common hypernym of the synset's hypernyms if it exists."""
    hypernyms = synset_data.get('hypernyms', [])
    hypernym_ids = [hypernym['id'] for hypernym in hypernyms]
    # List of sets of hypernyms' hypernyms' IDs
    hypernym_hypernyms = [
        set(
            hh['id'] for hh in noun_data[hyper_id].get('hypernyms', [])
            if hh.get('type') == 'regular'
        )
        for hyper_id in hypernym_ids if hyper_id in noun_data
    ]
    common_hypernyms = set.intersection(*hypernym_hypernyms) if hypernym_hypernyms else set()
    # return as list of synsets
    return [noun_data[hyper_id] for hyper_id in common_hypernyms if hyper_id in noun_data]

chain = RanlpHypernymResolver(model="llama3.1")

# synset_id = random.choice(synsets_with_multiple_hypernyms)

result = {}
for i, synset_id in enumerate(synsets_with_multiple_hypernyms):
    if i > 100: # Limit to the first 100 synsets
        break

    print(f"Selected synset ID: {synset_id} ({i + 1} / {len(synsets_with_multiple_hypernyms)}) ", end="", flush=True, file=sys.stderr)

    synset_data = noun_data[synset_id]
    hypernyms = [noun_data[hypernym['id']] for hypernym in synset_data.get('hypernyms', [])]
    other_synsets = sd_hypernyms(synset_data)

    start_time = time.time()
    try:
        result[synset_id] = chain.resolve_hypernym(
            main_synset=synset_data,
            hypernym_synsets=hypernyms,
            # other_synsets=other_synsets,
        )
        print(f"{time.time() - start_time:.3f} s", file=sys.stderr)
    except Exception as e:
        print(f"{time.time() - start_time:.3f} s until error:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("Words:", [word['word'] for word in synset_data['words']], file=sys.stderr)
        print("Gloss:", synset_data['gloss'], file=sys.stderr)

# Print the results
json.dump(result, sys.stdout, indent=4)
print()