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
new_relations = set()

bam_start_time = time.time()
for i, synset_id in enumerate(synsets_with_multiple_hypernyms):
    # if i >= 30:
    #     break

    print(f"Selected synset ID: {synset_id} ({i + 1} / {len(synsets_with_multiple_hypernyms)}) ", end="", flush=True, file=sys.stderr)

    synset_data = noun_data[synset_id]
    hypernyms = [noun_data[hypernym['id']] for hypernym in synset_data.get('hypernyms', [])]
    other_synsets = sd_hypernyms(synset_data)

    result[synset_id] = {}
    for hypernym in hypernyms:
        result[synset_id][hypernym['id']] = {"old": "hypernym", "new": None}
    if other_synsets:
        result[synset_id][other_synsets[0]['id']] = {"old": None, "new": None}

    start_time = time.time()
    try:
        if other_synsets:
            hypernym = chain.resolve_hypernym_extra_a(
                main_synset=synset_data,
                hypernym_synsets=hypernyms,
                other_synsets=other_synsets,
            )
        else:
            hypernym = chain.resolve_hypernym(
                main_synset=synset_data,
                hypernym_synsets=hypernyms,
                # other_synsets=other_synsets,
            )
        for hypernym_id in result[synset_id].keys():
            if hypernym_id == hypernym:
                result[synset_id][hypernym_id]['new'] = "hypernym"
            else:
                result[synset_id][hypernym_id]['new'] = None
                if result[synset_id][hypernym_id]['old'] == "hypernym":
                    new_relation = chain.propose_alternative_relation(
                        synset_a=noun_data[hypernym_id],
                        synset_b=synset_data,
                    )
                    result[synset_id][hypernym_id]['new'] = new_relation
                    # print(new_relation, file=sys.stderr)
                    if new_relation is not None and new_relation not in [
                        "holonym", "meronym", "domain", "domain member",
                        "function", "property", "characteristic", "used for",
                        "uses", "form", "purpose", "form of", "origin"
                    ]:
                        # print(f"New relation found: {new_relation}", file=sys.stderr)
                        new_relations.add(new_relation)
        now_time = time.time()
        print(f"{now_time - start_time:.3f} s / {now_time - bam_start_time:.3f} s ", file=sys.stderr)
    except Exception as e:
        result[synset_id]['error'] = str(e)
        print(f"{time.time() - start_time:.3f} s until error:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("Words:", [word['word'] for word in synset_data['words']], file=sys.stderr)
        print("Gloss:", synset_data['gloss'], file=sys.stderr)
        # raise e

print(f"Found {len(new_relations)} new relations.", file=sys.stderr)
new_relation_mapping = {}
if new_relations:
    new_relation_mapping = chain.group_new_relations(new_relations)

# Print the results
timestamp = time.strftime("%Y%m%d-%H%M%S")

with open(f"{SN3_ROOT}/Results/result-{timestamp}.txt", "w") as f:
    json.dump(result, f, indent=4)

with open(f"{SN3_ROOT}/Results/new_relations-{timestamp}.txt", "w") as f:
    json.dump(new_relation_mapping, f, indent=4)