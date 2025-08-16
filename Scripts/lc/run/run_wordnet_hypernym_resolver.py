from Scripts.lc.wordnet_hypernym_resolver import WordNetHypernymResolver
import json
import random

# Load the WordNet 3.0 JSON data
with open('Data/wn-3.0-json/noun.json', 'r') as f:
    noun_data = json.load(f)

# Find all synsets with 2 or more hypernyms
synsets_with_multiple_hypernyms = [
    synset for synset, details in noun_data.items()
    if len(details.get('hypernyms', [])) >= 2
]

if not synsets_with_multiple_hypernyms:
    raise ValueError("No synsets with 2 or more hypernyms found in wn-3.0-json/noun.json")

print(f"Found {len(synsets_with_multiple_hypernyms)} synsets with 2 or more hypernyms.")

# Randomly select 5 synsets
selected_synsets = random.sample(synsets_with_multiple_hypernyms, 1)
print(f"Selected synsets: {selected_synsets}")

def get_hypernym_data(synset_id):
    # Get relation data for the selected synset
    synset_data = noun_data[synset_id]
    hypernyms = synset_data.get('hypernyms', [])
    holonyms = synset_data.get('holonyms', [])
    meronyms = synset_data.get('meronyms', [])
    domains = synset_data.get('domains', [])
    domain_members = synset_data.get('domain_members', [])
    other_relations = synset_data.get('other_relations', [])

    # Add all hypernyms recursively to a set
    hypernym_set = set()
    def add_hypernyms(hypernym_id):
        if hypernym_id not in hypernym_set:
            hypernym_set.add(hypernym_id)
            for hypernym in noun_data[hypernym_id].get('hypernyms', []):
                add_hypernyms(hypernym['id'])
    for hypernym in hypernyms:
        add_hypernyms(hypernym['id'])

    # Add all other relations to a set only if they are not already in the hypernym set
    relation_set = set()
    for relation in holonyms + meronyms + domains + domain_members + other_relations:
        if relation['id'] not in hypernym_set:
            relation_set.add(relation['id'])

    # All synsets will be added to a stripped version of the original synset data map where
    # the synset ID is the key and the value is a dictionary with the synset ID, words, gloss,
    # and relations as follows:
    # - task synset will be added with all relations
    # - synsets from the hypernym set will be added only with hypernyms
    # - synsets from the relation set will be added with no relations
    stripped_data = [
        {
            "id": synset_id,
            "words": synset_data['words'],
            "gloss": synset_data['gloss'],
            "hypernyms": hypernyms,
            "holonyms": holonyms,
            "meronyms": meronyms,
            "domains": domains,
            "domain_members": domain_members,
            "other_relations": other_relations
        }
    ]
    for hypernym_id in hypernym_set:
        stripped_data.append({
            "id": hypernym_id,
            "words": noun_data[hypernym_id]['words'],
            "gloss": noun_data[hypernym_id]['gloss'],
            "hypernyms": noun_data[hypernym_id].get('hypernyms', [])
        })
    for relation_id in relation_set:
        stripped_data.append({
            "id": relation_id,
            "words": noun_data[relation_id]['words'],
            "gloss": noun_data[relation_id]['gloss']
        })

    return stripped_data

# Create a new instance of the WordNetHypernymResolver class
resolver = WordNetHypernymResolver(model="llama3.1")

# Run the resolver on the first selected synset
synset_id = selected_synsets[0]
stripped_data = get_hypernym_data(synset_id)

# print("Stripped data for synset ID:", synset_id)
# print(stripped_data)

result = resolver.run(
    synset_id=synset_id,
    wn_data=stripped_data
)
print(result)