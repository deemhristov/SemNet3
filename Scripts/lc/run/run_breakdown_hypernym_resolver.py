import sys
from Scripts.lc.breakdown_hypernym_resolver import BreakDownHypernymResolver
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

def get_other_relations(synset_id):
    other_relations = []

    # Get relation data for the selected synset
    synset_data = noun_data[synset_id]
    hypernyms = synset_data.get('hypernyms', [])

    hypernym_hypernyms = set()
    for hypernym in hypernyms:
        hypernym_id = hypernym['id']
        hypernym_data = noun_data[hypernym_id]
        hypernym_hypernyms.update([rel["id"] for rel in hypernym_data.get('hypernyms', [])])
    print(f"Hypernym hypernyms: {hypernym_hypernyms}")
    for id in list(hypernym_hypernyms):
        ss_data = noun_data[id]
        ss_data['relation_type'] = ''
        other_relations.append(ss_data)

    holonyms = synset_data.get('holonyms', [])
    for holonym in holonyms:
        holonym_id = holonym['id']
        holonym_data = noun_data[holonym_id]
        holonym_data['relation_type'] = "holonym"
        other_relations.append(holonym_data)

    meronyms = synset_data.get('meronyms', [])
    for meronym in meronyms:
        meronym_id = meronym['id']
        meronym_data = noun_data[meronym_id]
        meronym_data['relation_type'] = "meronym"
        other_relations.append(meronym_data)
    
    domains = synset_data.get('domains', [])
    for domain in domains:
        domain_id = domain['id']
        domain_data = noun_data[domain_id]
        domain_data['relation_type'] = "domain"
        other_relations.append(domain_data)

    domain_members = synset_data.get('domain_members', [])
    for domain_member in domain_members:
        domain_member_id = domain_member['id']
        domain_member_data = noun_data[domain_member_id]
        domain_member_data['relation_type'] = "domain member"
        other_relations.append(domain_member_data)

    attributes = [relation for relation in synset_data.get('attributes', []) if relation['type'] == 'attribute']
    for attribute in attributes:
        attribute_id = attribute['id']
        attribute_data = noun_data[attribute_id]
        attribute_data['relation_type'] = "attribute"
        other_relations.append(attribute_data)

    return other_relations

synset_id = random.choice(selected_synsets)
print(f"Selected synset ID: {synset_id}")

synset_data = noun_data[synset_id]
hypernyms = [noun_data[hypernym['id']] for hypernym in synset_data.get('hypernyms', [])]
other_relations = get_other_relations(synset_id)

chain = BreakDownHypernymResolver(model="llama3.2")

# Print the results
result = chain.run(
    main_synset=synset_data,
    hypernym_synsets=hypernyms,
    other_synsets=other_relations,
)

json.dump(result, sys.stdout, indent=4)
print("\n\n")