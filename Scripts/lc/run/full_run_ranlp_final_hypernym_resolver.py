import sys
import json
import time
import os
import argparse

SN3_ROOT = os.environ.get("SN3_ROOT")
if SN3_ROOT is None:
    raise EnvironmentError("SN3_ROOT environment variable is not set.")

# Import the RanlpHypernymResolver class from SN3_ROOT
sys.path.append(f"{SN3_ROOT}/Scripts")
from lc.ranlp_final_hypernym_resolver import RanlpHypernymResolver  # FIXED: Correct class name

# Load multiples data
with open(f'{SN3_ROOT}/Data/multiples-test.json', 'r') as f:
    data = json.load(f)

# Load the WordNet 3.0 JSON data
with open(f'{SN3_ROOT}/Data/wn-3.0-json/noun.json', 'r') as f:
    noun_data = json.load(f)

# 5 examples
examples = [
    {
        'main_synset': noun_data['30-08001685-n'],
        'hypernym_synsets': [noun_data[hypernym] for hypernym in data['30-08001685-n'].keys()],
        'response': '30-07999699-n'
    },
    {
        'main_synset': noun_data['30-12112789-n'],
        'hypernym_synsets': [noun_data[hypernym] for hypernym in data['30-12112789-n'].keys()],
        'response': '30-11556857-n'
    },
    {
        'main_synset': noun_data['30-09785891-n'],
        'hypernym_synsets': [noun_data[hypernym] for hypernym in data['30-09785891-n'].keys()],
        'response': '30-10285313-n'
    },
    {
        'main_synset': noun_data['30-07710616-n'],
        'hypernym_synsets': [noun_data[hypernym] for hypernym in data['30-07710616-n'].keys()],
        'response': '30-07710007-n'
    },
    {
        'main_synset': noun_data['30-07935504-n'],
        'hypernym_synsets': [noun_data[hypernym] for hypernym in data['30-07935504-n'].keys()],
        'response': '30-14940386-n'
    }
]


def run_hypernym_resolution(model, parameters=None, num_examples=0):
    chain = RanlpHypernymResolver(model=model, parameters=parameters)

    result = {}

    bam_start_time = time.time()
    for i, synset_id in enumerate(data.keys()):
        # if i >= 30:
        #     break

        print(f"Selected synset ID: {synset_id} ({i + 1} / {len(data)}) ", end="", flush=True, file=sys.stderr)

        if (synset_id in ['30-08001685-n', '30-12112789-n', '30-09785891-n', '30-07710616-n', '30-07935504-n']):
            print("Example synset, skipping...", file=sys.stderr)
            continue

        synset_data = noun_data[synset_id]
        hypernyms = [noun_data[hypernym] for hypernym in data[synset_id].keys()]

        result[synset_id] = {}

        start_time = time.time()
        try:
            hypernym, thinking = chain.resolve_hypernym(
                main_synset=synset_data,
                hypernym_synsets=hypernyms,
                examples=examples
            )
            for hypernym_id in data[synset_id].keys():
                result[synset_id][hypernym_id] = "hypernym" if hypernym_id == hypernym else None
            result[synset_id]['thinking'] = thinking

            now_time = time.time()
            print(f"{now_time - start_time:.3f} s / {now_time - bam_start_time:.3f} s ", file=sys.stderr)
        except Exception as e:
            result[synset_id]['error'] = str(e)
            print(f"{time.time() - start_time:.3f} s until error:", file=sys.stderr)
            print(str(e), file=sys.stderr)
            print("Words:", [word['word'] for word in synset_data['words']], file=sys.stderr)
            print("Gloss:", synset_data['gloss'], file=sys.stderr)
            # raise e

    return result

# Run the hypernym resolution
parser = argparse.ArgumentParser(description="Run hypernym resolution with specified model and parameters.")
parser.add_argument("--model", type=str, default="llama3.1:8b-instruct-q4_K_M", help="Model name")
parser.add_argument("--temperature", type=float, default=0.5, help="Temperature parameter")
parser.add_argument("--examples", type=int, default=5, help="Number of examples to use")
args = parser.parse_args()

model = args.model
temperature = args.temperature
num_examples = args.examples

result = run_hypernym_resolution(model=model, parameters={'temperature': temperature}, num_examples=num_examples)

# Print the results
timestamp = time.strftime("%Y%m%d-%H%M%S")

with open(f"{SN3_ROOT}/Runs/result-{timestamp}.json", "w") as f:
    json.dump(result, f, indent=4)
