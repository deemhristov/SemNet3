import random
import string
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

# Remove examples from the data
for example in examples:
    main_synset_id = example['main_synset']['id']
    if main_synset_id in data:
        del data[main_synset_id]

# Parse arguments
parser = argparse.ArgumentParser(description="Run hypernym resolution with specified model and parameters.")
parser.add_argument("--model", type=str, default="llama3.1:8b-instruct-q4_K_M", help="Model name")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter")
parser.add_argument("--examples", type=int, default=0, help="Number of examples to use")
parser.add_argument("--tokens", type=int, default=128, help="Maximum number of tokens in the response")
parser.add_argument("--retries", type=int, default=1, help="Number of retries")
parser.add_argument("--output", type=str, default=f"ranlp_resolved_hypernyms_{"".join(random.choices(string.ascii_lowercase + string.digits, k=8))}.json", help="Output file name")
args = parser.parse_args()


def run_hypernym_resolution(model, parameters=None, num_examples=0):
    chain = RanlpHypernymResolver(model=model, parameters=parameters)

    result = {}

    try:
        bam_start_time = time.time()
        for i, synset_id in enumerate(data.keys()):
            # if i >= 30:
            #     break

            if (synset_id in ['30-08001685-n', '30-12112789-n', '30-09785891-n', '30-07710616-n', '30-07935504-n']):
                continue

            synset_data = noun_data[synset_id]
            hypernyms = [noun_data[hypernym] for hypernym in data[synset_id].keys()]

            for _ in range(1, args.retries + 1):
                print(f"{model} {parameters} {num_examples}-shot: {synset_id} ({i + 1} / {len(data)}) ", end="", flush=True)

                start_time = time.time()
                hypernym, prompt, response = chain.resolve_hypernym(
                    main_synset=synset_data,
                    hypernym_synsets=hypernyms,
                    examples=examples[:num_examples]
                )

                result[synset_id] = hypernym

                now_time = time.time()
                print(f"{now_time - start_time:.3f} s / {now_time - bam_start_time:.3f} s = {hypernym if hypernym is not None else "ERROR"}")
                print(f"[{synset_id}] PROMPT:\n{prompt}\n", file=sys.stderr)
                print(f"[{synset_id}] RESPONSE:\n{response}\n", file=sys.stderr)
                
                if hypernym is not None:
                    break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print(f"\nTotal time taken: {time.time() - bam_start_time:.3f} seconds")
        return result

# Run the hypernym resolution
result = run_hypernym_resolution(model=args.model, parameters={'temperature': args.temperature, 'num_predict': args.tokens}, num_examples=args.examples)

# Print the results
with open(args.output, "w") as f:
    json.dump(result, f, indent=4)
