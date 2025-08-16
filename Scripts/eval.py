import os
import json

SN3_ROOT = os.environ.get('SN3_ROOT')
if SN3_ROOT is None:
    raise EnvironmentError("SN3_ROOT environment variable is not set.")

multiples_path = os.path.join(f'{SN3_ROOT}', 'Data', 'multiples.json')
with open(multiples_path, 'r') as f:
    multiples = json.load(f)

multiples_path = os.path.join(f'{SN3_ROOT}', 'Results', 'result-20250706-142554.json')
with open(multiples_path, 'r') as f:
    result = json.load(f)

errors = 0
different = 0
hyper = 0
hyper_bad = 0
other = 0
other_bad = 0

print (len(result.items()))
for synset_id, synset_data in result.items():
    if 'error' in synset_data:
        errors += 1
        continue

    if synset_id not in multiples.keys():
        different += 1
        continue
    for hn in multiples[synset_id].keys():
        if hn not in synset_data.keys():
            different += 1
            continue

        # print(synset_data[hn]['new'], multiples[synset_id][hn]['new'])
        if synset_data[hn]['new'] == 'hypernym' and multiples[synset_id][hn]['new'] == 'hypernym':
            hyper += 1
        elif synset_data[hn]['new'] == 'hypernym' and multiples[synset_id][hn]['new'] != 'hypernym':
            hyper_bad += 1
        elif synset_data[hn]['new'] == multiples[synset_id][hn]['new']:
            other += 1
        else:
            other_bad += 1

print(f"Errors: {errors}")
print(f"Different: {different}")
print(f"Hypernym: {hyper}")
print(f"Hypernym bad: {hyper_bad}")
print(f"Other: {other}")
print(f"Other bad: {other_bad}")