import os
import json
import random


# Load the WordNet 3.0 JSON data
sn3_root = os.getenv('SN3_ROOT', os.path.dirname(os.path.abspath(__file__)))
with open(f'{sn3_root}/Data/wn-3.0-json/noun.json', 'r') as f:
    noun_data = json.load(f)

def join_comma_and(words):
    if len(words) == 1:
        return words[0]
    elif len(words) == 2:
        return f'{words[0]} and {words[1]}'
    else:
        return ', '.join(words[:-1]) + ', and ' + words[-1]

def hypo_word_gloss_to_hyper_words_gloss(synset_hypo, synset_hyper):
    # Chose randomly one of the hyponym words
    hypo_word = random.choice(synset_hypo['words']).get('word')
    hypo_gloss = synset_hypo['gloss'].split('; "')[0]  # Take the first part of the gloss without the example

    hyper_words = [f'"{word.get('word')}"' for word in synset_hyper['words']]
    hyper_words_f = join_comma_and(hyper_words)
    hyper_gloss = synset_hyper['gloss'].split('; "')[0]  # Take the first part of the gloss without the example

    match random.randint(1, 2):
        case 1: return {
            "in": f'What are the hypernyms of "{hypo_word}" with meaning "{hypo_gloss}"?', 
            "out": f'{hyper_words_f}, {"all of which mean" if len(hyper_words) > 1 else "which means"} "{hyper_gloss}".',
            "synsets": [synset_hypo['id'], synset_hyper['id']],
            "variant": "g.wg.1",
            "type": "QA"
        }
        case 2: return {
            "in": f'"{hypo_word}" with meaning "{hypo_gloss}" has hypernym{"s" if len(hyper_words) > 1 else ""}',
            "out": f'{hyper_words_f}, {"all of which mean" if len(hyper_words) > 1 else "which means"} "{hyper_gloss}".',
            "synsets": [synset_hypo['id'], synset_hyper['id']],
            "variant": "g.wg.2",
            "type": "Statement"
        }

def hypo_id_words_gloss_to_hyper_id_words_gloss(synset_hypo, synset_hyper):
    # Chose randomly one of the hyponym words
    hypo_id = synset_hypo['id']
    hypo_words = [f'"{word.get('word')}"' for word in synset_hypo['words']]
    hypo_words_f = join_comma_and(hypo_words)
    hypo_gloss = synset_hypo['gloss'].split('; "')[0]  # Take the first part of the gloss without the example

    hyper_id = synset_hyper['id']
    hyper_words = [f'"{word.get('word')}"' for word in synset_hyper['words']]
    hyper_words_f = join_comma_and(hyper_words)
    hyper_gloss = synset_hyper['gloss'].split('; "')[0]  # Take the first part of the gloss without the example

    match random.randint(1, 4):
        case 1: return {
            "in": f'Give me an example hyponym synset of synset {hyper_id} with word{"s" if len(hyper_words) > 1 else ""} {hyper_words_f}, with gloss "{hyper_gloss}"',
            "out": f'Synset {hypo_id}, containing {hypo_words_f}, {"all of which mean" if len(hypo_words) > 1 else "which means"} "{hypo_gloss}".',
            "synsets": [synset_hypo['id'], synset_hyper['id']],
            "variant": "iwg.iwg.1",
            "type": "QA"
        }
        case 2: return {
            "in": f'Synset {hyper_id} with word{"s" if len(hyper_words) > 1 else ""} {hyper_words_f} and gloss "{hyper_gloss}" has hyponym',
            "out": f'synset {hypo_id}, containing {hypo_words_f}, {"all of which mean" if len(hypo_words) > 1 else "which means"} "{hypo_gloss}".',
            "synsets": [synset_hypo['id'], synset_hyper['id']],
            "variant": "iwg.iwg.2",
            "type": "Statement"
        }
        case 3: return {
            "in": f'Which synset is the hypernym of synset {hypo_id}, containing {hypo_words_f}, {"all of which mean" if len(hypo_words) > 1 else "which means"} "{hypo_gloss}"?',
            "out": f'Synset {hyper_id} with word{"s" if len(hyper_words) > 1 else ""} {hyper_words_f} and gloss "{hyper_gloss}".',
            "synsets": [synset_hypo['id'], synset_hyper['id']],
            "variant": "iwg.iwg.3",
            "type": "QA"
        }
        case 4: return {
            "in": f'The hypernym of synset {hypo_id}, containing {hypo_words_f}, {"all of which mean" if len(hypo_words) > 1 else "which means"} {hypo_gloss}, is',
            "out": f'synset {hyper_id} with word{"s" if len(hyper_words) > 1 else ""} {hyper_words_f} and gloss "{hyper_gloss}".',
            "synsets": [synset_hypo['id'], synset_hyper['id']],
            "variant": "iwg.iwg.4",
            "type": "Statement"
        }

def id_gloss(synset):
    synset_id = synset['id']
    gloss = synset['gloss'].split('; "')[0]  # Take the first part of the gloss without the example
    
    match random.randint(1, 4):
        case 1: return {
            "in": f'What is the gloss for synset with ID {synset_id}?',
            "out": gloss,
            "synsets": [synset_id],
            "variant": "ig.1",
            "type": "QA"
        }
        case 2: return {
            "in": f'The gloss for synset with ID {synset_id} is',
            "out": f'"{gloss}".',
            "synsets": [synset_id],
            "variant": "ig.2",
            "type": "Statement"
        }
        case 3: return {
            "in": f'Which synset has definition "{gloss}"?',
            "out": synset_id,
            "synsets": [synset_id],
            "variant": "ig.3",
            "type": "QA"
        }
        case 4: return {
            "in": f'The definition "{gloss}" belongs to',
            "out": f'synset with ID {synset_id}.',
            "synsets": [synset_id],
            "variant": "ig.4",
            "type": "Statement"
        }

def id_words(synset):
    synset_id = synset['id']
    synset_words = [f'"{word.get("word")}"' for word in synset['words']]
    synset_words_f = join_comma_and(synset_words)

    match random.randint(1, 2):
        case 1: return {
            "in": f'What words are in synset with ID {synset_id}?',
            "out": synset_words_f,
            "synsets": [synset_id],
            "variant": "iw.1",
            "type": "QA"
        }
        case 2: return {
            "in": f'Synset with ID {synset_id} contains',
            "out": f'the word{"s" if len(synset_words) > 1 else ""} {synset_words_f}.',
            "synsets": [synset_id],
            "variant": "iw.2",
            "type": "Statement"
        }

def word_ids(word, noun_data):
    synset_ids = [synset['id'] for synset in noun_data.values() if any(w.get('word') == word for w in synset['words'])]

    match random.randint(1, 2):
        case 1: return {
            "in": f'Which synsets is "{word}" part of?',
            "out": f'{join_comma_and(synset_ids)}.',
            "synsets": synset_ids,
            "variant": "i.1",
            "type": "QA"
        }
        case 2: return {
            "in": f'"{word}" is in',
            "out": f'synset{"s" if len(synset_ids) > 1 else ""} with ID{"s" if len(synset_ids) > 1 else ""} {join_comma_and(synset_ids)}.',
            "synsets": synset_ids,
            "variant": "i.2",
            "type": "Statement"
        }

# ss_hypo_hyper_ids = []
# while len(ss_hypo_hyper_ids) == 0:
#     ss_hypo = random.choice(list(noun_data.values()))
#     ss_hypo_hyper_ids = [ss.get('id') for ss in ss_hypo.get('hypernyms', []) if ss.get('type') == 'regular']

# print(ss_hypo['id'])

# ss_hyper_id = random.choice(ss_hypo_hyper_ids)
# ss_hyper = noun_data[ss_hyper_id]

# print(hypo_word_gloss_to_hyper_words_gloss(ss_hypo, ss_hyper))
# print(hypo_id_words_gloss_to_hyper_id_words_gloss(ss_hypo, ss_hyper))
# print(id_gloss(ss_hypo))
# print(id_words(ss_hypo))
# print(word_ids(random.choice(ss_hypo['words']).get('word'), noun_data))

# Create the dataset
examples = []
test = []
for ss_id in noun_data:
    ss = noun_data[ss_id]
    if 'done' not in ss:
        ss['done'] = False

    # Generate examples and test cases
    if ss.get('hypernyms'):
        for ss_hyper in ss['hypernyms']:
            if ss_hyper.get('type') == 'regular':
                ss_h_data = noun_data[ss_hyper['id']]
                eg_variant = random.choice(['g.wg', 'iwg.iwg'])
                if eg_variant == 'g.wg':
                    examples.append(hypo_word_gloss_to_hyper_words_gloss(ss, noun_data[ss_hyper['id']]))
                    test.append(hypo_id_words_gloss_to_hyper_id_words_gloss(ss, noun_data[ss_hyper['id']]))
                else:
                    examples.append(hypo_id_words_gloss_to_hyper_id_words_gloss(ss, noun_data[ss_hyper['id']]))
                    test.append(hypo_word_gloss_to_hyper_words_gloss(ss, noun_data[ss_hyper['id']]))
                    # Mark the synset as done when ID, words, and gloss are all used
                    ss['done'] = True
                    ss_h_data['done'] = True

for ss_id in noun_data:
    ss = noun_data[ss_id]
    if 'done' not in ss or not ss['done']:
        examples.append(id_gloss(ss))
        examples.append(id_words(ss))
    elif random.random() < 0.2:  # 20% chance to include in test set
        test.append(id_gloss(ss))
        test.append(id_words(ss))

# Save the examples to a file
with open(f'{sn3_root}/Data/wn-3.0-json/dataset.jsonl', 'w') as f:
    for example in examples:
        f.write(json.dumps(example) + '\n')

# Save the test cases to a file
with open(f'{sn3_root}/Data/wn-3.0-json/test.jsonl', 'w') as f:
    for example in test:
        f.write(json.dumps(example) + '\n')