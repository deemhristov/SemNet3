import re
import sys
from langchain_ollama import OllamaLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class RanlpHypernymResolver:
    def __init__(self, model, parameters=None):
        # Initialize the model
        self.model = OllamaLLM(
            model=model,
            **parameters if parameters is not None else {}
        )

    def resolve_hypernym(self, main_synset, hypernym_synsets, examples=None):
        prompt = "You are a WordNet expert. Your task is to evaluate hypernymy relations between semantic concepts. Each semantic concept is represented by a group of words with common meaning. This group is called a synset. If concept A is a hypernym of concept B, then concept B is a type of concept A, and concept A is a more generic version of concept B.\n\n"
        prompt += "Each synset is presented by its ID, group of words and meaning. You will be given a synset and its hypernyms and will be instructed to choose a single hypernym.\n\n"
        prompt += "Reply only with the chosen hypernym synset ID with format 30-<8 digits>-n and no other words. Do not give any reasoning and do not generate other text.\n\n"

        if examples:
            # prompt += f"Here {"is an example" if len(examples) == 1 else "are some examples"} for solving the task:\n\n"
            for num, example in enumerate(examples, start=1):
                if len(examples) == 1:
                    prompt += f"EXAMPLE\n\n"
                else:
                    prompt += f"EXAMPLE {num}\n\n"
                prompt += self.construct_task_prompt(example['main_synset'], example['hypernym_synsets'])
                prompt += f"\n\n{example['response']}\n\n"

        if len(examples) > 0:
            prompt += "TASK\n\n"
        prompt += self.construct_task_prompt(main_synset, hypernym_synsets)
        
        response = self.model.invoke(prompt).strip()

        synset_id_matches = re.findall(fr"\b30-\d{{8}}-n\b", response)
        synset_id_matches = [match for match in synset_id_matches if match != main_synset['id']]
        if len(synset_id_matches) == 0 or synset_id_matches[-1] not in [hn['id'] for hn in hypernym_synsets]:
            result = None
        else:
            result = synset_id_matches[-1]

        return result, prompt, response

    def construct_task_prompt(self, main_synset, hypernym_synsets):
        task_prompt = "You are given the following synsets:\n"

        for synset in hypernym_synsets:
            words = ", ".join(f'"{word["word"]}"' for word in synset['words'])
            gloss = synset.get('gloss').split('; "')[0]
            task_prompt += f'- ID {synset["id"]} with words {words} and meaning "{gloss}"\n'
        
        main_words = ", ".join(f'"{word["word"]}"' for word in main_synset['words'])
        main_gloss = main_synset.get('gloss').split('; "')[0]
        
        task_prompt += f"\nWhich of the synsets {", ".join([hn['id'] for hn in hypernym_synsets][:-1])} and {hypernym_synsets[-1]['id']} is most likely to be the hypernym of synset {main_synset['id']} defined as:\n"
        task_prompt += f"- ID {main_synset['id']} with words {main_words} and meaning \"{main_gloss}\""
        return task_prompt
