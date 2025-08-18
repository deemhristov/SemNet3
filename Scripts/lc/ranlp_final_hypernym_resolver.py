import re
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class RanlpHypernymResolver:
    def __init__(self, model, parameters=None):
        # Initialize the model
        self.model = ChatOllama(
            model=model,
            **parameters if parameters is not None else {}
        )

    def resolve_hypernym(self, main_synset, hypernym_synsets, examples=None):
        system_prompt = "You are a WordNet expert. Your task is to evaluate hypernymy relations between semantic concepts. Each semantic concept is represented by a group of words with common meaning. This group is called a synset. If concept A is a hypernym of concept B, then concept B is a type of concept A, and concept A is a more generic version of concept B.\n\n"
        system_prompt += "Each synset is presented by its ID, group of words and meaning. You will be given a synset and its hypernyms and will be instructed to choose a single hypernym.\n\n"
        # system_prompt += "Reply with the following format:\n"
        # system_prompt += "<think>\n"
        # system_prompt += "Your reasoning goes here.\n"
        # system_prompt += "</think>\n"
        # system_prompt += "Chosen hypernym synset ID goes here with no other words.\n\n"
        system_prompt += "Reply only with the chosen hypernym synset ID with format 30-<8 digits>-n and no other words.\n\n"

        if examples:
            system_prompt += "Here are some examples for solving the task:\n\n"
            for num, example in enumerate(examples, start=1):
                system_prompt += f"Example {num}\n\n"
                system_prompt += "QUERY:\n"
                system_prompt += self.construct_task_prompt(example['main_synset'], example['hypernym_synsets'])
                system_prompt += "\nRESPONSE:\n"
                system_prompt += f"{example['response']}\n\n"

        user_prompt = self.construct_task_prompt(main_synset, hypernym_synsets)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        response = self.model.invoke(messages).content.strip()
        thinking_match = re.search(r"<think>((?:.|\n)*)</think>", response)
        thinking = thinking_match.group(1).strip() if thinking_match else "N/A"
        chosen_id = re.sub(r"<think>(.|\n)*?</think>", "", response).strip()
        
        # Accept only if the response matches the pattern "30-<8 digits>-n"
        if not re.fullmatch(r"30-\d{8}-n", chosen_id):
            raise ValueError(f"Rejected: {chosen_id}\nThinking: {thinking}")
        return chosen_id, thinking

    def construct_task_prompt(self, main_synset, hypernym_synsets):
        user_prompt = "You are given the following synsets:\n"

        for synset in hypernym_synsets:
            words = ", ".join(f'"{word["word"]}"' for word in synset['words'])
            gloss = synset.get('gloss').split('; "')[0]
            user_prompt += f'- ID {synset["id"]} with words {words} and meaning "{gloss}"\n'
        
        main_words = ", ".join(f'"{word["word"]}"' for word in main_synset['words'])
        main_gloss = main_synset.get('gloss').split('; "')[0]
        
        user_prompt += f"\nWhich of the above {len(hypernym_synsets)} synsets is most likely to be the hypernym of the synset below?\n"
        user_prompt += f"- ID {main_synset['id']} with words {main_words} and meaning \"{main_gloss}\""
        return user_prompt
