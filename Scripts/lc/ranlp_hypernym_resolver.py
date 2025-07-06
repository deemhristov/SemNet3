import json
import sys
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class RanlpHypernymResolver:
    def __init__(self, model):
        # Initialize the model
        self.model = ChatOllama(
            model=model,
            temperature=0.5,
        )

    def resolve_hypernym(self, main_synset, hypernym_synsets):
        system_prompt = "You are a WordNet expert. Your task is to evaluate hypernymy relations between semantic concepts. Each semantic concept is represented by a group of words with common meaning. This group is called a synset. If concept A is a hypernym of concept B, then concept B is a type of concept A, and concept A is a more generic version of concept B.\n\n"
        system_prompt += "Each synset is presented by its ID, group of words and meaning. Reply only with the chosen synset ID and no other words."

        user_prompt = "You are given the following synsets:\n"

        for synset in hypernym_synsets:
            words = ", ".join(f'"{word["word"]}"' for word in synset['words'])
            gloss = synset.get('gloss').split('; "')[0]
            user_prompt += f'- ID {synset["id"]} with words {words} and meaning "{gloss}"\n'
        
        main_words = ", ".join(f'"{word["word"]}"' for word in main_synset['words'])
        main_gloss = main_synset.get('gloss').split('; "')[0]
        
        user_prompt += f"\nWhich of the above {len(hypernym_synsets)} synsets is most likely to be the hypernym of the synset below?\n"
        user_prompt += f"- ID {main_synset['id']} with words {main_words} and meaning \"{main_gloss}\""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        return self.model.invoke(messages).content.strip()

    def resolve_hypernym_extra_a(self, main_synset, hypernym_synsets, other_synsets):
        system_prompt = "You are a WordNet expert. Your task is to evaluate hypernymy relations between semantic concepts. Each semantic concept is represented by a group of words with common meaning. This group is called a synset. If concept A is a hypernym of concept B, then concept B is a type of concept A, and concept A is a more generic version of concept B.\n\n"
        system_prompt += "Each synset is presented by its ID, group of words and meaning. Reply only with the chosen synset ID and no other words."

        user_prompt = "You are given the following synsets:\n"

        for synset in hypernym_synsets:
            words = ", ".join(f'"{word["word"]}"' for word in synset['words'])
            gloss = synset.get('gloss').split('; "')[0]
            user_prompt += f'- ID {synset["id"]} with words {words} and meaning "{gloss}"\n'
        
        main_words = ", ".join(f'"{word["word"]}"' for word in main_synset['words'])
        main_gloss = main_synset.get('gloss').split('; "')[0]
        sd_words = ", ".join(f'"{word["word"]}"' for word in other_synsets[0]['words'])
        sd_gloss = other_synsets[0].get('gloss').split('; "')[0]
        
        user_prompt += f"\nWhich of the above {len(hypernym_synsets)} synsets is most likely to be the hypernym of the synset below?\n"
        user_prompt += f"- ID {main_synset['id']} with words {main_words} and meaning \"{main_gloss}\""

        user_prompt += "If the likelihood of the above synsets to be the chosen hypernym is very close, consider choosing the more generic synset:\n"
        user_prompt += f"- ID {other_synsets[0]['id']} with words {sd_words} and meaning \"{sd_gloss}\""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        return self.model.invoke(messages).content.strip()

    def resolve_hypernym_extra_b(self, main_synset, hypernym_synsets, other_synsets):
        system_prompt = "You are a WordNet expert. Your task is to evaluate hypernymy relations between semantic concepts. Each semantic concept is represented by a group of words with common meaning. This group is called a synset. If concept A is a hypernym of concept B, then concept B is a type of concept A, and concept A is a more generic version of concept B.\n\n"
        system_prompt += "Each synset is presented by its ID, group of words and meaning. Reply only with the chosen synset ID and no other words."

        user_prompt = "You are given the following synsets:\n"

        all_hypernyms = hypernym_synsets + [other_synsets[0]]  # Add the other synset to the hypernym list
        for synset in all_hypernyms:
            words = ", ".join(f'"{word["word"]}"' for word in synset['words'])
            gloss = synset.get('gloss').split('; "')[0]
            user_prompt += f'- ID {synset["id"]} with words {words} and meaning "{gloss}"\n'
        
        main_words = ", ".join(f'"{word["word"]}"' for word in main_synset['words'])
        main_gloss = main_synset.get('gloss').split('; "')[0]
        
        user_prompt += f"\nWhich of the above {len(all_hypernyms)} synsets is most likely to be the hypernym of the synset below?\n"
        user_prompt += f"- ID {main_synset['id']} with words {main_words} and meaning \"{main_gloss}\""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        return self.model.invoke(messages).content.strip()
    
    def propose_alternative_relation(self, synset_a, synset_b):
        system_prompt = "You are a WordNet expert. Your task is to evaluate hypernymy relations between semantic concepts. Each semantic concept is represented by a group of words with common meaning. This group is called a synset. You will work with the following types of relations:\n\n"
        system_prompt += "If concept A is a holonym of concept B, then concept B is a substance, part or member of concept A, and concept A contains concept B as a substance, part of member.\n"
        system_prompt += "If concept A is a meronym of concept B, then concept B contains concept A as a substance, part of member, and concept A is a substance, part or member of concept B.\n"
        system_prompt += "If concept A is a domain of concept B, then concept B is within the topic, region or usage for concept A, and concept A describes a topic, region or usage for concept B.\n"
        system_prompt += "If concept A is a domain member of concept B, then concept B describes a topic, region or usage for concept A, and concept A is within the topic, region or usage for concept B.\n\n"
        system_prompt += "Reply as the user requests within the user prompt."

        a_words = ", ".join(f'"{word["word"]}"' for word in synset_a['words'])
        a_gloss = synset_a.get('gloss').split('; "')[0]
        b_words = ", ".join(f'"{word["word"]}"' for word in synset_b['words'])
        b_gloss = synset_b.get('gloss').split('; "')[0]

        user_prompt_1 = f"You are given the following synsets:\n"
        user_prompt_1 += f"- ID {synset_a['id']} with words {a_words} and meaning \"{a_gloss}\"\n"
        user_prompt_1 += f"- ID {synset_b['id']} with words {b_words} and meaning \"{b_gloss}\"\n\n"
        user_prompt_1 += "Is there a notable semantic link between the above synsets? Reply only with \"Yes\" or \"No\".\n"

        user_prompt_2 = "If you replied \"Yes\", specify the type of relation between the above synsets, where the first synset describes concept A, and the second synset describes concept B.\n"
        user_prompt_2 += "Reply only with one of the following types:\n"
        user_prompt_2 += "- holonym\n"
        user_prompt_2 += "- meronym\n"
        user_prompt_2 += "- domain\n"
        user_prompt_2 += "- domain member\n"
        user_prompt_2 += "If the above list does not contain the type of relation you want to specify, reply with \"other\".\n"

        user_prompt_3 = "If you replied \"other\", specify the type of relation between the above synsets, where the first synset describes concept A, and the second synset describes concept B.\n"
        user_prompt_3 += "Reply only with a short descriptive name of one or, if needed, two words.\n"

        messages = [
            SystemMessage(content=system_prompt),
        ]

        messages.append(HumanMessage(content=user_prompt_1) )
        response = self.model.invoke(messages)
        if response.content.strip().lower() != "yes":
            return None
        
        messages.append(HumanMessage(content=user_prompt_2))
        response = self.model.invoke(messages)
        relation_type = response.content.strip().lower()
        if relation_type != "other":
            return relation_type
        
        messages.append(HumanMessage(content=user_prompt_3))
        response = self.model.invoke(messages)
        return response.content.strip().lower()

    def group_new_relations(self, relation_types):
        system_prompt = "You are a WordNet expert. Your task is to group semantic relations between semantic concepts. Each semantic concept is represented by a group of words with common meaning. This group is called a synset.\n\n"
        system_prompt += "You are given a list of semantic relation types. Each type describes a relation between two concepts. Execute the user's request based on your understanding of the relations' meanings.\n"
        
        user_prompt_1 = "Group the following relations into groups such that all relation names in a group are synonyms:\n"
        for relation in relation_types:
            user_prompt_1 += f"- {relation}\n"
        user_prompt_1 += "\nReply with the grouped relations, where each group is written on one line as a comma separated list."

        user_prompt_2 = "If you grouped the relations into groups, specify the group names, where each group name is a short descriptive name of one or, if needed, two words, with the same meaning as the members of the group.\n"
        user_prompt_2 += "Reply with each group on one line, in the format \"group_name: relation1, relation2, ...\".\n"

        messages = [
            SystemMessage(content=system_prompt),
        ]

        messages.append(HumanMessage(content=user_prompt_1))
        response = self.model.invoke(messages).content.strip()

        messages.append(AIMessage(content=response))
        messages.append(HumanMessage(content=user_prompt_2))
        response = self.model.invoke(messages).content.strip()

        mapping = {}
        for line in response.split('\n'):
            if ':' in line:
                group_name, relations = line.split(':', 1)
                group_name = group_name.strip()
                relations = [r.strip() for r in relations.split(',')]
                for relation in relations:
                    mapping[relation] = group_name
            else:
                # If the line does not contain a colon, it might be a single relation
                mapping[line.strip()] = line.strip()

        return mapping
