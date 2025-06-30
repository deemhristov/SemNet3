import json
import sys
import time

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class BreakDownHypernymResolver:
    def __init__(self, model):
        # Initialize the model
        self.model = OllamaLLM(
            model=model,
            temperature=0.5,
        )

        # Augmentation prompt to extend the glosses with LLM knowledge
        self.augmentation_prompt = PromptTemplate(
            template_format="jinja2",
            template="""Give a common meaning which applies for all of the following list of words:
({% for word in words %}{{ word }}{% if not loop.last %}, {% endif %}{% endfor %})
Reply only with the result.""",
            input_variables=["words"],
        )

        # Prompt for identifying possible alternative relations for hypernyms
        self.rank_relations_for_hypernym_prompt = PromptTemplate(
            template_format="jinja2",
            template="""In terms of semantic relations between concepts, rank the following possible semantic relations from the concept ({{ gloss_a }}) to the concept ({{ gloss_b }}) in order from most likely to least likely: "holonym", "meronym", "domain", "domain member", "attribute", "no relation".
A relation R from concept A to concept B means "B is a R of A".
Reply with a comma-delimited list only, without bullets, numbers, or reasoning. Include all options in the list.""",
            input_variables=["gloss_a", "gloss_b"],
        )

        # Prompt for identifying possible hypernyms among non-hypernyms
        self.assess_non_hypernyms_prompt = PromptTemplate(
            template_format="jinja2",
            template="""State whether concept ({{ gloss_b }}) is likely to be a hypernym of the concept ({{ gloss_a }}).
Reply with yes/no without any explanation.""",
            input_variables=["gloss_a", "gloss_b"],
        )

        # Prompt for ranking all possible hypernyms to identify the most likely one
        self.rank_all_possible_hypernyms_prompt = PromptTemplate(
            template_format="jinja2",
            template="""You are given the concept ({{ main_gloss }}). Rank the following concepts from most likely to least likely to be a hypernym of the previously given concept:
{% for gloss in glosses %}{{ loop.index }}. ({{ gloss }})
{% endfor %}
The hypernymy relation from concept A to concept be entails that the statements "A is a B" and "A is a type of B" are true. Reply with with a comma-delimited list of the concept numbers. Include all listed concepts. Do not add any additional text before or after the list.""",
            input_variables=["main_gloss", "glosses"],
        )

    def run(self, main_synset, hypernym_synsets, other_synsets):
        # Get the glosses of the synsets
        main_gloss = main_synset["gloss"]
        hypernym_glosses = [synset["gloss"] for synset in hypernym_synsets]
        other_glosses = [synset["gloss"] for synset in other_synsets]

        # Get the words of the synsets
        main_words = [word["word"] for word in main_synset["words"]]
        hypernym_words = [[word["word"] for word in synset["words"]] for synset in hypernym_synsets]
        other_words = [[word["word"] for word in synset["words"]] for synset in other_synsets]

        # Get the common meaning of the words

        # Main synset
        common_meaning_prompt = self.augmentation_prompt.format(words=main_words)
        common_meaning_response = self.model.invoke(common_meaning_prompt)
        main_gloss = main_gloss + " / " + common_meaning_response

        # Hypernyms
        for i, hypernym_gloss in enumerate(hypernym_glosses):
            common_meaning_prompt = self.augmentation_prompt.format(words=hypernym_words[i])
            common_meaning_response = self.model.invoke(common_meaning_prompt)
            hypernym_glosses[i] = hypernym_gloss + " / " + common_meaning_response

        # Other synsets
        for i, other_gloss in enumerate(other_glosses):
            common_meaning_prompt = self.augmentation_prompt.format(words=other_words[i])
            common_meaning_response = self.model.invoke(common_meaning_prompt)
            other_glosses[i] = other_gloss + " / " + common_meaning_response
            # print(other_gloss, "/", common_meaning_response, file=sys.stderr)

        # Identify alternative relations for hypernyms

        # Rank relations for hypernyms
        alternative_relations = []
        for hypernym_gloss in hypernym_glosses:
            relation_ranking_prompt = self.rank_relations_for_hypernym_prompt.format(gloss_a=main_gloss, gloss_b=hypernym_gloss)
            relation_ranking_response = self.model.invoke(relation_ranking_prompt)
            alternative_relations.append(relation_ranking_response.split(",")[0].strip().lower())  # Take the first relation as the most likely one

        # Assess non-hypernyms
        other_is_alternative_hypernym = []
        for other_gloss in other_glosses:
            non_hypernym_assessment_prompt = self.assess_non_hypernyms_prompt.format(gloss_a=main_gloss, gloss_b=other_gloss)
            non_hypernym_assessment_response = self.model.invoke(non_hypernym_assessment_prompt)
            other_is_alternative_hypernym.append(non_hypernym_assessment_response.strip().lower() == "yes")

        # Rank all possible hypernyms
        possible_hypernym_ids = [*[synset["id"] for synset in hypernym_synsets], *[synset["id"] for synset in other_synsets if other_is_alternative_hypernym[i]]]
        possible_hypernym_glosses = [main_gloss, *hypernym_glosses, *[synset["gloss"] for synset in other_synsets if other_is_alternative_hypernym[i]]]
        rank_hypernyms_prompt = self.rank_all_possible_hypernyms_prompt.format(main_gloss=main_gloss, glosses=possible_hypernym_glosses)
        rank_hypernyms_response = self.model.invoke(rank_hypernyms_prompt)

        first_hypernym_index = rank_hypernyms_response.split(",")[0].strip()
        if not first_hypernym_index.isdigit():
            raise ValueError(f"LLM has refused: '{rank_hypernyms_response}'")
        first_hypernym_index = int(first_hypernym_index) - 1
        first_hypernym_id = possible_hypernym_ids[first_hypernym_index]

        # print(rank_hypernyms_response, file=sys.stderr)
        # print(first_hypernym_index, file=sys.stderr)
        # print(possible_hypernym_ids, file=sys.stderr)
        # print(first_hypernym_id, file=sys.stderr)

        # JSON output with the format
        # [
        #     {
        #         "synset_id": <synset_id>,
        #         "words": <synset_words>,
        #         "gloss": <synset_gloss>,
        #         "old_type": <old_relation_type>,
        #         "new_type": <new_relation_type>
        #     },
        #     ...
        # ]
        # The first object in the list must be the task synset. Leave the old_type and
        # new_type properties empty for the task synset.

        result = []
        result.append({
            "synset_id": main_synset["id"],
            "words": main_words,
            "gloss": main_gloss,
            "old_type": "",
            "new_type": ""
        })
        for i, hypernym_gloss in enumerate(hypernym_glosses):
            # print(hypernym_synsets[i]["id"], first_hypernym_id, file=sys.stderr)
            result.append({
                "synset_id": hypernym_synsets[i]["id"],
                "words": hypernym_words[i],
                "gloss": hypernym_gloss,
                "old_type": "hypernym",
                "new_type": "hypernym" if hypernym_synsets[i]["id"] == first_hypernym_id else (alternative_relations[i] if alternative_relations[i] != "no relation" else "")
            })
        for i, other_gloss in enumerate(other_glosses):
            result.append({
                "synset_id": other_synsets[i]["id"],
                "words": other_words[i],
                "gloss": other_gloss,
                "old_type": other_synsets[i]["relation_type"],
                "new_type": "hypernym" if other_synsets[i]["id"] == first_hypernym_id else other_synsets[i]["relation_type"]
            })

        return result
