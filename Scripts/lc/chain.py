import json
import time

from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


class WordNetHypernymResolver:
    def __init__(self, model):
        self.wn_data_prompt = PromptTemplate(
            template_format="jinja2",
            template="""{% for synset in wn_data -%}
## Data for {{ synset.id }}
Words for {{ synset.id }}: {% for word in synset.words %}{{ word.word }}{% if not loop.last %}, {% endif %}{% endfor %}
Gloss (meaning) for {{ synset.id }}: {{ synset.gloss }}
{%- if synset.hypernyms is defined %}{% for relation in synset.hypernyms %}
{{ synset.id }} has a hypernym relation to {{ relation.id }}.
I.e. {{ relation.id }} is a hypernym of {{ synset.id }} and {{ synset.id }} is a hyponym of {{ relation.id }}.
{%- endfor %}{% endif %}{# synset.hypernyms #}
{%- if synset.holonyms is defined %}{% for relation in synset.holonyms %}
{{ synset.id }} has a {{ relation.type }} holonym relation to {{ relation.id }}.
I.e. {{ relation.id }} is a {{ relation.type }} holonym of {{ synset.id }} and {{ synset.id }} is a {{ relation.type }} meronym of {{ relation.id }}.
{%- endfor %}{% endif %}{# synset.holonyms #}
{%- if synset.meronyms is defined %}{% for relation in synset.meronyms %}
{{ synset.id }} has a {{ relation.type }} meronym relation to {{ relation.id }}.
I.e. {{ relation.id }} is a {{ relation.type }} meronym of {{ synset.id }} and {{ synset.id }} is a {{ relation.type }} holonym of {{ relation.id }}.
{%- endfor %}{% endif %}{# synset.meronyms #}
{%- if synset.domains is defined %}{% for relation in synset.domains %}
{{ synset.id }} has a {{ relation.type }} domain relation to {{ relation.id }}.
I.e. {{ relation.id }} is a {{ relation.type }} domain of {{ synset.id }} and {{ synset.id }} is a member of the {{ relation.type }} domain {{ relation.id }}.
{%- endfor %}{% endif %}{# synset.domains #}
{%- if synset.domain_members is defined %}{% for relation in synset.domain_members %}
{{ synset.id }} has a {{ relation.type }} domain member relation to {{ relation.id }}.
I.e. {{ relation.id }} is a member of the {{ relation.type }} domain {{ synset.id }} and {{ synset.id }} is a {{ relation.type }} domain of {{ relation.id }}.
{%- endfor %}{% endif %}{# synset.domain_members #}
{%- if synset.other_relations is defined %}{% for relation in synset.other_relations %}
{%- if relation.type == "antonym" -%}
{{ synset.id }} has an antonym relation to {{ relation.id }}.
I.e. {{ relation.id }} is an antonym of {{ synset.id }} and {{ synset.id }} is an antonym of {{ relation.id }}.
{%- elif relation.type == "attribute" -%}
{{ synset.id }} has an attribute relation to {{ relation.id }}.
I.e. {{ relation.id }} is an attribute of {{ synset.id }} and {{ synset.id }} has the attribute {{ relation.id }}.
{%- endif %}{# relation.type #}
{%- endfor %}{% endif %}{# synset.other_relations #}
{% endfor %}{# wn_data #}
""",
            input_variables=["wn_data"],
        )
        self.main_prompt = PromptTemplate(
            template="""You are a WordNet engineer.
----------------------------------------------------------------
Instructions:
You are given a list of synsets with their full or partial data,
and a synset ID for which to perform the task.
Your job is to analyze the words, glosses and relations of the
given synsets under "WordNet data" and to execute the task.
Your task is to resolve the multiple hypernyms of the task synset ID.
You can do this by adding, removing or changing the relations
of the task synset as described under "Rules". The resulting set of
relations must include exactly one hypernym relation for the task synset.
All synsets are referenced by their synset ID. All synsets are nouns.
----------------------------------------------------------------
Rules:
The given task synset will have a list of relations of the following types:
- hypernym - regular or instance
- hyponym - regular or instance
- holonym - part, substance or member
- meronym - part, substance or member
- domain - topic, region or usage
- domain member - topic, region or usage
- attribute
- antonym (only for reference)
You can add, remove or change the relations of the task synset as follows:
- Add a new hypernym relation
- Remove an existing hypernym relation
- Change an existing hypernym relation to any other relation, except for antonym
- Change any other relation to a hypernym relation, except for antonym
You cannot delete or make further changes to already changed relations.
All relations must be between the task synset and other given synsets.
----------------------------------------------------------------
Output format:
Format the result in a table with the following columns:
- Old relation type
- New relation type
- Synset ID
- Synset words
- Synset gloss
First row of the table must be the task synset ID. Leave the relation
type columns empty for the task synset ID.
All other rows must list synsets to which the task synset's relation was:
- Added
- Removed
- Changed
- Kept
After the table, provide a short reasoning for the changes.
----------------------------------------------------------------
Task synset ID:
{synset_id}
----------------------------------------------------------------
WordNet data:
{wn_data}
----------------------------------------------------------------
Response:
""",
            input_variables=["synset_id", "wn_data"],
        )
        self.llm = OllamaLLM(
            model=model,
            temperature=0,
        )
        self.chain = (
            self.main_prompt |
            self.llm |
            StrOutputParser()
        )

    def load_synsets(self):
        with open(self.json_path, "r") as json_file:
            return json.load(json_file)

    def run(self, synset_id, wn_data):
        wn_data_text = self.wn_data_prompt.format(
            wn_data=wn_data
        )
        return self.main_prompt.format(
            synset_id=synset_id,
            wn_data=wn_data_text
        )
        # return self.chain.invoke({
        #     "instructions": instructions,
        #     "rules": rules,
        #     "wn_data": wn_data,
        #     "output_format": output_format,
        # })


# Chain architecture
# 1. Input - Synset ID
# 2. Retrieval of the hypernym tree, direct holonyms and attributes
# 3. Formatting the information for the retrieved synsets
# 4. Formatting with a system prompt, rules, instructions, and response format
# 5. Model inference
# 6. Output

# The expected output from the model is in JSON format or a table, clearly showing the old and new relations.
# Additionally, reasoning for the changes is provided in 3-5 sentences.
