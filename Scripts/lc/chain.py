import time
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Get data from WN JSON
with open("../Data/wn-3.1-json/noun.json", "r") as json_file:
    synsets = json.load(json_file)

# Chain architecture
# 1. Input - Synset ID
# 2. Retrieval of the hypernym tree, direct holonyms and attributes
# 3. Formatting the information for the retrieved synsets
# 4. Formatting with a system prompt, rules, instructions, and response format
# 5. Model inference
# 6. Output

# The expected output from the model is in JSON format or a table, clearly showing the old and new relations.
# Additionally, reasoning for the changes is provided in 3-5 sentences.

from langchain_ollama import ChatOllama, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Дефиниране на шаблона за заявка към големия езиков модел
prompt = PromptTemplate(
    template="""You are a WordNet analyst.
    You are given a synset ID and its relations.
    Your task is to analyze the synset words, gloss and relations,
    as well as the same information for the related synsets under "Data",
    and to perform the given task.
    Instructions:
    {instructions}
    Rules:
    {rules}
    WordNet data:
    {wn_data}
    Output format:
    {output_format}
    
    Response:
    """,
    input_variables=["instructions", "rules", "wn_data", "output_format"],
)

llm = OllamaLLM(
    model="llama3.2",
    temperature=0,
)

# Създаване на верига, която комбинира шаблона за заявка и големия езиков модел
chain = prompt | llm | StrOutputParser()

# Test the chain with a sample synset ID
synset_id = "00430140-n"  # Example synset ID
hypernym_synset_ids = {synset_id}  # Set of all hypernym synset IDs from the starting synset to the root (00001740-n)
other_synset_ids = set()  # Set of synset IDs from other relations

# Iterate through the expanding hypernym_synset_ids
