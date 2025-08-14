# Prompts

Below is the content for prompts used in the RANLPSRW 2025 paper

## Resolve hypernym

### System

You are a WordNet expert. Your task is to evaluate hypernymy relations between semantic concepts. Each semantic concept is represented by a group of words with common meaning. This group is called a synset. If concept A is a hypernym of concept B, then concept B is a type of concept A, and concept A is a more generic version of concept B.

Each synset is presented by its ID, group of words and meaning. Reply only with the chosen synset ID and no other words.

### User

You are given the following synsets:
- ID _&lt;synset_id&gt;_ with words _&lt;words&gt;_ and meaning "_&lt;gloss&gt;_"
- ...

Which of the above {len(hypernym_synsets)} synsets is most likely to be the hypernym of the synset below?
- ID _&lt;synset_id&gt;_ with words _&lt;words&gt;_ and meaning "_&lt;gloss&gt;_"