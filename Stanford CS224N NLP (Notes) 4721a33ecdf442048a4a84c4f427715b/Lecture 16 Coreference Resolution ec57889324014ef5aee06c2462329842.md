# Lecture 16: Coreference Resolution

## What is Coreference Resolution?

Identify all mentions that refer to the same real world entity.

### Applications

- Full text understanding
    - information extraction, summarization, question answering
- Machine translation
    - languages have different features for gender, number, dropped pronouns, etc.
- Dialogue systems

### Coreference Resolution in Two Steps

1. Detect the mentions (easy)
2. Cluster the mentions (hard)

## Mention Detection

Mention: span of text referring to some entity

Three kinds of mentions:

1. Pronouns
2. Named entities
3. Noun phrases

For detection: use other NLP systems

1. Pronouns: use a part-of-speech tagger
2. Named entities: Use a NER system (like hw3)
3. Noun phrases: Use a parser (especially a constituency parser)

### Not So Simple

Marking all pronouns, named entities, and NPs as mentions over-generated mentions

**How to deal with these bad mentions?**

Could train a classifier to filter out spurious mentions, but much more common is to keep all mentions as "candidate mentions"

- After your coreference system is done running discard all singleton mentions (i.e., ones that have not been marked as coreference with anything else)

**Can we avoid a pipelined system?**

We could instead train a classifier specifically for mention detection instead of using a POS tagger, NER system, and parser. Or, even jointly do mention-detection and coreference resolution end-to-end instead of in two steps.

## Linguistics of Coreference

**Coreference** is when two mentions refer to the same entity in the world

A related linguistic concept is **anaphora**: when a term (anaphor) refers to another term (antecedent)

- The interpretation of the anaphor is in some way determined by the interpretation of the antecedent

### Not all anaphoric relations are coreferential

Not all noun phrases have reference

*Every dancer* twisted her knee.

*No dancer* twisted her knee.

There are three NPs in each of these sentences; because the first one is non-referential, the other two aren't either.

### Anaphora vs. Coreference

Not all anaphoric relations are corefential.

`We want to see *a concert* last night. *The tickets* were really expensive.`

This is referred to as a bridging anaphora.

### Anaphora vs. Cataphora

Usually the antecedent comes before the anaphor (e.g., a pronoun), but not always.

## Four Kinds of Coreference Models

1. Rule-based (pronominal anaphora resolution)
2. Mention Pair
3. Mention Ranking
4. Clustering

# Additional Resources

[https://www.sciencedirect.com/science/article/pii/S1566253519303677](https://www.sciencedirect.com/science/article/pii/S1566253519303677)

[https://www.sciencedirect.com/science/article/pii/S153204641100133X](https://www.sciencedirect.com/science/article/pii/S153204641100133X)

[https://nlp.stanford.edu/projects/coref.shtml](https://nlp.stanford.edu/projects/coref.shtml)

[https://en.wikipedia.org/wiki/Coreference](https://en.wikipedia.org/wiki/Coreference)

[http://nlpprogress.com/english/coreference_resolution.html](http://nlpprogress.com/english/coreference_resolution.html)