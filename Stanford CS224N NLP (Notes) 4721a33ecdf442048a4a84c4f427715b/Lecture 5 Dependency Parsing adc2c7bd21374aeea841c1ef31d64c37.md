# Lecture 5: Dependency Parsing

## Two views of linguistic structure: Constituency = phrase structure grammar = context-free grammars (CFGs)

**Phrase structure** organizes words into nested constituents

1. **Starting unit: words**
    
    the, cat, cuddly, by, door
    
2. **Words combine into phrases**
    
    the cuddly cat, by the door
    
3. **Phrases can combine into bigger phrases**
    
    the cuddle cat by the door
    

Dependency structure shows which words depend on (modify or are arguments of) which other words. A dependency structure often contains a *root* node that explicitly marks the root of the tree, which is the head of the entire structure

**Why do we need sentence structure?**

We need to understand sentence structure in order to be able to interpret language correctly

Humans communicate complex ideas by composing words together into bigger units to convey complex meanings

We need to know what is connected to what

**Prepositional phrase attachment ambiguity:** Depending on how a sentence or phrase is constructed, this sentence could be interpreted in two different ways – the prepositional phrase can go with anything preceding it (verbs, objects, subjects, etc.) as a modifier.

As prepositional phrases follow each other, these ambiguities increase. A key parsing decision is how we "attach" various constituents.

While the number of possible attachments by an increasing number of prepositional phrases may seem like a factorial, it's actually a sequence called the Catalan numbers:

`C_n = (2n)!/[(n+1)!n!]`

This is still an exponential series, which arises in many tree-like contexts.

**Coordination scope ambiguity:** In some phrases, it can be tough to determine what structures the dependency is referring to or what a phrase is assigning itself. This occurs often when information is in list notation or connected with `and`

**Adjective modifier ambiguity**

**Verb Phrase (VP) attachment ambiguity**

## Dependency Grammar and Treebanks

Dependency syntax postulates that syntactic structure consists of relations between lexical items, normally binary asymmetric relations ("arrows") called **dependencies**. The arrows are commonly **typed** with the name of grammatical relations (subject, prepositional object, apposition, etc.)

The arrow connects a head with a dependent. Usually, dependencies form a tree (connected, acyclic, single-head).

**Universal Dependencies treebanks** are tree structures that provide manually checked annotated data for grammatical structure within languages.

### The rise of annotate data

Starting off, building a treebank seems a lot slower and less useful than building a grammar. But a treebank gives us many things:

- Reusability of the labor
    - Many parsers, part-of-speech taggers, etc., can be built on it
    - Valuable resource for linguistics
- Broad coverage, not just a few intuitions
- Frequencies and distributional information
- A way to evaluate systems

Treebanks are highly useful in machine learning as they provide information on what the right structure is for a given context, when provided a sentence or corpus.

### Dependency Condition Preferences

What are the sources of information for dependency parsing?

1. Bilexical affinities (Discussion → issues, rather than discussion → completed, is plausible)
2. Dependency distances (Mostly with nearby words)
3. Intervening material (Dependencies rarely span intervening verbs or punctuation)
4. Valency of heads (How many dependents on which side are usual for a head?)

When we build a dependency parser, we effectively want to say each word is going to be the dependency of some root (or some other word). From this, we want to build a tree structure by connecting roots to dependencies, which create nodes in our tree.

A sentence is parsed by choosing for each word what other word (including ROOT) is it a dependent of. Usually there are some constraints:

- Only one word is a dependent of ROOT
- Don't want cycles A → B, B → A
- This makes the dependencies a tree
- Final issue is whether arrows can cross (non-projective) or not

## Transition-based dependency parsing (Greedy)

Also known as deterministic dependency parsing (Google uses this!)

This is a simple form of greedy discriminative dependency parser

The parser does a sequence of bottom up actions

- Roughy like "shift" or "reduce" in a shift-reduce parser, but the "reduce" actions are specialized to create dependencies with head on left or right

## Neural dependency parsing

Neural networks can accurately determine the structure of sentences, supporting interpretation

The dense representations let it outperform other greedy parsers in both accuracy and speed