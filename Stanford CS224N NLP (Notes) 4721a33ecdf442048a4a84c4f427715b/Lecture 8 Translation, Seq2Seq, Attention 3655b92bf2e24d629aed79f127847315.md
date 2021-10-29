# Lecture 8: Translation, Seq2Seq, Attention

## Machine Translation

**Machine Translation (MT)** is the task of translating a sentence x from one language (the source language) to a sentence y in another language (the target language).

x: доброе утро, как спалось?

y: Good morning, how did you sleep?

### 1950s: Early Machine Translation

Russian → English (motivated by the Cold War)

Systems were mostly rule-based, using a bilingual dictionary to map Russian words to their English counterparts

### 1990s-2010s: Statistical Machine Translation

Core idea: Learn a probabilistic model from data

Suppose we're translating French → Englilsh

We want to find the best English sentence y, given French sentence x: 

argmax_yP(y|x)

Use Bayes Rule to break this down into two components to be learnt separately: 

argmax_yP(x|y)P(y)

Translation Model (P(x|y)): Models how words and phrases should be translated (fidelity). Learned from parallel data.

Language Model (P(y)): Models how to write good English (fluency). Learnt from monolingual data.

Instead of using a single probability distribution to take care of all components and tasks related to translation (grammar, sentence structure, translation), the idea is to break this down into separate distributions to better assist.

Question: How to learn translation model P(x|y)?

First, need a large amount of parallel data (e.g., pairs of human-translated French/English sentences). An example of this is The Rosetta Stone.

SMT was a huge research field

The best systems were extremely complex

- Hundreds of important details that aren't mentioned here
- Systems had many separately-designed subcomponents
- Lots of feature engineering
    - Need to design features to capture particular language phenomena
- Require compiling and maintaining extra resources
    - Like tables of equivalent phrases
- Lots of human effort to maintain
    - Repeated effort for each language pair

### Learning alignment for SMT

Question: How to learn translation model P(x|y) from the parallel corpus?

Break it down further: we actually want to consider:

P(x, a|y)

where a is the alignment, i.e. word-level correspondence between French sentence x and English sentence y

Question: How to compute the argmax?

We could enumerate every possible y and calculate the probability, but that's too expensive.

Answer: Use a heuristic search algorithm to search for the best translation, discarding hypotheses that are too low-probability

### What is alignment?

**Alignment** is the correspondence between particular words in the translated sentence pair.

Note: Some words have no counterpart or direct translation.

Alignment can be many-to-one. Several x words may map to a single y word.

A one-to-many word is called a **fertile** word as it has many children in the target sentence.

You can also have many-to-many alignments, which is also known as phrase level translation or phrase-to-phrase.

We learn P(x, a|y) as a combination of many factors, including:

- Probability of particular words aligning (also depends on position in sentence)
- Probability of particular words having particular fertility (number of corresponding words)
- etc.

## Neural Machine Translation

**Neural Machine Translation (NMT)** is a way to do Machine Translation with a single neural network

The neural network architecture is called sequence-to-sequence (aka **seq2seq**) and it involves two RNNs.

### Sequence-to-sequence is versatile

Sequence-to-sequence is useful for more than just MT

Many NLP tasks can be phrased as sequence-to-sequence:

- Summarization (long text → short text)
- Dialogue (previous utterances → next utterance)
- Parsing (input text → output parse as sequence)
- Code generation (natural language → Python code)

### Neural Machine Translation (NMT)

The sequence-to-sequence model is an example of a **Conditional Language Model.**

- Language Model because the decoder is predicting the next word the target sentence y
- Conditional because its predictions are also conditioned on the source sentence x

Question: How to train an NMT system?

Answer: Get a big parallel corpus...

### Training a Neural Machine Translation system

Feed the source sentence from the corpus into the encoder RNN, and the target sentence from the corpus into the decoder RNN. For every step of the decoder RNN, you're going to produce the probability distribution of every next word. 

Seq2seq is optimized as a single system. Backpropagation operates "end-to-end".

### Greedy decoding

Generating (decoding) the target sentence by taking argmax on each step of the decoder.

Problems with this method:

- Greedy decoding has no way to undo decisions

How do we fix this?

### Beam search decoding

Core idea: On each step of decoder, keep track of the k most probable partial translations (which we call hypotheses)

- k is the beam size (in practice around 5 to 10); how big is your search space at any point in time

Beam search is not guaranteed to find the optimal solution. While exhaustive search, which will iterate through all of the possibilities will, it is obviously infeasible because it is too expensive. Beam search is simply more efficient than exhaustive search.

### Advantages of NMT

Compared to SMT, NMT has many advantages:

- Better performance
    - More fluent
    - Better use of context
    - Better use of phrase similarities
- A single neural network to be optimized end-to-end
    - No subcomponents to be individually optimized
- Requires much less human engineering effort
    - No feature engineering
    - Same method for all language pairs

### Disadvantages of NMT

Compared to SMT:

- NMT is less interpretable
    - Hard to debug
- NMT is difficult to control
    - For example, can't easily specify rules or guidelines for translation
    - Safety concerns

## How do we evaluate Machine Translation?

**BLEU (Bilingual Evaluation Understudy)**

BLEU compares the machine-written translation to one or several human-written translation(s), and computes a similarity score based on:

- n-gram precision (usually for 1, 2, 3, and 4-grams)
- Plus a penalty for too-short system translations

BLEU is useful but imperfect 

- There are many valid ways to translate a sentence
- A good translation can get a poor BLEU score because it has a low n-gram overlap with the human translation

## Attention

An informational bottleneck places too much pressure on a single vector to capture all information about the source sentence in its encoding. If information about the source sentence is not given to the vector, there's no way for the vector to know about the rest of information.

**Attention** provides a solution to the bottleneck problem.

Core idea: on each step of the decoder, use *direct connection to the encoder* to *focus on a particular part* of the source sequence

### Attention is great

Attention significantly improves NMT performance

- It's very useful to allow decoder to focus on certain parts of the source

Attention solves the bottleneck problem

- Attention allows decoder to look directly at source; bypass bottleneck

Attention helps with vanishing gradient problem

- Provides shortcut to faraway states

Attention provides some interpretability

- By inspecting attention distribution, we can see what the decoder was focusing on
- We get (soft) alignment for free
- We never explicitly trained an alignment system, yet the network learned alignment by itself

### Attention is a general Deep Learning technique

Attention is a great way to improve the sequence-to-sequence model for Machine Translation. However, you can use attention in many architectures (not just seq2seq) and many tasks (not just MT).

**More general definition of attention:**

Given a set of vector values, and a vector query, attention is a technique to compute a weighted sum of the values, dependent on the query.

We sometimes say that the query attends to the values. For example, in the seq2seq + attention model, each decoder hidden state (query) attends to all the encoder hidden states (values).

**Intuition:**

The weighted sum is a selective summary of the information contained in the values, where the query determines which values to focus on.

Attention is a way to obtain a fixed-size representation of an arbitrary set of representations (the values), dependent on some other representation (the query).

# Recap

Sequence-to-sequence is the architecture for NMT (uses 2 RNNs)

Attention is a way to focus on particular parts of the input; improves sequence-to-sequence a lot