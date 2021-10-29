# Lecture 13: Contextual Word Embeddings

## Representations for a word

Up until now, we've basically said that we have one representation of words: the word vectors that were learned about at the beginning (word2vec, GloVe, fastText)

Unsupervised pre-training followed by a supervised NN improved/maintained performance as state-of-the-art models.

### Pre-trained word vectors

We can just start with random word vectors and train them on our task of interest – but in most cases, use of pre-trained word vectors helps, because we can train them for more words on much more data

### Tips for unknown words with word vectors

Simplest and common solution:

Use <UNK> when out-of-vocabulary (OOV) words occur. But this poses problems – there is no way to distinguish different UNK words, either for identity or meaning

Solution:

Use char-level models to build vectors

### Representations for a word

The initially learned word vector representations (word2vec, GloVe, fastText) have two problems:

- Always the same representation for a word type regardless of the context in which a word token occurs; we might want very fine-grained word sense disambiguation
- We just have one representation for a word, but words have different aspects, including semantics, syntactic behavior, and register/connotations

## Named Entity Recognition (NER)

A very important NLP sub-task: find and classify (person, date, location, organization, etc.) names in text, for example

## TagLM - "Pre-ELMo"

Idea: Want meaning of word in context, but standardly learn task RNN only on small task-labeled data (e.g., NER)

Why don't we do semi-supervised approach where we train NLN on large unlabeled corpus, rather than just word vectors?

Language model is trained on 800 million training words of "Billion word benchmark"

Language model observations:

- An LM trained on supervised data does not help
- Having a bidirectional LM helps over only forward, by about 0.2
- Having a huge LM design helpers over a smaller model by about 0.3

Task-specific BiLSTM observations:

- Using just the LM embeddings to predict isn't great; the metric is well below just using a BiLSTM tagger on labeled data

## ELMo: Embeddings from Language Models

Train a bidirectional LM, and aim at performant but not overly large LM:

- Use 2 BiLSTM layers
- Use character CNN to build initial word representation (only)
- Use 4096 dimension hidden/cell LSTM states with 512 dimension projects to next input
- Use a residual connection
- Tie parameters of token input and output (softmax) and tie these between forward and backward LMs

ELMo learns task-specific combination of BiLM representations. This is an innovation that improves on just using the top layer of the LSTM stack.

### ELMO: Weighting of layers

The two BiLSTM NLM layers have differentiated uses/meanings

- Lower layer is better for lower-level syntax, etc.
    - Part-of-speech tagging, syntactic dependencies, NER
- Higher layer is better for higher-level semantics
    - Sentiment, semantic role labeling, question answering, SNLI

## The Motivation for Transformers

We want parallelization but RNNs are inherently sequential

Despite GRUs and LSTMs, RNNs still need attention mechanism to deal with long range dependencies – path length between states grows with sequence otherwise

But if attention gives us access to any state, maybe we can just use attention and don't need the RNN!

## Transformer Overview

The basic building blocks of transformer networks: new attention layers.

The goal is to make a complex encoder and decoder while removing the recurrent dependency, but is able to translate sentences well by using lots of attention distributions

The basic idea is to use attention everywhere to compute things. The simplest kind of attention is used, where dot products between key-value pairs are calculated.

### Dot-Product Attention

Inputs: a query q and a set of key-value (k-v) pairs to an output

Query, keys, values, and output are all vectors.

The output is a weighted sum of values, where the weight of each value is computed by an inner product of query and corresponding key.

### Self-attention in the encoder

Inside of the encoder, everything is a vector; the queries, keys, and values.

### Multi-head attention

Problem with simple self-attention:

- Only one way for words to interact with one-another
- Solution: multi-head attention; with this, you have your hidden state projections mapped through lower dimensional space matrices

### Complete transform block

Each block has two "sublayers":

1. Multihead attention
2. 2-layer feed-forward NNet (with ReLU)

Each of these two steps also has:

- Residual (short-circuit) connection and LayerNorm (normalization)

### Encoder Input

Actual word representations are byte-pair encodings; a positional encoding is added so the same words at different locations (position in sentence, i.e., beginning or end) have different overall representations

### Complete Encoder

For each encoder, at each block, we use the same Q, K, and V from the previous layer

Blocks are repeated 6 times in a vertical stack

### Transformer Decoder

2 sublayer changes in decoder

Mashed decoder self-attention on previously generated outputs

Encoder-Decoder Attention, where queries come from previous decoder layer and keys and values come from output of encoder

Blocks are also repeated 6 times

## BERT

BERT (Bidirectional Encoder Representations from Transformers)

Problem: Language models only use left context *or* right context, but language understanding is bidirectional

Why are LMs unidirectional?

1. Directionality is needed to generate a well-formed probability distribution
2. Words can "see themselves" in a bidirectional encoder

Solution: Mask out k% of the input words, and then predict the masked words

Too little masking: too expensive to train

Too much masking: not enough context

### BERT complication: next sentence prediction

To learn relationships between sentences, predict whether sentence B is an actual sentence that proceeds sentence A, or a random sentence

# Additional Resources

[http://nlp.seas.harvard.edu/2018/04/03/attention.html](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)

[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)