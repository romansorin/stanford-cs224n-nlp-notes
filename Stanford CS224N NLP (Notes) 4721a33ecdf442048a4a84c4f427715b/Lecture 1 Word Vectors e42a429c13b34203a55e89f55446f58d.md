# Lecture 1: Word Vectors

Commonest linguistic way of thinking of meaning:

`[ signifier (symbol) <> signified (idea or thing) ]` → **derivational semantics**

A common solution to achieve "usable" meaning is through the use of synonym sets and hypernyms **("is a" relationships)*.*

[NLTK](https://www.nltk.org/) Is like the "swiss army knife" of NLP – not terribly good for anything but has a lot of basic tools

### Representing words as different symbols:

In traditional NLPs, we regard words as discrete symbols: hotel, conference, motel – a *localist* representation. Words can be represented by [one-hot vectors](https://en.wikipedia.org/wiki/One-hot).

**Derivational morphology:** creating new words by adding endings onto existing words

With this, we end up with very big vectors if we want to represent a sensible size vocabulary. If we're trying to search "Seattle hotel", we'd also want to match "Seattle motel" – but the two vectors are orthogonal and there is no natural notion of **similarity** for one-hot vectors.

Vector dimension: number of words in vocabulary

### Representing words by their context:

**Distributional semantics**: a word's meaning is given by the words that frequently appear close-by

When a word *w* appears in a text, the context is the set of words that appear nearby (within a fixed-size window). Use the many contexts of *w* to build up a representation of *w.*

## Word Vectors

Build a dense* vector for each word, chosen so that it is similar to vectors of words that appear in similar contexts (*All numbers are non-zero) 

Note: word vectors are sometimes called word embeddings or word representations. They are a *distributed* representation.

Because words can be represented as vectors, they can be grouped together and placed in a vector space.

Learning algorithms decide the dimensions, but they are still somewhat arbitrary.

A large pile of text is a "corpus" (corpora, plural)

**word2vec** is a neural net that vectorizes words, and predicts context words based on the center (or provided) word through these vectorizations.

### Training a model by optimizing parameters

To train a model, we adjust parameters to minimize a loss (i.e., minimize our objective function <> maximize prediction accuracy). When viewing graphs of our vector space, contour lines show different levels of the objective function (and subsequent values of loss).

## Additional Resources

[Introduction to Word Vectors - DZone AI](https://dzone.com/articles/introduction-to-word-vectors)

[https://wiki.pathmind.com/word2vec](https://wiki.pathmind.com/word2vec)