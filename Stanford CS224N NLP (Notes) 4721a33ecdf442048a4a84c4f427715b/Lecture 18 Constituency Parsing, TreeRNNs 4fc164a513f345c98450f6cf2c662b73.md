# Lecture 18: Constituency Parsing, TreeRNNs

**Semantic interpretation of language – not just word vectors:**

How can we work out the meaning of larger phrases? People interpret the meaning of larger text units – entities, descriptive terms, facts, arguments, stories – by semantic composition of smaller elements.

**Principle of compositionality**: 

By knowing the meaning of components (phrases/words within a sentence) and putting them together, a person can understand a sentence, regardless of length (could be infinitely long).

Language understanding in artificial intelligence requires being able to understand bigger things from knowing about smaller parts.

### Are languages recursive?

The structure of human language sentences have constituents that form hierarchically or recursively as you go further up in the language tree.

Recursion is natural for describing language (noun phrase containing a noun phrase containing a noun phrase). Recursion isn't the best formal definition for how language is structured, as there is a hard limit/bound placed on language given the sentences/phrases, whereas recursion technically expands to infinity.

## Building on Word Vector Space Models

We can represent the meaning of longer phrases by mapping them into the same vector space. This can be done through the use of the principle of compositionality:

The meaning (vector) of a sentence is determined by:

1. the meanings of its words and
2. the rules that combine them.

### Recursive vs. Recurrent Neural Networks

Recursive neural networks require a tree structure.

Recurrent neural nets cannot capture phrases without prefix context and often capture too much of the last words in the final vector.

### Recursive Neural Networks for Structure Prediction

Inputs: two candidate children's representations

Outputs:

1. The semantic representation if the two nodes are merged
2. Score of how plausible the new node would be

### Scene Parsing

Similar principle of compositionality:

- The meaning of a scene image is also a function of smaller regions,
- how they combine as parts to form larger objects,
- and how the objects interact.

### Capturing operators

In a traditional/simple recursive neural network, simply concatenating the children matrices and multiplying it by a weight is not powerful enough to capture operators (such as "very" in very good). This can be achieved through a new composition function, which will allow the phrase to be appropriately weighted (in the negative/positive direction) based on the operator, by tying both a matrix and a vector to every word. This is called a "matrix-vector" RNN.

# Additional Resources

[http://nlpprogress.com/english/constituency_parsing.html](http://nlpprogress.com/english/constituency_parsing.html)

[https://web.stanford.edu/~jurafsky/slp3/13.pdf](https://web.stanford.edu/~jurafsky/slp3/13.pdf)

[https://www.baeldung.com/cs/constituency-vs-dependency-parsing](https://www.baeldung.com/cs/constituency-vs-dependency-parsing)