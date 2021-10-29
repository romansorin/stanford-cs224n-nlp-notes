# Lecture 2: Word Vectors and Senses

## Optimization: Gradient Descent

- We have a cost function J(θ) that we want to minimize
- Gradient Descent is an algorithm to minimize J(θ)
- Idea: for current value of θ, calculate gradient of J(θ), then take a small step in the direction of that negative gradient. Repeat until minimum.
- Note: Objectives will not always be convex or easy to step through.

## Stochastic Gradient Descent

**Problem:** J(θ) is a function of *all* windows in the corpus (potentially billions!) – so the gradient of this unction is very expensive to compute. You would have to wait a very long time before making a single update.

- Very bad idea for pretty much all neural nets

**Solution:** Stochastic gradient descent [(SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) – repeatedly sample windows and update after each one

## word2vec models + details:

1. Skip-grams (SG): predict context ("outside") words (position independent) given a center word
2. Continuous Bag of Words (CBOW): predict center word from (bag of) context words

### Skip-gram model with negative sampling

The normalization factor is too computationally expensive, so we implement the skip-gram model with negative sampling. The main idea is to train binary logistic regression for a true pair (center word and word in its context window) vs. several noise pairs (the center word paired with a random word).

- More details from the paper: *Distributed Representation of Words and Phrases and their Compositionality* (Mikolov et. al. 2013)

We want to maximize the probability that real outside words appears, and minimize the probability that random words appear around the center word.

**Hyperparameter:** a parameter whose value is used to control the learning process. Try a few different numbers and see which one works best.

## How to evaluate word vectors?

Related to general evaluation in NLP: Intrinsic vs. Extrinsic

### Intrinsic:

- evaluation on a specific/intermediate subtask
- fast to compute
- helps to understand that system
- not clear if really helpful unless correlation to real task is established

### Extrinsic:

- evaluation on a real task
- can take a long time to compute accuracy
- unclear if the subsystem is the problem or its interaction of other substems
- if replacing exactly one subsystem with another improves accuracy, that's ideal
- lots of variance

## Additional Resources

[https://en.wikipedia.org/wiki/Word_sense](https://en.wikipedia.org/wiki/Word_sense)

[http://deeplearning.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/](http://deeplearning.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/)