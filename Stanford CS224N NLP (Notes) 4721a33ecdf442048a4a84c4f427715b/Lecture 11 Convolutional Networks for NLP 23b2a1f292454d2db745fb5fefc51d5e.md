# Lecture 11: Convolutional Networks for NLP

## From RNNs to Convolutional Neural Nets

Recurrent neural nets cannot capture phrases without prefix context. They often capture too much of the last words in the final vector (e.g.., softmax is often only calculated at the last step).

Main CNN/ConvNet idea:

What if we compute vectors for every possible word subsequence of a certain length?

Example: "tentative deal reached to keep government open" computes vectors for:

tentative deal reached, deal reached to, reached to keep, to keep government open, keep government open

Regardless of whether the phrase is grammatical, and it's not very linguistically or cognitively plausible. We group these afterwards.

### What is a convolution?

Convolution is classically used to extract features from images.

## Model comparison

**Bag of Vectors:** Surprisingly good baseline for simple classification problems, especially if followed by a few ReLU layers.

**Window Model:** Good for single word classification for problems that do not need wide context: e.g., POS, NER

**CNNs:** good for classification, need zero padding for shorter phrases, hard to interpret, easy to parallelize on GPUs. Efficient and versatile

**Recurrent Neural Networks:** Cognitively plausible (reading from left to right), not best for classification (if use just last state), much slower than CNNs, good for sequence tagging and classification, great for language models, can be amazing with attention mechanisms

### Gated units used vertically

The gating/skipping that we saw in LSTMs and GRUs is a general idea, which is now used in a whole bunch of places. You can also gate vertically. Summing candidate update with shortcut connections is needed for very deep networks to work.

### Batch Normalization (BatchNorm)

Often used in CNNs, transforms the convolution output a batch by scaling the activations to have zero mean and unit variance. This is the familiar Z-transformation of statistics, but is updated per batch so fluctuation doesn't affect things too much.

The use of BatchNorm makes model *much* less sensitive to parameter initialization, since outputs are automatically rescaled. It also tends to make tuning of learning rates simpler.

### 1x1 Convolutions (1-convolutions)

1x1 convolutions, aka. Network-in-Network (NiN) connections, are convolutional kernels with kernel_size=1.

A 1x1 convolution gives you a fully connected linear layer across channels; it can be used to map from many channels to fewer channels.

1x1 convolutions add additional neural network layers with very few additional parameters, unlike Fully Connected (FC) layers which add a lot of parameters.

## Very Deep Convolutional Networks for Text Classification

Starting point: sequence models (LSTMs) have been very dominant in NLP; also CNNs, Attention, etc., but all the models are basically not very deep – not like the deep models in Vision.

What happens when we build a vision-like system for NLP, working from the character level?

## RNNs are slow

RNNs are a very standard building block for deep NLP, but they parallelize badly and are slow.

Idea: Take the best and parallelizable parts of RNNs and CNNs → **Quasi-Recurrent Neural Network (Q-RNN)**, which tries to combine the best of both model families.

### Q-RNNs for Sentiment Analysis

Often better and faster than LSTMs

More interpretable

### Q-RNN limitations

Didn't work for character-level LMs as well as LSTMs

- Trouble modeling much longer dependencies?

Often need deeper network to get as good performance as LSTM

- They're still faster when deeper
- Effectively they use depth as a substitute for true recurrence

### Problems with RNNs and Motivation for Transformers

We want parallelization but RNNs are inherently sequential

Despite GRUs and LSTMs, RNNs still gain from attention mechanism to deal with long range dependencies – path length between states grows with sequence otherwise.

# Additional Resources

[https://cs231n.github.io/convolutional-networks/](https://cs231n.github.io/convolutional-networks/)

[https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)