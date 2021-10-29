# Lecture 14: Transformers and Self-Attention

# Learning Representations of Variable Length Data

**Basic building block of sequence-to-sequence learning:** Neural machine translation, summarization, QA, etc.

### Recurrent Neural Networks

Model of choice for learning variable-length representations.

Natural fit for sentences and sequences of pixels.

LSTMs, GRUs, and variants dominate recurrent models.

*But:*

Sequential computation inhibits parallelization. ← recurrence, by design, depends on the previous output for the next input, and has to wait for the previous process to finish

No explicit modeling of long and short range dependencies.

We want to model hierarchy.

**RNNs (w/sequence-aligned states) seem wasteful!**

### Convolutional Neural Networks

Trivial to parallelize (per layer).

Exploits local dependencies

'Interaction distance' between positions linear or logarithmic. The number of layers required are still a function of the characters in your string.

**Long-distance dependencies require many layers.**

### Attention

Attention between encoder and decoder is crucial in NMT.

Why not use attention for representations?

### Self-Attention

Constant 'path length' between any two positions.

Gating/multiplicative interactions.

Trivial to parallelize (per layer).

Can it replace sequential computation entirely?

### Importance of Residuals

Residuals (residual connections between every pair of layers within the model) carry positional information to higher layers, among other information

### Probabilistic Image Generation

Model the joint distribution of pixels, turning it into a sequence modeling problem

**Assigning probabilities allows measuring generalization**

RNNs and CNNs are state-of-the-art

**CNNs incorporating gating now match RNNs in quality; CNNs are much faster due to parallelization**

Long-range dependencies matter for images (e.g., symmetry)

Likely increasingly important with increasing image size

Modeling long-range dependencies with CNNs requires either:

1. Many layers likely making training harder
2. Large kernels at large parameter/computational cost

### Self-Attention

Constant 'path length' between any two positions.

Unbounded memory.

Trivial to parallelize (per layer).

Models self-similarity.

Relative attention provides expressive timing, equivariance, and extends naturally to graphs