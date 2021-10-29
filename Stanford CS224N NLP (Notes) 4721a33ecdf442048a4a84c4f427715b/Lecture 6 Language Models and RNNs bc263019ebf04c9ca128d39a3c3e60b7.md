# Lecture 6: Language Models and RNNs

## Language Modeling

**Language Modeling** is the task of predicting what word comes next

ex. "the students opened their" → {books, laptops, exams, minds}

More formally: given a sequence of words

x^(1), x^(2), ... , x^(t),

compute the probability distribution of the next word x^(t+1):

*P*(x^(t+1)|x^(t), ... , x^(1))

where x^(t+1) can be any word in the fixed vocabulary *V* = {w_1, ... , w_|*V*|}

Language modeling can be thought of as a classification task, as there is a predefined set of possibilities.

You can also think of a Language Model as a system that assigns probability to a piece of text.

### n-gram Language Models

Question: How to learn a Language Model?

Answer: (pre-Deep learning): learn a **n-gram Language Model**

Definition: A **n-gram** is a chunk of *n* consecutive words.

- **uni**grams: "the", "students", "opened", "their"
- **bi**grams: "the students," "students opened", "opened their"
- **tri**grams: "the students opened", "students opened their"
- **4-**grams: "the students opened their"

Idea: Collect statistics about how frequent different n-grams are, and use these to predict the next word

More detail: 

Make a *simplifying assumption*: x^(t+1) depends only on the preceding *n-1* words.

Question: How do we get these n-gram and (n-1)-gram probabilities?

Answer: By counting them in some large corpus of text

Equation: P(w|"some phrase") = count("some phrase" w)/count("some phrase")

With this, based off of the corpus, we would count the number of times a word precedes some phrase, and use that probability to predict what the future word will be in provided contexts.

In a case where you are learning a 4-gram language model, discarding the preceding context (based on our simplifying assumption) may result in the wrong words being predicted based on the given context. This is one problem with n-gram language models.

### Sparsity problems

1. What if the count on top of the prediction equation is zero? What if, for some particular word W, "some phrase w" never occurred due to not being present in the training data, even if it's a perfectly valid scenario in other contexts.

A partial solution to this is to add a slight delta to the count for every word in the vocabulary, this way every word in the vocabulary has at least some small probability. This technique is called *smoothing* – you are going from a very sparse probability distribution, to a very smooth probability distribution, where everything at least has some probability.

1. What if the count on the bottom of the prediction is zero? That is, what if you cannot find "some phrase" in the corpus (it never occurred)? In this case, you would reduced your n-gram model to (n-1), so you would instead look for "some" instead of "some phrase" in the corpus, and subsequently run the model using this new equation. This technique is called *backoff*.

These sparsity problems get worse as you increase *n* in your n-gram language model. In practice, you usually can't have *n* any greater than five (5).

Additional problems can occur when there is not much granularity in the probability distribution between two or three words.

### Storage problems

**Storage:** Need to store count for all *n*-grams you saw in the corpus

Increasing *n* or increasing corpus size increases model size!

## Generating text with an n-gram Language Model

You can also use a Language Model to generate text. By continually predicting the succeeding word of a n-gram (such as a trigram) language model, you can eventually generate some piece of text. While this piece of text might be grammatical, it will likely be incoherent. Thus, we need to consider more than a couple of *n* words at a time if we want to model language well. But, as stated before, increasing *n* worsens the sparsity problem and increases model size.

## How to build a *neural* Language Model?

### Fixed-window neural Language Model

1. Given a string of text, discard everything outside of the fixed-size window that you have set.
2. Similar to using NER to classify a word (see Lecture 3), represent the remaining words as one-hot vectors.
3. Use these vectors to look up the word embeddings for the vectors and concatenate these together. Put these through some linear layer and a nonlinearity function to get a hidden layer.
4. Put this through another linear layer and a softmax function to generate a probability distribution of all of the words in the vocabulary. 

**Improvements** over n-gram LM:

- No sparsity problem
- Don't need to store all observed n-grams

**Remaining problems:**

- Fixed window is *too small*
- Enlarging window enlarges *W* (weight matrix)
- Window can never be large enough!
- *No symmetry* in how inputs are processed, as inputs are multiplied by different weights

We need a neural architecture that can process any length input – most of the problems here occurred because we had to make a simplifying assumption that there is some fixed-window.

## Recurrent Neural Networks (RNN)

A family of neural architectures

In an RNN, we have a sequence of hidden states (unlike a singular hidden state in the previous model), and we have as many of them as we have inputs. Each hidden state *ht* is computed based on the previous hidden state and also the input on that step.

**Why are they called hidden states?** You can think of these as a single-state that is mutating over time, several versions of the same thing. We often call these time-steps.

**Core idea:** Apply the same weights *W* repeatedly, on every-time step. This is called *unrolling*, and allows us to process any length input we want – we can apply the exact same transformation on every step, as our weights do not change.

The "outputs" of an RNN are optional and do not have to be computed, but can be if you would like to use them in your next step.

### RNN Language Model

**Advantages:**

- Can process any length input
- Computation for step *t* (in theory) use information from many steps back
- Model size doesn't increase for long input
- Same weights applied on every timestep, so there is symmetry in how inputs are processed

**Disadvantages:**

- Recurrent computation is slow
- In practice, difficult to access information from many steps back

### Training a RNN Language Model

1. Get a big corpus of text which is a sequence of words
2. Feed into RNN-LM; compute output distribution y-hat^(t) for every step t
    - i.e. predict probability distribution of *every word*, given words so far
3. Loss function on step t is cross-entropy between predicted probability distribution y-hat^(t), and the true next word y^(t) (one-hot for x^(t+1))
4. Average this to get overall loss for entire training set

However: Computing loss and gradients across entire corpus is *too expensive*

In practice, consider the sequence of words as a sentence (or a document)

Recall: Stochastic Gradient Descent (SGD) allows us to compute loss and gradients for small chunk of data, and update

Compute loss J(θ) for a sentence (actually a batch of sentences), compute gradients and update weights. Repeat.

### Generating text with a RNN Language Model

Just like with an n-gram LM, you can use an RNN LM to generate text by *repeated sampling*. Sampled output is the next step's input.

You can train an RNN-LM on any kind of text and then generate text in that style.

## Evaluating Language Models

The standard evaluation metric for Language Models is **perplexity.** Perplexity is defined as the inverse probability of the corpus according to the language model.

## Why should we care about Language Modeling?

Language Modeling is a benchmark task that helps us measure our progress on understanding language.

Language Modeling is a subcomponent of many NLP tasks, especially those involving generating text or estimating the probability of text:

- Predictive typing
- Speech recognition
- Handwriting recognition
- Spelling/grammar correction
- Authorship identification
- Machine translation
- Summarization
- Dialogue
- etc.

# Recap

- **Language Model:** A system that predicts the next word
- **Recurrent Neural Network:** A family of neural networks that:
    - Take sequential input of any length
    - Apply the same weights on each step
    - Can optionally produce output on each step
- Recurrent Neural Network ≠ Language Model
- RNNs are a great way to build an LM, but they're useful for so much more
    - RNNs can be used for tagging (part-of-speech tagging, named entity recognition)
    - RNNs can be used for sentence classification (sentiment classification)
        - How to compute sentence encoding? Compute the element-wise max/mean of every hidden state to get sentence encoding, which works better than using the final hidden state alone
    - RNNs can be used as an encoder module (question answering, machine translation)
    - RNN-LMs can be used to generate text (speech recognition, machine translation, summarization)
    

## Terminology:

RNN described in this lecture = "vanilla RNN"

There are other types of RNNs: GRU, LSTM, and multi-layer RNNs (stacking RNNs on top of one another)