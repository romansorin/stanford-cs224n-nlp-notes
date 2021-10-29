# Lecture 7: Vanishing Gradients, Fancy RNNs

## Vanishing Gradient

Imagine you are trying the compute the derivate of the loss of h^4 with respect to the first hidden layer, h^1. To get this, you'd have to compute the gradients of all of the hidden layers and move through these layers until you reach h^4.

**Problem:** When the gradients of the intermediate/hidden layers are small, the gradient signal gets smaller and smaller as it backpropagates further.

**Why is this a problem?:** Gradient signal from faraway is lost because it's much smaller than gradient signal from close-by. So model weights are only updated with respect to near effects, not long-term effects.

Gradient can be viewed as a measure of the effect of the past on the future

If the gradient becomes vanishingly small over longer distances (step to to step t+n), then we can't tell whether:

1. There's no dependency between step t and t+n in the data
2. We have wrong parameters to capture the true dependency between t and t+n

It's tough to tell which of these two is the problem.

### Effect of vanishing gradient on RNN-LM

**LM task:** *When she tried to print her tickets, she found that the printer was out of toner. She went to the stationery store to buy more toner. It was very overpriced. After installing the toner into the printer, she finally printed her ____*

To learn from this training example, the RNN-LM needs to model the dependency between "tickets" on the 7th step and the target word "tickets" at the end.

But if the gradient is small, the model can't learn this dependency.

- So the model is unable to predict similar long-distance dependencies at test time

**LM task:** *The writer of the books ___* (is, are)

**Correct answer:** *The writer of the books is planning a sequel*

**Syntactic recency:** *The writer of the books is* (correct)

**Sequential recency:** *The writer of the books are* (incorrect)

Due the vanishing gradient, RNN-LMs are better at learning from sequential recency than syntactic recency, so they make this type of error more often than we'd like

### Why is exploding gradient a problem?

If the gradient becomes too big, then the SGD update step becomes too big.

This can cause bad updates: we take too large a step and reach a bad parameter configuration (with large loss)

In the worst case, this will result in ****Inf** **or **NaN** in your network (then you have to restart training from an earlier checkpoint)

### Gradient clipping is the solution

**Gradient clipping:** if the norm of the gradient is greater than some threshold, scale it down before applying SGD update

**Intuition:** take a step in the same direction, but a smaller step

When looking at a loss surface of an RNN, a "cliff" may occur (which is showing a very steep gradient). Without gradient clipping, gradient descent can take very big steps due to the steep gradient, resulting in bad, extreme updates.

Gradient clipping reduces the size of those steps, so effect is less drastic. 

### How to fix the vanishing gradient problem?

The main problem is that it's too difficult for the RNN to learn to preserve information over many timesteps.

In a vanilla RNN, the hidden state is constantly being rewritten

How about an RNN with separate memory? This is where **Long Short-Term Memory (LSTM)** come into play.

## Long Short-Term Memory (LSTM)

A type of RNN proposed by Hochreiter and Schmidhuber in 1997 as a solution to the vanishing gradients problem.

On step t, there is a hidden state h^t and a cell state c^t:

- Both are vectors length n
- The cell stores long-term information
- The LSTM can erase, write, and read information from the cell

The section of which information is erased/written/read is controlled by three corresponding gates:

- The gates are also vectors of length n
- On each timestep, each element of the gates can be open (1), closed (0), or somewhere in-between. If the gate is open, some type of information can pass through; if the gate is closed, information is not allowed to pass through.
- The gates are dynamic: their value is computed based on the current context

In brief, we have a sequence of inputs x^t and we will compute a sequence of hidden states h^t and cell states c^t. On timestep t:

f^t = **Forget gate:** controls what is kept vs forgotten, from previous cell state. This is computed from the previous hidden state (h^t-1) and current input (x^t). It is computed using the sigmoid function, so the value lies somewhere between 0 and 1.

i^t = **Input gate:** controls what parts of the the new cell content are written to cell

o^t = **Output gate:** controls what parts of cell are output to hidden state

c(~)^t = **New cell content:** this is the new content to be written to the cell

c^t = **Cell state:** erase ("forget") some content from last cell state, and write ("input") some new cell content

h^t = **Hidden state:** read ("output") some content from the cell

### How does LSTM solve vanishing gradients?

The LSTM architecture makes it easier for the RNN to preserve information over many timesteps

- e.g., if the forget gate is set to remember everything on every timestep, then the info in the cell is preserved indefinitely
- By contrast, it's harder for vanilla RNN to learn a recurrent weight matrix W_h that preserves info in hidden state

LSTM doesn't guarantee that there is no vanishing/exploding gradient, but it does provide an easier way for the model to learn long-distance dependencies

### LSTMs: Real-world success

In 2013-2015, LSTMs started achieving state-of-the-art results:

- Successful tasks include: handwriting recognition, speech recognition, machine translation, parsing, image captioning
- LSTM became the dominant approach

Now (2019), other approaches (e.g., Transformers) have become more dominant for certain tasks  

## Gated Recurrent Units (GRU)

Proposed as a simpler alternative to the LSTM

On each timestep t we have an input x^t and hidden state h^t (no cell state.).

u^t = **Update gate:** controls what is parts of hidden state are updated vs preserved

r^t = **Reset gate:** controls what parts of previous hidden state are used to compute new content

h(~)^t = **New hidden state content:** reset gate selects useful parts of previous hidden state. Use this and current input to compute new hidden content.

h^t = **Hidden state:** update gate simultaneously controls what is kept from previous hidden state, and what is updated to new hidden state content

GRUs are a solution to the vanishing gradient problem as they make it easier to retain information long-term (e.g., by setting update gate to 0)

## LSTM vs GRU

Researchers have proposed many gated RNN variants, but LSTM and GRU are the most widely-used

The biggest difference is that GRU is quicker to compute and has fewer parameters

There is no conclusive evidence that one consistently performs better than the other

LSTM is a good default choice (especially if your data has particularly long dependencies, or you have lots of training data)

Rule of thumb: start with LSTM, but switch to GRU if you want something more efficient

## Is vanishing/exploding gradient just an RNN problem?

No, it can be a problem for all neural architectures (including feed-forward and convolutional), especially deep ones.

- Due to chain rule/choice of nonlinearity function, gradient can become vanishingly small as it backpropagates.
- Thus lower layers are learnt very slowly (hard to train)
- Solution: lots of new deep feedforward/convolutional architectures that add more direct connections (thus allowing the gradient to flow)

Conclusion: Though vanishing/exploding gradients are a general problem, RNNs are particularly unstable due to the repeated multiplication by the same weight matrix

## Bidirectional RNNs

A hidden state in the sentence and its word representation, in the context of the sentence, is called a contextual representation. Contextual representations only contain information about the *left* context. The *right* context may modify the meaning of the word, but it is not considered in the hidden state as it is a *future* timestep.

This can be a problem in an example like sentiment classification, as the succeeding timesteps/hidden steps may be important in modifying the context of a given word (such as "terribly", which is followed by "exciting!").

A **bidirectional RNN** has two RNNs in practice: a forward RNN as before, which computes/encides hidden states from left to right, and then a backward RNN that computes from right to left. This has completely separate weights from the forward RNN. These hidden states from both RNNs are then concatenated, which leaves you with your final representations.

Now, any given vector and its contextual representation will have *both* left and right context.

RNN_FW: General notation to mean "compute one forward step of the RNN" – it could be a vanilla, LSTM, or GRU computation.

**Note:** bidirectional RNNs are only applicable if you have access to the entire input sequence.

- They are **not** applicable to Language Modeling, because in LM you *only* have left context available.

If you do have entire input sequence (e..g, any kind of encoding), bidirectionality is powerful (you should use it by default).

For example, **BERT** (Bidirectional Encoder Representations from Transformers) is a powerful pretrained contextual representation system built on bidirectionality.

## Multi-layer RNNs

RNNs are already "deep" on one dimension (they unroll over many timesteps)

We can also make them "deep" in another dimension by applying multiple RNNs – this is a multi-layer RNN..

This allows the network to compute more complex representations

- The lower RNNs should compute lower-level features and the higher RNNs should compute higher-level features.

Multi-layer RNNs are also called *stacked* RNNs.

### Multi-layer RNNs in practice

High-performing RNNs are often multi-layer (but aren't as deep as convolutional or feed-forward networks)

Transformer-based networks (e.g., BERT) can be up to 24 layers 

# Recap

**Vanishing gradient problem:** what it is, why it happens, and why it's bad for RNNs

**LSTMs and GRUs:** more complicated RNNs that use gates to control information flow; they are more resilient to vanishing gradients

1. LSTMs are powerful but GRUs are faster
2. Clip your gradients
3. Use bidirectionality when possible
4. Multi-layer RNNS are powerful, but you might need skip/dense-connections if it's deep

# Additional resources

[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)