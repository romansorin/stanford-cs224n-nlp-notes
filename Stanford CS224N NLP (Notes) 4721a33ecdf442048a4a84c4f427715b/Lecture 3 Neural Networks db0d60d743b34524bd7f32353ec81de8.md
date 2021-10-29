# Lecture 3: Neural Networks

### Classification difference with word vectors

Commonly in NLP deep learning:

- We learn **both** *W* **and** word vectors *x*
- We learn **both** conventional parameters **and** representations
- The word vectors re-represent one-hot vectors – move them around in an intermediate layer vector space – for easy classification with a (linear) softmax classifier via layer *x = Le*

Neural networks were created in a way to attempt to model the brain's natural neuron functionality, but artificial neurons carry their own terminological "baggage"

### A neural network = running several logistic regressions at the same time

If we feed a vector of inputs through a bunch of logistic regression functions, then we get a vector of outputs. We want the neural network to self organize – so, we feed the vector of outputs into another logistic regression. Training the model through several intermediate functions (or logistic regressions) minimizes the cross entropy loss. The intermediate functions are tasked with finding a useful way to calculate values from the underlying data, such that it'll help our final classifier make a good decision.

Before we know it, there is a multilayer neural network.

In order for a neural network to learn anything interesting and function in a complex manner, non-linearities (e.g., function approximation [regression, classification]) are needed. Without them, deep neural nets can't do anything more than a linear transform.

- Extra layers could just be compiled down into a single linear transform: W(1)W(2)x = Wx. Multiple linear transforms in a sequence can be transformed into a single linear transform.
- With more layers, they can approximate more complex functions.
- We want to be able to do function or curve-fitting in a given space, and deal with sporadic data that does not conform to linear approximations; thus, using non-linearities allows us to learn any kind of "curvy" pattern.

## Named Entity Recognition (NER)

The task: find and classify names in text (ex: European Commission [ORG], Britian [LOC])

Possible purposes:

- Tracking mentions of particular entities in documents
- For question answering, answers are usually named entities
- A lot of wanted information is really associated between named entities
- The same techniques can be extended to other slot-filling classifications

Often following by Named Entity Linking/Canonicalization into Knowledge Base

A common way of doing this is by predicting entities by classifying words in context, and then extracting entities as word subsequences

### Why might NER be hard?

Hard to work out boundaries of entity. Example:

`First National Bank Donates 2 Vans to Future School of Fort Smith`

- Is the first entity "First National Bank" or "National Bank"

Hard to know if something is an entity

- Is there a school called "Future School" or is it a future school?

Hard to know class of unknown/novel entity

Entity class is ambiguous and depends on context. Example:

Charles Schwab in text is 90% a brokerage, but next to Larry Ellison, it references Charles Schwab the person

## Binary word window classification

In general, classifying single words is rarely done

Interesting problems like ambiguity arise in context!

Example: auto-antonyms:

- "To sanction" can mean "to permit" or "to punish"
- "To seed" can mean "to place seeds" or "to remove seeds"

Example: resolving linking of ambiguous named entities:

- Paris → Paris, France vs Paris Hilton vs. Paris, Texas
- Hathaway → Berkshire Hathaway vs. Anne Hathaway

### Window classification

**Idea:** classify a word in its context window of neighboring words.

For example, **Named Entity Classification** of a word in context:

- Person, Location, Organization, None

A simple way to classify a word in context might be to average the word vectors in a window and to classify the average vector

- Problem: that would lose position information

Classify a center word by concatenating the word vectors surrounding it in a window

### Backpropagation

Computing gradients algorithmically and efficiently

Used by deep learning software frameworks (TensorFlow, PyTorch, Chainer, etc.)

## Additional Resources

[https://wiki.pathmind.com/neural-network](https://wiki.pathmind.com/neural-network)