# Stanford CS224N NLP (Notes)

[Lecture 1: Word Vectors](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%201%20Word%20Vectors%20e42a429c13b34203a55e89f55446f58d.md)

[Lecture 2: Word Vectors and Senses](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%202%20Word%20Vectors%20and%20Senses%20dbcf9a10d1174f94bc0af236bd6f88a4.md)

[Lecture 3: Neural Networks](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%203%20Neural%20Networks%20db0d60d743b34524bd7f32353ec81de8.md)

[Lecture 4: Backpropagation](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%204%20Backpropagation%2023487439819b4ba380e983ab852cfc6b.md)

[Lecture 5: Dependency Parsing](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%205%20Dependency%20Parsing%20adc2c7bd21374aeea841c1ef31d64c37.md)

[Lecture 6: Language Models and RNNs](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%206%20Language%20Models%20and%20RNNs%20bc263019ebf04c9ca128d39a3c3e60b7.md)

[Lecture 7: Vanishing Gradients, Fancy RNNs](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%207%20Vanishing%20Gradients,%20Fancy%20RNNs%20d5239ead35e04f059ba0eb540f26fd48.md)

[Lecture 8: Translation, Seq2Seq, Attention](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%208%20Translation,%20Seq2Seq,%20Attention%203655b92bf2e24d629aed79f127847315.md)

[Lecture 10: Question Answering](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2010%20Question%20Answering%200c1633f3f46848889121214d87bbc555.md)

[Lecture 11: Convolutional Networks for NLP](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2011%20Convolutional%20Networks%20for%20NLP%2023b2a1f292454d2db745fb5fefc51d5e.md)

[Lecture 12: Subword Models](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2012%20Subword%20Models%20d4cc460ae2e5409d9e0a4154eb66b2a3.md)

[Lecture 13: Contextual Word Embeddings](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2013%20Contextual%20Word%20Embeddings%20b79dbc0ae96a4ca78e2b031d09b86a5b.md)

[Lecture 14: Transformers and Self-Attention](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2014%20Transformers%20and%20Self-Attention%20187a70dd9bce4714b488fa9e5f801209.md)

[Lecture 15: Natural Language Generation ](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2015%20Natural%20Language%20Generation%208380d6b10adf4c9e8c5a4c3e4efe056f.md)

[Lecture 16: Coreference Resolution](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2016%20Coreference%20Resolution%20ec57889324014ef5aee06c2462329842.md)

[Lecture 17: Multitask Learning](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2017%20Multitask%20Learning%208f74c76dbb504ae0a79b7a25439aa3e0.md)

[Lecture 18: Constituency Parsing, TreeRNNs](Stanford%20CS224N%20NLP%20(Notes)%204721a33ecdf442048a4a84c4f427715b/Lecture%2018%20Constituency%20Parsing,%20TreeRNNs%204fc164a513f345c98450f6cf2c662b73.md)

# Summary

Natural Language Processing is the field concerned with allowing machines to understand and derive meaning from human language. Largely, the goal is to understand the context and meaning of provided language (or texts) in order to make some sort of decision, provide an answer, or perform computations.

Historically, NLP has followed a path from being symbolic → statistical → neural. In traditional NLP, words are represented as discrete symbols.

A big focus in NLP is being able to represent and understand words by their context. For any given word in a text, the context is the set of words that appear nearby within a fixed sized window.

Words can be represented by one-hot vectors – vectors are also sometimes called embeddings or representations. Because they are vectors, they can be grouped together and put into a vector/embedding space. These are distributed representations.

With our models, a common goal is to minimize the loss in our objective function, and maximize prediction accuracy. When viewed graphically, contour lines show different levels of the objective function, and subsequent values of loss. We use stochastic gradient descent (SGD) as an alternative to regular gradient descent to optimize our objective function, as the gradient of the function is very expensive to compute. SGD allows use to repeatedly sample windows and update the function after each sample.

For basic toolsets, NLTK is a "swiss army knife" for NLP, allowing you to do a lot of basic things without too much nuance. word2vec is a neural net that vectorizes words and predicts context words based on the center word through surrounding vectorizations.

word2vec utilizes two different models: the Skip-Gram model, which predicts context (or outside) words given a center word, independent of position; and the Continuous Bag of Words, which predicts the center word from a bag of context words. 

With Skip-Gram, implement the model with negative sampling, where we select a random number of "negative" words to update the weights with. A negative word is one where we want the word to be represented by a zero in a one-hot vector, whereas our center word is the positive word. Attempting to train the network with all of the words and combinations is too computationally expensive. We want to maximize the probability that real outside words appear, and minimize the probability that random words appear around the center word.

In the CBOW model, the distributed representations of context are combined to prefix the word in the middle. In the Skip-gram model, the distributed representation of the input is used to predict the context.

Biggest concepts: 

- Word Vectors, Word Embeddings
- Neural Networks and their variants
- Backpropagation
- Dependency vs Constituency Parsing
- Neural Networks and their impact on accuracy, usability, etc.
- The use of transformers and attention to improve models
- NLG vs NLP vs NLU
- NLP major tasks (machine translation, question answering, classification)
- Linguistics as a whole, use of symbols in language, complexity of language with context, meaning, etc.
- BERT, GloVE and similar models/concepts