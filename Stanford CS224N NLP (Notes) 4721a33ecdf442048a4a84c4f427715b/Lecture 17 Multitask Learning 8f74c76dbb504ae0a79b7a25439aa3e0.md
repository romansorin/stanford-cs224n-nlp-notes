# Lecture 17: Multitask Learning

### Why has weight and model sharing not happened as much in NLP?

NLP requires many types of reasoning: logical, linguistic, emotional, visual, etc.

Requires short and long term memory.

NLP has been divided into intermediate and separate tasks to make progress; each different community has been benchmark chasing.

A single unsupervised task cannot solve it all. Language clearly requires supervision in nature.

### Why a unified multi-task model for NLP?

We want this unified model to decide how to transfer knowledge and not have to manually assign it.

Multi-task learning is a blocker for general NLP systems. Unified models can decide how to transfer knowledge (domain adaptation, weight sharing, transfer and zero shot learning)

Unified, multi-task models can:

- More easily adapt to new tasks
- Make deploying to production X times simpler
- Lower the bar for more people to solve new tasks
- Potentially move towards continual learning

### Expressing many NLP tasks in the same framework:

Sequence tagging:

- named entity recognition, aspect specific sentiment

Text classification:

- dialogue state tracking, sentiment classification

Seq2seq:

- machine translation, summarization, question answering

Three (3) equivalent "supertasks" of NLP: Language Modeling, Question Answering, Dialogue

### Multitask Learning as Question Answering

- Question Answering
- Machine Translation
- Summarization
- Natural Language Inference
- Sentiment Classification
- Semantic Role Labeling
- Relation Extraction
- Dialogue
- Semantic Parsing
- Commonsense Reasoning

Meta-Supervised learning: From {x, y} to {x, t, y} where t is the task

Use a question, q, as a natural description of the task, t, to allow the model to use linguistic information to connect tasks

y is the answer to q and x is the content necessary to answer q

# Additional Resources

[https://arxiv.org/abs/2009.09796#:~:text=Multi-task learning (MTL),learning by leveraging auxiliary information](https://arxiv.org/abs/2009.09796#:~:text=Multi%2Dtask%20learning%20(MTL),learning%20by%20leveraging%20auxiliary%20information).

[https://ruder.io/multi-task/](https://ruder.io/multi-task/)

[https://towardsdatascience.com/applications-of-zero-shot-learning-f65bb232963f#:~:text=Zero-shot learning refers to,means classifying on the fly](https://towardsdatascience.com/applications-of-zero-shot-learning-f65bb232963f#:~:text=Zero%2Dshot%20learning%20refers%20to,means%20classifying%20on%20the%20fly).