# Lecture 10: Question Answering

## Motivation: Question Answering

With massive collections of full-text documents, i.e., the web, simply returning relevant documents is of limited use. We often want answers to our questions, especially using mobile devices or assistants like Alexa, Siri, Google Assistant.

We can factor this into two parts:

1. Finding documents that (might) contain an answer
    1. This can be handled by traditional information retrieval/web search
2. Finding an answer in a paragraph or a document
    1. This problem is often termed **Reading Comprehension**

### Machine Comprehension

A machine **comprehends** a passage of text if, for any question regarding that text can be answered correctly by a majority of native speakers, that machine can provide a string which those speakers would agree both answers that question, and does not contain information irrelevant to that question.

## BiDAF: Bi-Directional Attention Flow for Machine Comprehension

There are variants of and improvements to the BiDAF architecture over the years, but the central idea is the **Attention Flow Layer**

**Idea:** attention should flow both ways – from the context to the question and from the question to the context

# Additional Resources

[https://towardsdatascience.com/nlp-building-a-question-answering-model-ed0529a68c54](https://towardsdatascience.com/nlp-building-a-question-answering-model-ed0529a68c54)

[https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b](https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b)