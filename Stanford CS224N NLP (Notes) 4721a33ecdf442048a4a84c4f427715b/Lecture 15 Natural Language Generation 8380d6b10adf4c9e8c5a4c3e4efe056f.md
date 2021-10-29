# Lecture 15: Natural Language Generation

## Natural Language Generation (NLG)

Natural Language Generation refers to any setting in which we *generate* (i.e. write) new text.

NLG is a subcomponent of:

- Machine Translation
- (Abstractive) Summarization
- Dialogue (chit-chat and task-based)
- Creative writing: storytelling, poetry generation
- Freeform Question Answering (i.e. answer is generated, not extracted from text or knowledge base)
- Image captioning

## Recap

Language Modeling: the task of predicting the next word, given the words so far

A system that produces this probability distribution is called a Language Model

If that system is an RNN, it's called a RNN-LM

Conditional Language Modeling: the task of predicting the next word, given the words so far, and also some other input x

Examples of conditional language modeling tasks:

- Machine Translation (x=source sentence, y=target sentence)
- Summarization (x=input text, y=summarized text)
- Dialogue (x=dialogue history, y=next utterance)

### Decoding algorithms

Question: Once you've trained your (conditional) language model, how do you use it to generate text?

Answer: A *decoding algorithm* is an algorithm you use to generate text from your language model.

We've learned about two deciding algorithms:

- Greedy decoding
- Beam search

### Greedy decoding

A simple algorithm:

On each step, take the most probable word

Use that as the next word, and feed it as input on the next step

Keep going until you produce <END> (or reach some max length)

Due to lack of backtracking (unable to go make and modify previous input if you made a mistake), output can be poor (e.g., ungrammatical, unnatural, nonsensical)

### Beam search decoding

A search algorithm which aims to find a high-probability sequence (not necessarily the optimal sequence, though) by tracking multiple possible sequences at once.

Core idea: On each step of the decoder, keep track of the k most probable partial sequences (which we call hypotheses)

- k is the beam size

After you reach some stopping criterion, choose the sequence with the highest probability (factoring in some adjustment for length)

### What's the effect of changing beam size k?

Small k has similar problems to greedy decoding (k=1)

- Ungrammatical, unnatural, nonsensical, incorrect

Larger k means you consider more hypotheses

- Increasing k reduces some of the problems above
- Larger k is more computationally expensive
- But increasing k can introduce other problems:
    - For NMT, increasing k too much decreases BLEU score. This is primarily because large-k beam search produces too-short translations (even with score normalization).
    - In open-ended tasks like chit-chat dialogue, large k can make output more generic.

### Sampling-based deocding

Pure sampling:

On each step t, randomly sample from the probability distribution P_t to obtain your next word.

Like greedy decoding, but sample instead of argmax.

Top-n sampling:

On each step t, randomly sample from P_t, restricted to just the top-n most probably words

Like pure sampling, but truncate the probability distribution

n=1 is greedy search, n=V is pure sampling

Increase n to get more diverse/risky output

Decrease n to get more generic/safe output

Both of these are more efficient than beam search – no multiple hypotheses

### Softmax temperature

Recall: On timestep t, the LM computes a probability distribution P_t by applying the softmax function to a vector of scores

You can apply a *temperature hyperparameter (tau)* to the softmax

Raise the temperature (tau): P_t becomes more uniform:

Thus more diverse output (probability is spread around vocab)

Lower the temperature (tau): P_t becomes more spiky:

Thus less diverse output (probability is concentrated on top words)

This is not a decoding algorithm. It is used in conjunction with a decoding algorithm, applied at test time.

## Decoding algorithms: in summary

Greedy decoding is a simple method; gives low quality output

Beam search (especially with high mean size) searches for high-probability output

- Delivers better quality than greedy, but if beam size is too high, can return high-probability but unsuitable output (e.g., generic, short)

Sampling methods are a way to get more diversity and randomness

- Good for open-ended/creative generation (poetry, stories)
- Top-n sampling allows you to control diversity

Softmax temperature is another way to control diversity

- It's not a decoding algorithm; it's a technique that can be applied alongside any decoding algorithm.

## NLG tasks and neural approaches to them

### Summarization: task definition

Task: given input text x, write a summary y which is shorter and contains the main information of x.

Summarization can be single-document or multi-document.

- Single-document means we write a summary y of a single document x.
- Multi-document means we write a summary y of multiple documents x_1,...,x_n.

Typically x_1,...,x_n have overlapping content: e.g., news articles about the same event

*Sentence simplification* is a different but related task: rewrite the source text in a simpler (sometimes shorter) way

### Summarization: two main strategies

Extraction summarization: Select parts (typically sentences) of the original text to form a summary. This is easier, but restrictive (no paraphrasing).

Abstractive summarization: Generate new text using natural language generation techniques. This is more difficult, but more flexible (more human).

### Pre-neural summarization

Pre-neural summarization systems were mostly extractive. Like pre-neural MT, they typically had a pipeline:

- **Content selection:** choose some sentences to include
- **Information ordering:** choose an ordering of those sentences
- **Sentence realization:** edit the sequence of sentences (e.g., simplify, remove parts, fix continuitiy issues)

Pre-neural content selection algorithms:

- Sentence scoring functions can be based on:
    - Presence of topic keywords
    - Features such as where the sentence appears in the document
- Graph-based algorithms view the document as a set of sentences (nodes), with edges between each sentence pair:
    - Edge weight is proportional to sentence similarity
    - Use graph algorithms to identify sentences which are *central* in the graph

### Summarization evaluation: ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Like BLEU, it's based on n-gram overlap. Here are the differences:

- ROUGE has no brevity penalty
- ROUGE is based on recall, while BLEU is based on precision
    - Arguably, precision is more important for MT (then add brevity penalty to fix under-translation), and recall is more important for summarization (assuming you have a max length constraint)
    - However, often a F1 (combination of precision and recall) version of ROUGE is reported anyway
    

BLEU is reported as a single number, which is a combination of the precisions for n=1,2,3,4 n-grams

ROUGE scores are reported separately for each n-gram

The most commonly-reported ROUGE scores are:

- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap
- ROUGE-L: longest common subsequence overlap

### Neural summarization: copy mechanisms

seq2seq+attention systems are good at writing fluent output, but bad at copying over details (like rare words) correctly

Copy mechanisms use attention to enable a seq2seq system to easily copy words and phrases from the input to the output

- This is very useful for summarization
- Allowing both copying and generating gives us a hybrid extractive/abstractive approach

Big problem with copying mechanisms:

- They copy too much! Mostly long phrases, sometimes even whole sentences
- What *should* be an abstractive system collapses to a mostly extractive system

Another problem:

- They're bad at overall content selection, especially if the input document is long
- No overall strategy for selecting content

### Neural summarization: better content selection

Recall: pre-neural summarization had separate stages for content selection and surface realization (i.e. text generation)

In a standard seq2seq+attention summarization system, these two stages are mixed in together:

- On each step of the decoder (i.e. surface realization), we do word-level content selection (attention)
- This is bad: there is no *global* content selection strategy

One solution is "bottom-up summarization"

## Bottom-up summarization

Content selection stage: Use a neural sequence-tagging model to tag words as *include* or *don't-include*

Bottom-up attention stage: The seq2seq+attention system can't attend to words tagged *don't include* (apply a mask)

This is simple, but effective! Better overall content selection strategy and less copying of long sequences (i.e., more abstractive output)

### Neural summarization via Reinforcement Learning

Main idea: Use Reinforcement Learning (RL) to directly optimize ROUGE-L

- By contrast, standard maximum likelihood (ML) training can't directly optimize ROUGE-L because it's a non-differentiable function
- Interesting finding: Using RL instead of ML achieved higher ROUGE scores, but lower human judgment scores. Overall, a hybrid approach does best.

## Dialogue

"Dialogue" encompasses a large variety of settings:

- **Task-oriented** dialogue:
    - Assistive (e.g., customer service, giving recommendations, question answering, helping user accomplish a task like buying or booking something)
    - Co-operative (two agents solve a task together through dialogue)
    - Adversarial (two agents compete in a task through dialogue)
- **Social** dialogue:
    - Chit-chat (for fun or company)
    - Therapy / mental wellbeing

### Pre and post-neural dialogue

Due to the difficult of open-ended freeform NLG, pre-neural dialogue systems more often used predefined templates, or retrieve an appropriate response from a corpus of responses.

As in summarization research, since 2015 there have been many papers applying seq2seq methods to dialogue – thus leading to a renewed interest in open-ended freeform dialogue systems

### Seq2seq-based dialogue

However, it quickly became apparent that naive application of standard seq2seq+attention methods has serious pervasive deficiencies for (chitchat) dialogue:

- Genericness/boring responses
- Irrelevant responses (not sufficiently related to context)
- Repetition
- Lack of context (not remembering conversation history)
- Lack of consistent persona

### Irrelevant response problem

Problem: seq2seq often generates response that's unrelated to user's utterance

- Either because it's generic (e.g., "I don't know")
- Or because changing the subject to something unrelated

One solution: optimize for Maximum Mutual Information (MMI) between input S and response T

### Generic/boring response problem

East test-time fixes:

- Directly upweight rare words during beam search
- Use a sampling decoding algorithm rather than beam search

Conditioning fixes:

- Condition the decoder on some additional content (e.g., sample some content words and attend to them)
- Train a retrieve-and-refine model rather than a generate-from-scratch model
    - i.e., sample an utterance from your corpus of human-written utterances, and edit it to fit the current scenario
    - This usually produces much more diverse/human-like/interesting utterances

### Repetition problem

Simple solution:

- Directly block repeating n-grams during beam search; this is usually pretty effective

More complex solutions:

- Train a coverage mechanism – in seq2seq, this is an objective that prevents the attention mechanism from attending to the same words multiple times.
- Define a training objective to discourage repetition
    - If this is a non-differentiable function of the generated output, then will need some technique like RL to train

### Lack of consistent persona problem

Li et. al. (2016) proposed a seq2seq dialogue model that learns to encode both conversation partners' personas as embeddings

- The generated utterances are conditioned on the embeddings

## Storytelling

Most neural storytelling work uses some kind of prompt

- Generate a story-like paragraph given an image
- Generate a story given a brief writing prompt
- Generate the next sentence of a story, given the story so far (story continuation). This is different to the previous two, as we are not concerned with the system's performance over several generated sentences.

### Generating a story from an image

Question: How to get around the lack of parallel data?

Answer: Use a common sentence-encoding space

**Skip-thought vectors** are a type of general-purpose sentence embedding method

- The idea is similar to how we learn an embedding for a word by trying to predict the words around it

Using COCO (an image captioning dataset), learn a mapping from images to the skip-thought encodings of their captions

Using the target style corpus, train an RNN-LM to decode a skip-thought vector to the original text – then put these two together.

### Generating a story from a writing prompt

seq2seq prompt-to-story model proposed by Fan et. al.:

- It's convolutional-based
    - This makes it faster than RNN-based seq2seq
- Gated multi-head multi-scale self-attention
    - The self-attention is important for capturing long-range context
    - The gates allow the attention mechanism to be more selective
    - The different attention heads attend at different scales – this means there are different attention mechanisms dedicated to retrieving fine-grained information and coarse-grained information
- Model fusion:
    - Pretrain one seq2seq model, then train a *second* seq2seq model that has access to the hidden states of the first
    - The idea is that the first seq2seq model learns general LM and the second learns to condition on the prompt

With this, the results are impressive – related to the prompt, diverse/non-generic, and stylistically dramatic. However, it is mostly atmospheric/descriptive/scene-setting; less events or plot. When generating for longer, it mostly stays on the same idea without moving forward to new ideas – coherence issues.

### Challenges in storytelling

Stories generated by neural LMs can sound fluent, but are meandering, nonsensical, with no coherent plot.

What's missing?

LMs model *sequences of words*. Stores are *sequences of events*.

To tell a story, we need to understand and model:

- Events and the causality structure between them
- Characters, their personalities, motivations, histories, and relationships to other characters
- State of the world (who and what is where and why)
- Narrative structure (e.g., exposition → conflict → resolution)
- Good storytelling principles (don't introduce a story element then never use it)

This is incredibly difficult.

## NLG Evaluation

### Automatic evaluation metrics for NLG

Word overlap based metrics (BLEU, ROUGE, METEOR, F1, etc.)

We know that they're not ideal for machine translation. They're even worse for summarization, which is more open-ended than machine translation.

- Unfortunately, ROUGE also rewards extractive summarization systems more than abstractive systems

And they're even worse for dialogue, which is more open-ended than summarization.

What about perplexity?

- Captures how powerful your LM is, but doesn't tell you anything about generation (e.g., if your decoding algorithm is bad, perplexity is unaffected)

Word embedding based metrics?

- Main idea: compare the similarity of the word embeddings (or average of word embeddings), not just the overlap of the words themselves. Captures semantics in a more flexible way.
- Unfortunately, still doesn't correlate well with human judgments for open-ended tasks like dialogue

We have no automatic metrics to adequately capture overall quality (i.e. a proxy for human quality judgment).

But we can define more focused automatic metrics to capture particular aspects of generated text:

- Fluency (compute probability with respect to well-trained LM)
- Correct style (probability with respect to LM trained on target corpus)
- Diversity (rare word usage, uniqueness of n-grams)
- Relevance to input (semantic similarity measures)
- Simple things like length and repetition
- Task-specific metrics e.g., compression rate for summarization

Though these don't measure overall quality, they can help us track some important qualities that we care about.

### Human evaluation

Human judgments are regarded as the gold standard. Of course, we know that human evaluation is slow and expensive.

Conducting human evaluation effectively is very difficult:

- inconsistent
- can be illogical
- lose concentration
- misinterpret your question
- can't always explain why they feel the way they do

# Additional Resources

[https://www.ibm.com/blogs/watson/2020/11/nlp-vs-nlu-vs-nlg-the-differences-between-three-natural-language-processing-concepts/](https://www.ibm.com/blogs/watson/2020/11/nlp-vs-nlu-vs-nlg-the-differences-between-three-natural-language-processing-concepts/)