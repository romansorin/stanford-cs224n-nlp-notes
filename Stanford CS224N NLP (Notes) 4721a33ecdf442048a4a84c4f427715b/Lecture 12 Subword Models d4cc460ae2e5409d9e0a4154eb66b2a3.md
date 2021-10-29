# Lecture 12: Subword Models

## Human language sounds: Phonetics and phonology

Phonetics is the sound stream – uncontroversial "physics" – understanding the sounds and physiology of speech.

Phonology posits a small set of sets of distinctive, categorical units: **phonemes** or distinctive features. Though we are capable of making an infinite variety of sounds due to our mouths being a "continuous space", there is actually a small, distinguished space of sounds.

- Universal typology but language-particular realization
- Best evidence of categorical perception comes from phonology; within phoneme differences shrink, between phoneme magnified

### Morphology: Parts of words

Traditionally, we have morphemes as the smallest semantic unit

Deep learning: A possible way of dealing with a larger vocabulary – most unseen words are new morphological forms (or numbers)

An easy alternative is to work with character n-grams

### Words in writing system

Writing systems vary in how they represent words – or don't

Some languages may have word segmentation (ex. English puts spaces between words), whereas Chinese has no word segmentation.

Words (mainly) segmented:

- Clitics
    - Separated: Je vous ai apporté des bonbons
    - Joined: it+we+said+so = ف+ لاق + ان + اھ = ا
- Compounds
    - Separated: life insurance company employee
    - Joined: Lebensversicherungsgesellschaftsangestellter

### Models below the word level

Need to handle large, open vocabulary

- Rich morphology – languages can have very complex, long words
- Transliteration – Rewriting words not literally, but based on the letter level (rather than the word level) to make the sound systems roughly represent the same word in the target language
- Informal spelling

## Character-Level Models

1. Word embeddings can be composed from character embeddings
    1. Generates embeddings for unknown words
    2. Similar spellings share similar embeddings
    3. Solves OOV (out-of-vocabulary) problem
2. Connected language can be processed as characters

Both methods have been proven to work very successfully.

Most deep learning NLP work begins with language in its written form – it's the easily processed, found data; but human language writing systems aren't one thing:

- Phonemic (maybe digraphs)
- Fossilized phonemic
- Syllabic/moraic
- Ideographic (syllabic)
- Combination of the above

### Purely character-level models

Use of VD-CNN is a good example of a purely character-level model for sentence/text classification. Strong results via a deep convolutional stack.

## Sub-word models: two trends

Same architecture as for word-level model:

- But use smaller units: "word pieces"

Hybrid architectures:

- Main model has words; something else for characters

## Byte Pair Encoding

Originally a compression algorithm:

- Most frequent byte pair → a new byte (replace bytes with character ngrams)

A word segmentation algorithm:

- Start with vocabulary of characters
- Most frequent ngram pairs → a new ngram

Have a target vocabulary size and stop when you reach it

Do deterministic longest piece segmentation of words

Segmentation is only within words identified by some prior tokenizer (commonly Moses tokenizer for MT)

Automatically decides vocabulary for system

- No longer strongly "word" based in conventional way

### Character-level to build word-level

Convolution over characters to generate word embeddings

Fixed window of word embeddings used for PoS tagging

## Hybrid NMT

A best-of-both-worlds architecture:

- Translate mostly at the word level
- Only go to the character level when needed