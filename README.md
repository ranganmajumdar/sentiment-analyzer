# Sentiment Analyzer 🧠

A Python + NLP project that detects whether a sentence is Positive, Negative, or Neutral. Built in two versions using different NLP libraries.

---

## Version 1 — TextBlob (`app.py`)
Type sentences one by one in the terminal and get instant sentiment results.

### How to Run
1. Install dependencies: `pip3 install textblob`
2. Run: `python3 app.py`
3. Type any sentence and get instant sentiment analysis!

---

## Version 2 — VADER (`app_2.py`)
Write multiple sentences in a text file and get a full analysis report with summary.

### How to Run
1. Install dependencies: `pip3 install vaderSentiment`
2. Add your sentences to `sentences.txt` (one per line)
3. Run: `python3 app_2.py`
4. Get a full report + summary!

---

## Example Output (Version 2)
```
==================================================
       SENTIMENT ANALYSIS REPORT
==================================================

1. I love this phone, it is amazing!
   Sentiment : Positive 😊
   Score     : 0.8126

2. This movie was absolutely terrible.
   Sentiment : Negative 😞
   Score     : -0.6249

==================================================
               SUMMARY
==================================================
  Total Sentences : 6
  Positive        : 3
  Negative        : 2
  Neutral         : 1
==================================================
```

---

## Tech Used
- Python 3
- TextBlob (Version 1)
- VADER - Valence Aware Dictionary and sEntiment Reasoner (Version 2)

## Why two versions?
TextBlob is simple but struggles with context and informal language.
VADER handles slang, social media text, and informal sentences much better.

## What I Learned
- Python basics (functions, loops, conditionals)
- Natural Language Processing (NLP) concepts
- Difference between TextBlob and VADER
- Reading and processing text files in Python
- Git & GitHub for version control
