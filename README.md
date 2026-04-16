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

version 3
A sentiment analysis tool built with HuggingFace Transformers and DistilBERT. Analyzes text and classifies it as POSITIVE or NEGATIVE along with a model confidence score.

What It Does

Analyzes any English sentence or paragraph
Returns a sentiment label (POSITIVE / NEGATIVE)
Shows model confidence as a visual progress bar
Supports single sentence, batch mode, and live interactive mode


Tech Stack
ToolPurposePython 3.8+Core languageHuggingFace TransformersNLP pipelineDistilBERT (SST-2)Pre-trained sentiment modelPyTorchModel backend
Model: distilbert-base-uncased-finetuned-sst-2-english
A compressed version of BERT, fine-tuned on the Stanford Sentiment Treebank dataset. First run downloads ~260MB and caches it locally.

Installation
bashpip install transformers torch

Usage
bashpython3 app_4.py
The script runs three modes automatically:
1. Single Sentence
Analyzes one hardcoded example sentence on startup.
2. Batch Mode
Analyzes a predefined list of reviews and prints a summary showing total positive vs negative count and average confidence.
3. Interactive Mode
Live terminal input — type any sentence and get instant results. Type quit to exit.

Understanding the Output
───────────────────────────────────────────────────────
  Text       : Best purchase I've made this year!
  Sentiment  : 😊 POSITIVE
  Confidence : [██████████████████████████████] 99.8%
───────────────────────────────────────────────────────
What "Confidence" Means
The confidence score is the model's certainty in its own prediction — not a measure of your writing quality or how emotional your sentence is.
Internally the model outputs two probabilities (POSITIVE + NEGATIVE) that always sum to 100%. The higher score wins and becomes the label.

High confidence (95%+) — sentence contains strong, clear words like amazing, terrible, love, hate
Medium confidence (70–90%) — sentence is mixed, mild, or slightly ambiguous
Low confidence (50–70%) — sentence is sarcastic, neutral, or genuinely unclear

Known Limitation
This model was trained on movie reviews, which tend to be strongly worded. As a result it can be overconfident on everyday sentences. Ambiguous inputs like "It was okay" or "Not bad" may still show high confidence even when the sentiment is unclear.

What it does and the tech stack
Installation and usage with all 3 modes explained
Confidence explained in plain English — since you just figured that out, it's worth documenting for anyone who reads the repo
Version history v1 → v4 so your GitHub shows the learning progression
What's next section so it looks like an active project