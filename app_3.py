import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# ── Download required NLTK data (only if missing) ─────────────────────────────
def download_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")

download_nltk()

# ── Initialize VADER Sentiment Analyzer ───────────────────────────────────────
sia = SentimentIntensityAnalyzer()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> dict:
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    cleaned_text = " ".join(filtered_tokens)

    return {
        "original": text,
        "filtered_tokens": filtered_tokens,
        "cleaned_text": cleaned_text,
    }


# ── Sentiment Analysis using VADER ────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    processed = preprocess(text)

    scores = sia.polarity_scores(processed["original"])
    compound = scores["compound"]

    # Classification logic
    if compound >= 0.05:
        label = "POSITIVE"
        emoji = "😊"
    elif compound <= -0.05:
        label = "NEGATIVE"
        emoji = "😞"
    else:
        label = "NEUTRAL"
        emoji = "😐"

    confidence = round(abs(compound) * 100, 2)

    return {
        "label": label,
        "confidence": confidence,
        "emoji": emoji,
        "scores": scores,
        "filtered_tokens": processed["filtered_tokens"],
        "cleaned_text": processed["cleaned_text"],
    }


# ── Display Result ────────────────────────────────────────────────────────────
def display_result(text: str, result: dict):
    print("\n" + "─" * 50)
    print(f"Input Text   : {text}")
    print(f"Cleaned Text : {result['cleaned_text']}")
    print(f"Key Tokens   : {result['filtered_tokens']}")
    print(f"Sentiment    : {result['emoji']}  {result['label']}")
    print(f"Confidence   : {result['confidence']}%")
    print(f"Detailed     : {result['scores']}")
    print("─" * 50)


# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("   NLTK Sentiment Analyzer (VADER)")
    print("=" * 50)
    print("Type any sentence to analyze sentiment.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Enter text: ").strip()

            if not user_input:
                print("⚠️  Please enter some text.\n")
                continue

            if user_input.lower() in ("quit", "exit"):
                print("\n👋 Exiting. Bye!")
                break

            result = analyze_sentiment(user_input)
            display_result(user_input, result)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Bye!")
            break


if __name__ == "__main__":
    main()