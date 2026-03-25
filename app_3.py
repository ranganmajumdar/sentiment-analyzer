
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

# ── Download NLTK data (only runs on first use) ───────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Load HuggingFace sentiment pipeline ──────────────────────────────────────
print("\n🔄 Loading sentiment model (DistilBERT)... this may take a moment on first run.")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("✅ Model loaded.\n")


# ── Preprocessing with NLTK ──────────────────────────────────────────────────
def preprocess(text: str) -> dict:
    """Tokenize and remove stopwords using NLTK. Returns tokens + cleaned text."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    cleaned_text = " ".join(filtered_tokens)
    return {
        "original": text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "cleaned_text": cleaned_text,
    }


# ── Sentiment prediction with HuggingFace ────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """Run full pipeline: preprocess → predict → return result."""
    processed = preprocess(text)

    # DistilBERT works best on the original text (it handles context well)
    # We pass the original, but show cleaned tokens as a transparency layer
    result = sentiment_pipeline(processed["original"])[0]

    label = result["label"]          # POSITIVE or NEGATIVE
    score = round(result["score"] * 100, 2)

    # Map to a simple emoji for readability
    emoji = "😊" if label == "POSITIVE" else "😞"

    return {
        "label": label,
        "confidence": score,
        "emoji": emoji,
        "filtered_tokens": processed["filtered_tokens"],
        "cleaned_text": processed["cleaned_text"],
    }


# ── Display result ────────────────────────────────────────────────────────────
def display_result(text: str, result: dict):
    print("\n" + "─" * 50)
    print(f"  Input Text   : {text}")
    print(f"  Cleaned Text : {result['cleaned_text']}")
    print(f"  Key Tokens   : {result['filtered_tokens']}")
    print(f"  Sentiment    : {result['emoji']}  {result['label']}")
    print(f"  Confidence   : {result['confidence']}%")
    print("─" * 50)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("   NLP Sentiment Analyzer")
    print("   NLTK + HuggingFace Transformers")
    print("=" * 50)
    print("Type any sentence to analyze its sentiment.")
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
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Bye!")
            break


if __name__ == "__main__":
    main()