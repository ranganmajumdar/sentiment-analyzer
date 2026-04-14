
from transformers import pipeline


# ---------- Load Model (once, at startup) ----------

def load_model():
    """
    Loads a pre-trained DistilBERT model fine-tuned on SST-2
    (Stanford Sentiment Treebank). Returns a HuggingFace pipeline object.
    """
    print("⏳ Loading model... (first run downloads ~260MB, cached after)")
    classifier = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("✅ Model loaded.\n")
    return classifier


# ----------- Core Analysis -----------

def analyze(classifier, text):
    """
    Runs sentiment analysis on a single string.
    Returns label ('POSITIVE'/'NEGATIVE') and confidence score (0–1).
    """
    result = classifier(text, truncation=True, max_length=512)[0]
    label  = result["label"]
    score  = result["score"]
    return label, score


# ---------- Display ----------

def display_result(text, label, score):
    bar_len  = int(score * 30)
    bar      = "█" * bar_len + "░" * (30 - bar_len)
    emoji    = "😊" if label == "POSITIVE" else "😞"

    print("\n" + "─" * 55)
    print(f"  Text       : {text}")
    print(f"  Sentiment  : {emoji} {label}")
    print(f"  Confidence : [{bar}] {score:.1%}")
    print("─" * 55)


# ---------- Batch Mode ----------

def analyze_batch(classifier, texts):
    """Analyze a list of texts. Returns list of result dicts."""
    results = []
    for text in texts:
        label, score = analyze(classifier, text)
        results.append({"text": text, "label": label, "score": score})
        display_result(text, label, score)
    return results


# ---------- Summary ----------

def summary(results):
    total    = len(results)
    positive = sum(1 for r in results if r["label"] == "POSITIVE")
    negative = total - positive
    avg_conf = sum(r["score"] for r in results) / total

    print("\n📊 BATCH SUMMARY")
    print("─" * 40)
    print(f"  Total      : {total}")
    print(f"  Positive   : {positive} {'🟢' * positive}")
    print(f"  Negative   : {negative} {'🔴' * negative}")
    print(f"  Avg Confidence: {avg_conf:.1%}")
    print("─" * 40)


# ---------- Interactive Mode ----------

def interactive_mode(classifier):
    print("\n🔍 Interactive Mode — type a sentence, Enter to analyze.")
    print("   Type 'quit' to exit.\n")

    while True:
        text = input(">> ").strip()

        if text.lower() == "quit":
            print("Exiting. Goodbye!")
            break

        if not text:
            print("  (empty input, try again)")
            continue

        label, score = analyze(classifier, text)
        display_result(text, label, score)


# ---------- Main ----------

if __name__ == "__main__":

    clf = load_model()

    # --- 1. Single sentence ---
    print("─── Single Sentence Demo ───")
    text  = "I absolutely loved every part of this experience!"
    label, score = analyze(clf, text)
    display_result(text, label, score)

    # --- 2. Batch ---
    print("\n─── Batch Demo ───")
    reviews = [
        "The product broke after one day. Total waste of money.",
        "Decent quality but nothing extraordinary.",
        "Best purchase I've made this year. Highly recommended!",
        "Terrible customer support. Never buying again.",
        "Works exactly as described. Very happy with it.",
        ]
    results = analyze_batch(clf, reviews)
    summary(results)

    # --- 3. Interactive ---
    interactive_mode(clf)