from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Setup ---
analyzer = SentimentIntensityAnalyzer()

# --- Counters for summary ---
positive_count = 0
negative_count = 0
neutral_count = 0

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        return "Positive 😊", compound
    elif compound <= -0.05:
        return "Negative 😞", compound
    else:
        return "Neutral 😐", compound

# --- Read the file ---
print("=" * 50)
print("       SENTIMENT ANALYSIS REPORT")
print("=" * 50)

with open("sentences.txt", "r") as file:
    lines = file.readlines()

# --- Analyze each sentence ---
for i, line in enumerate(lines):
    line = line.strip()
    if line == "":
        continue

    result, score = analyze_sentiment(line)

    if "Positive" in result:
        positive_count += 1
    elif "Negative" in result:
        negative_count += 1
    else:
        neutral_count += 1

    print(f"\n{i+1}. {line}")
    print(f"   Sentiment : {result}")
    print(f"   Score     : {score}")

# --- Summary ---
print("\n" + "=" * 50)
print("               SUMMARY")
print("=" * 50)
print(f"  Total Sentences : {len(lines)}")
print(f"  Positive        : {positive_count}")
print(f"  Negative        : {negative_count}")
print(f"  Neutral         : {neutral_count}")
print("=" * 50)