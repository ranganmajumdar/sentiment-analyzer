from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity  # score between -1 and +1

    if score > 0.1:
        return "Positive 😊", score
    elif score < -0.1:
        return "Negative 😞", score
    else:
        return "Neutral 😐", score

print("=== Sentiment Analyzer ===")
print("Type a sentence and I'll tell you if it's Positive, Negative, or Neutral.")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter text: ")
    if user_input.lower() == "quit":
        break
    result, score = analyze_sentiment(user_input)
    print(f"Sentiment: {result}  |  Score: {round(score, 2)}\n")