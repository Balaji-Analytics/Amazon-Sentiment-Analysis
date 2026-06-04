import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load first 100 reviews
df = pd.read_csv("Amazon_Reviews.csv", engine="python")
df = df.head(100)

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Review Text"].apply(get_sentiment)

print(df[["Rating", "Review Title", "Sentiment"]].head(10))

print("\nSentiment Summary:")
print(df["Sentiment"].value_counts())

# Rating vs Sentiment
print("\nRating vs Sentiment")

rating_sentiment = pd.crosstab(
    df["Rating"],
    df["Sentiment"]
)

print(rating_sentiment)

rating_sentiment.plot(kind="bar", figsize=(10,5))

plt.title("Rating vs Sentiment")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")

plt.show()

df.to_csv("amazon_sentiment_results.csv", index=False)

print("Results saved successfully!")
