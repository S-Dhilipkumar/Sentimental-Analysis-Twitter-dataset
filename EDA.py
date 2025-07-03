import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("D:\\Sentimental Analysis\\dataset_cleaned.csv")

# Set a clean style
sns.set(style="whitegrid")

# Plot sentiment counts
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='viridis')

# Add titles and labels
plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Tweet Count", fontsize=12)
plt.tight_layout()
plt.show()


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Group tweets by sentiment
positive_tweets = df[df['target'] == 'positive']['clean_text']
negative_tweets = df[df['target'] == 'negative']['clean_text']
neutral_tweets  = df[df['target'] == 'neutral']['clean_text']

# Create word clouds
def generate_wordcloud(text_series, color, title):
    try:
        text_series = text_series.dropna().astype(str)
        combined_text = " ".join(text_series)

        if not combined_text.strip():
            raise ValueError("No text data found.")

        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=color).generate(combined_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.show()

    except Exception as e:
        print(f"⚠️ Could not generate word cloud for '{title}': {e}")



# Plot each sentiment
generate_wordcloud(positive_tweets, 'Greens', '🌟 Positive Tweets Word Cloud')
generate_wordcloud(negative_tweets, 'Reds', '💢 Negative Tweets Word Cloud')
generate_wordcloud(neutral_tweets, 'Blues', '😐 Neutral Tweets Word Cloud')
