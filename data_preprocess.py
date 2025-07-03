import pandas as pd
import re

# Load original file (if not loaded already)
df = pd.read_csv("D:\\Sentimental Analysis\\dataset_uncleaned.csv", encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Optional: Convert target labels to readable form
df['target'] = df['target'].replace({0: 'negative', 2: 'neutral', 4: 'positive'})

# Cleaning function
def clean_tweet(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'#\w+', '', text)  # remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_tweet)

# Save to new CSV
df[['target', 'clean_text']].to_csv("D:\\Sentimental Analysis\\dataset_cleaned.csv", index=False)

print("✅ Cleaned dataset saved as 'dataset_cleaned.csv'")
