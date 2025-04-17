# 01_data_preprocessing.py

import pandas as pd
import re
import string

# Load the dataset
df = pd.read_csv('/home/admin1/MLProjects/customer_feedback_sentiments/data/large_sentiment_feedback_data.csv', encoding='latin-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Use only text and sentiment
df = df[['text', 'sentiment']]

# Convert sentiment: 0 → negative, 4 → positive
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})  # No neutral in this dataset

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove links
    text = re.sub(r'\@\w+|\#','', text)  # remove mentions and hashtags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuations
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# After cleaning text, save the preprocessed dataframe to a new CSV
df.to_csv('/home/admin1/MLProjects/customer_feedback_sentiments/data/cleaned_sentiment_data.csv', index=False)


print(df.head())

