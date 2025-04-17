import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the cleaned dataset
df = pd.read_csv("/home/admin1/MLProjects/customer_feedback_sentiments/data/cleaned_sentiment_data.csv", encoding='ISO-8859-1')

print(df['sentiment'].value_counts())

# Handle NaN values in the cleaned_text column
df['cleaned_text'] = df['cleaned_text'].fillna('')

# -------------------------------
# Parameters
# -------------------------------
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100  # Can be tuned later
OOV_TOKEN = "<OOV>"

# -------------------------------
# Split the data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
)

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train)

# -------------------------------
# Convert text to sequences
# -------------------------------
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# -------------------------------
# Pad the sequences
# -------------------------------
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# -------------------------------
# Save the sequences and labels
# -------------------------------
# Option 1: Save as CSV (flatten if needed for sequences)
pd.DataFrame(X_train_pad).to_csv('/home/admin1/MLProjects/customer_feedback_sentiments/data/X_train_pad.csv', index=False)
pd.DataFrame(X_test_pad).to_csv('/home/admin1/MLProjects/customer_feedback_sentiments/data/X_test_pad.csv', index=False)
pd.DataFrame(y_train).to_csv('/home/admin1/MLProjects/customer_feedback_sentiments/data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('/home/admin1/MLProjects/customer_feedback_sentiments/data/y_test.csv', index=False)

print("Data has been saved successfully!")
