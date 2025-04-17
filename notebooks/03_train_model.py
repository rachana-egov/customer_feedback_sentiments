import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# Load and Prepare Data
# -------------------------------
df = pd.read_csv("data/cleaned_sentiment_data.csv", encoding='ISO-8859-1')
df['cleaned_text'] = df['cleaned_text'].fillna('')

# -------------------------------
# Parameters
# -------------------------------
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
OOV_TOKEN = "<OOV>"
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5

# -------------------------------
# Tokenization & Padding
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# -------------------------------
# Build Model (BiLSTM)
# -------------------------------
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# -------------------------------
# Train Model (Comment out if already trained)
# -------------------------------
# history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping]
)

# -------------------------------
# Evaluate
# -------------------------------
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# Save Model and Tokenizer
# -------------------------------
model.save("models/sentiment_model.h5")
print("✅ Model saved to models/sentiment_model.h5")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved to models/tokenizer.pkl")
