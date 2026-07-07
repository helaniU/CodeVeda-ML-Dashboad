import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# -----------------------
# 1. Load dataset
# -----------------------
df = pd.read_csv("../datasets/3) Sentiment dataset.csv")

# Keep only required columns
df = df[['Text', 'Sentiment']].dropna()

# -----------------------
# 2. Clean labels
# -----------------------
df['Sentiment'] = df['Sentiment'].astype(str).str.lower().str.strip()

df = df[df['Sentiment'].isin(['positive', 'negative'])]

df['Sentiment'] = df['Sentiment'].map({
    'positive': 1,
    'negative': 0
})

# -----------------------
# 3. Features & Target
# -----------------------
X = df['Text']
y = df['Sentiment']

# -----------------------
# 4. Text Vectorization (IMPORTANT)
# -----------------------
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_vec = tfidf.fit_transform(X).toarray()

# -----------------------
# 5. Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
# -----------------------
# 6. Build Neural Network (CLASSIFICATION)
# -----------------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # IMPORTANT for binary classification
])

# -----------------------
# 7. Compile
# -----------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# 8. Train
# -----------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

# -----------------------
# 9. Evaluate
# -----------------------
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

# -----------------------
# 10. Predictions
# -----------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------
# 11. Plot Accuracy
# -----------------------
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# -----------------------
# 12. Plot Loss
# -----------------------
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()