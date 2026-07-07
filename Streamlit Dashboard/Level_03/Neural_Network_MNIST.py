import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import pickle


# -----------------------
# 1. Load dataset
# -----------------------

df = pd.read_csv("../datasets/3) Sentiment dataset.csv")

# Keep required columns
df = df[['Text', 'Sentiment']].dropna()

print("Dataset Shape:", df.shape)


# -----------------------
# 2. Clean labels
# -----------------------

df['Sentiment'] = (
    df['Sentiment']
    .astype(str)
    .str.lower()
    .str.strip()
)

# Keep only positive and negative
df = df[
    df['Sentiment'].isin(
        ['positive', 'negative']
    )
]


# Convert labels

df['Sentiment'] = df['Sentiment'].map({
    'positive': 1,
    'negative': 0
})


print("\nClass Distribution:")
print(df['Sentiment'].value_counts())


# -----------------------
# 3. Features and Target
# -----------------------

X = df['Text']
y = df['Sentiment']


# -----------------------
# 4. TF-IDF Vectorization
# -----------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)


X_vec = tfidf.fit_transform(X).toarray()


print("\nFeature Shape:", X_vec.shape)


# -----------------------
# 5. Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# -----------------------
# 6. Neural Network Architecture
# -----------------------

model = Sequential([

    Input(
        shape=(X_train.shape[1],)
    ),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.3),


    Dense(
        64,
        activation='relu'
    ),

    Dropout(0.2),


    Dense(
        32,
        activation='relu'
    ),


    Dense(
        1,
        activation='sigmoid'
    )
])


# Model information

model.summary()


# -----------------------
# 7. Compile Model
# -----------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# -----------------------
# 8. Early Stopping
# -----------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)



# -----------------------
# 9. Train Model
# -----------------------

history = model.fit(
    X_train,
    y_train,

    validation_split=0.2,

    epochs=30,

    batch_size=32,

    callbacks=[
        early_stop
    ],

    verbose=1
)



# -----------------------
# 10. Evaluation
# -----------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)


print("\nTest Accuracy:", accuracy)



# -----------------------
# 11. Predictions
# -----------------------

y_probability = model.predict(
    X_test
)


y_pred = (
    y_probability > 0.5
).astype("int32")



# -----------------------
# 12. Performance Metrics
# -----------------------

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


auc = roc_auc_score(
    y_test,
    y_probability
)



print("\nPrecision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

print("ROC-AUC:", auc)



print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# -----------------------
# 13. Confusion Matrix
# -----------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Negative",
        "Positive"
    ],
    yticklabels=[
        "Negative",
        "Positive"
    ]
)


plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "Neural Network Confusion Matrix"
)

plt.show()



# -----------------------
# 14. Training Accuracy Plot
# -----------------------

plt.figure(figsize=(7,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)


plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)


plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel(
    "Epochs"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid()

plt.show()



# -----------------------
# 15. Loss Plot
# -----------------------

plt.figure(figsize=(7,5))


plt.plot(
    history.history['loss'],
    label='Training Loss'
)


plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)


plt.title(
    "Training vs Validation Loss"
)

plt.xlabel(
    "Epochs"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid()

plt.show()



# -----------------------
# 16. Save Model
# -----------------------

model.save(
    "sentiment_nn_model.h5"
)


with open(
    "tfidf_vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tfidf,
        file
    )


print("\nModel and TF-IDF saved successfully!")

# -----------------------
# 17. Test New Sentence
# -----------------------

def predict_sentiment(text):

    text_vector = tfidf.transform(
        [text]
    ).toarray()


    prediction = model.predict(
        text_vector
    )


    if prediction[0][0] >= 0.5:
        return "Positive 😊"

    else:
        return "Negative 😞"


print(
    predict_sentiment(
        "This movie was amazing and I loved it"
    )
)