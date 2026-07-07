import streamlit as st
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Neural Network Sentiment Dashboard",
    layout="wide"
)

st.title("🧠 Sentiment Analysis Dashboard - Neural Network")
st.write(
    "Deep Learning based sentiment classification using TensorFlow/Keras"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    BASE_DIR = os.path.dirname(
        os.path.dirname(__file__)
    )

    df = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "datasets",
            "3) Sentiment dataset.csv"
        )
    )

    df = df[
        ['Text','Sentiment']
    ].dropna()

    df['Sentiment'] = (
        df['Sentiment']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df[
        df['Sentiment']
        .isin(
            ['positive','negative']
        )
    ]

    df['Sentiment'] = df['Sentiment'].map(
        {
            "positive":1,
            "negative":0
        }
    )

    return df

df = load_data()

# -----------------------------
# Dataset Section
# -----------------------------

st.subheader("📂 Dataset Overview")

col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "Total Reviews",
        df.shape[0]
    )

with col2:
    st.metric(
        "Features",
        "Text"
    )

with col3:
    st.metric(
        "Classes",
        "2"
    )

if st.checkbox("Show Dataset"):
    st.dataframe(
        df.head(10)
    )

# -----------------------------
# Class Distribution
# -----------------------------
st.subheader(
    "📊 Sentiment Distribution"
)

fig,ax = plt.subplots(
    figsize=(6,4)
)

df['Sentiment'].value_counts().plot(
    kind='bar',
    ax=ax
)

ax.set_xticklabels(
    ["Negative","Positive"],
    rotation=0
)

ax.set_ylabel(
    "Count"
)

st.pyplot(fig)

# -----------------------------
# Train Button
# -----------------------------
train = st.button(
    "🚀 Train Neural Network"
)

if train:
    X = df['Text']
    y = df['Sentiment']

    # TF-IDF

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )

    X_vec = tfidf.fit_transform(
        X
    ).toarray()

    X_train,X_test,y_train,y_test = train_test_split(
        X_vec,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Model
    # -----------------------------

    model = Sequential([
        Input(
            shape=(X_train.shape[1],)
        ),
        Dense(
            128,
            activation="relu"
        ),
        Dropout(0.3),
        Dense(
            64,
            activation="relu"
        ),
        Dropout(0.2),
        Dense(
            32,
            activation="relu"
        ),
        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    with st.spinner(
        "Training Neural Network..."
    ):
        history = model.fit(
            X_train,
            y_train,

            validation_split=0.2,

            epochs=30,

            batch_size=32,

            callbacks=[
                early_stop
            ],

            verbose=0
        )

    # -----------------------------
    # Metrics
    # -----------------------------

    loss,accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    probability = model.predict(
        X_test
    )

    y_pred = (
        probability > 0.5
    ).astype(int)

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
        probability
    )

    st.success(
        "Training Completed!"
    )

    st.subheader(
        "📈 Model Performance"
    )

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    c2.metric(
        "Precision",
        f"{precision:.3f}"
    )