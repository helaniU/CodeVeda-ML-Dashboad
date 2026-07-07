import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# -----------------------
# CACHE MODEL TRAINING
# -----------------------
@st.cache_resource
def train_models(_X_train, _y_train):
    svm_linear = SVC(kernel='linear', probability=True)
    svm_rbf = SVC(kernel='rbf', probability=True)

    svm_linear.fit(_X_train, _y_train)
    svm_rbf.fit(_X_train, _y_train)

    return svm_linear, svm_rbf

# -----------------------
# UI CONFIG
# -----------------------
st.set_page_config(page_title="SVM Sentiment Dashboard", page_icon="🧠", layout="wide")

# -----------------------
# THEME / CSS (visual only — no logic changes below this block)
# -----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --primary: #6C63FF;
    --primary-dark: #4B3FE0;
    --accent: #00C2A8;
    --bg: #121826;          /* 🔥 DARK MAIN BACKGROUND */
    --panel: #1A2233;       /* cards */
    --text: #E8EAF6;        /* main text */
    --muted: #A7B0C0;
}

/* ---------------- MAIN APP BACKGROUND ---------------- */
.stApp {
    background: linear-gradient(180deg, #121826 0%, #0F141F 100%) !important;
    color: var(--text);
}

/* ---------------- TEXT VISIBILITY ---------------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text) !important;
}

/* headings */
h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text) !important;
}

/* title gradient (keep but readable) */
h1 {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---------------- METRICS ---------------- */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid #2A3350;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ---------------- INPUT BOX ---------------- */
textarea {
    background-color: #151C2B !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #2A3350 !important;
}

/* placeholder text */
textarea::placeholder {
    color: #8FA3C8 !important;
}

/* ---------------- BUTTON ---------------- */
.stButton > button {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    color: white !important;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
}

/* ---------------- DATAFRAME ---------------- */
[data-testid="stDataFrame"] {
    background: var(--panel);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #2A3350;
}

/* ---------------- CHART CARD ---------------- */
.chart-card {
    background: var(--panel);
    border-radius: 14px;
    padding: 14px;
    border: 1px solid #2A3350;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

/* ---------------- TEXT FIX ---------------- */
p, span, div {
    color: var(--text);
}

hr {
    border-color: #2A3350;
}
</style>
""", unsafe_allow_html=True)

# Matplotlib palette to match theme
sns.set_theme(style="whitegrid")
THEME_PRIMARY = "#6C63FF"
THEME_ACCENT = "#00C2A8"
THEME_DARK = "#2B2640"
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#D8D3FF",
    "axes.labelcolor": THEME_DARK,
    "text.color": THEME_DARK,
    "xtick.color": THEME_DARK,
    "ytick.color": THEME_DARK,
    "axes.titleweight": "600",
})

# -----------------------
# HEADER
# -----------------------
st.title("🧠 Sentiment Analysis Dashboard")
st.markdown(
    "<p style='color:#6B6483; font-size:16px; margin-top:-10px;'>"
    "🔍 Comparing <b>Linear</b> vs <b>RBF</b> SVM kernels for text sentiment classification"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)


# -----------------------
# LOAD DATA
# -----------------------
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "3) Sentiment dataset.csv")

df = pd.read_csv(DATA_PATH)

df = df[['Text', 'Sentiment']].dropna()

df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()
df = df[df['Sentiment'].isin(['positive', 'negative'])]

df['Sentiment'] = df['Sentiment'].map({
    'positive': 1,
    'negative': 0
})


# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.markdown("## ⚙️ Controls")
st.sidebar.markdown("---")

show_data = st.sidebar.checkbox("📄 Show Dataset")
train_model = st.sidebar.button("🚀 Train Models")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:13px; color:#C9C3F5;'>💡 Tip: Train the models first, "
    "then try the live predictor below.</p>",
    unsafe_allow_html=True
)


# -----------------------
# DATA PREVIEW
# -----------------------
if show_data:
    st.markdown("### 📄 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# -----------------------
# FEATURE ENGINEERING
# -----------------------
X = df['Text']
y = df['Sentiment']

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)


# -----------------------
# TRAIN MODELS
# -----------------------
if train_model:

    st.markdown("### 🧠 Training Models…")
    with st.spinner("Fitting Linear & RBF SVMs…"):
        svm_linear, svm_rbf = train_models(X_train, y_train)

    y_pred_l = svm_linear.predict(X_test)
    y_pred_r = svm_rbf.predict(X_test)

    y_prob_l = svm_linear.predict_proba(X_test)[:, 1]
    y_prob_r = svm_rbf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred_l)

    # -----------------------
    # METRICS FUNCTION
    # -----------------------
    def metrics(y_true, y_pred, y_prob):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob)
        }

    linear_metrics = metrics(y_test, y_pred_l, y_prob_l)
    rbf_metrics = metrics(y_test, y_pred_r, y_prob_r)

    # -----------------------
    # METRICS DISPLAY
    # -----------------------
    st.markdown("### 📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📐 Linear Accuracy", f"{linear_metrics['Accuracy']:.2f}")
    col2.metric("🌀 RBF Accuracy", f"{rbf_metrics['Accuracy']:.2f}")
    col3.metric("📐 Linear F1", f"{linear_metrics['F1']:.2f}")
    col4.metric("🌀 RBF F1", f"{rbf_metrics['F1']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------
    # ACCURACY COMPARISON + CONFUSION MATRIX (side by side, smaller)
    # -----------------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📊 Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=140)
        bars = ax.bar(
            ["Linear", "RBF"],
            [linear_metrics["Accuracy"], rbf_metrics["Accuracy"]],
            color=[THEME_PRIMARY, THEME_ACCENT],
            width=0.5
        )
        ax.set_ylim(0, 1)
        #ax.set_title("Accuracy Comparison", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.subheader("🔥 Confusion Matrix (Linear SVM)")
        fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=140)
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap=sns.light_palette(THEME_ACCENT, as_cmap=True),
            ax=ax, cbar=False, annot_kws={"size": 9}
        )
        #ax.set_title("Linear SVM Confusion Matrix", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# LIVE PREDICTION
# -----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### ✍️ Live Sentiment Prediction")

user_text = st.text_area("Enter a sentence:", placeholder="e.g. This product completely exceeded my expectations!")

if st.button("🔮 Predict Sentiment"):

    if user_text.strip() == "":
        st.warning("⚠️ Please enter text")

    else:
        sample = tfidf.transform([user_text])

        # train simple model if not trained
        svm_linear, _ = train_models(X_train, y_train)

        model = svm_linear

        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]

        st.markdown("#### 🧾 Result")

        if pred == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😠 Negative Sentiment")

        st.progress(int(prob * 100))
        st.caption(f"Confidence: {prob:.2%}")