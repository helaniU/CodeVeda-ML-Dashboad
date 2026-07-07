import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Forest Console | Decision Tree", layout="wide", page_icon="🌲")

# =========================
# 🎨 DARK THEME — SLATE + SUBTLE GREEN ACCENTS (UI ONLY)
# =========================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #0b0f14;
    --panel: #11161d;
    --panel-2: #151b23;
    --border: #232b36;
    --emerald: #22c55e;
    --emerald-soft: #86efac;
    --text: #e6edf3;
    --muted: #8b98a9;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

h1 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    font-size: 2.3rem !important;
    border-bottom: 2px solid var(--emerald);
    padding-bottom: 14px;
    margin-bottom: 4px !important;
}
h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    color: var(--emerald-soft) !important;
}
h3 {
    font-size: 1.15rem !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px !important;
}

p, span, label, div, li {
    color: var(--text);
}
.stCaption, .muted {
    color: var(--muted) !important;
}

section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: var(--emerald-soft) !important;
}

.app-banner {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 4px solid var(--emerald);
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-banner h1 {
    margin: 0 !important;
    font-size: 38px !important;
    color: #f1f3f5 !important;
    font-weight: 700 !important;
}
.app-banner span.tag {
    color: var(--emerald);
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid var(--emerald);
    padding: 4px 10px;
    border-radius: 20px;
}

div[data-testid="column"] > div {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
}
div[data-testid="column"] {
    padding: 6px;
}

.metric-card {
    background: linear-gradient(135deg, #17332a, var(--panel));
    border: 1px solid var(--emerald);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin-top: 8px;
}
.metric-card .label {
    font-size: 13px;
    color: var(--muted);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.metric-card .value {
    font-size: 34px;
    font-weight: 700;
    color: var(--emerald);
    font-family: 'JetBrains Mono', monospace;
}

.predict-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 4px solid var(--emerald);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.predict-card .predict-label {
    font-size: 13px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.predict-card .predict-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--emerald);
    text-transform: capitalize;
}

.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.conf-name {
    width: 90px;
    color: #cfd6dd;
    text-transform: capitalize;
}
.conf-bar-bg {
    flex: 1;
    background: var(--border);
    border-radius: 6px;
    overflow: hidden;
    height: 10px;
}
.conf-bar-fill {
    background: linear-gradient(90deg, #1a7a55, var(--emerald));
    height: 100%;
    border-radius: 6px;
}
.conf-pct {
    width: 46px;
    text-align: right;
    color: var(--emerald);
}

div[data-testid="stAlertContainer"] {
    background: #17332a !important;
    color: #d7f7e6 !important;
    border-radius: 10px !important;
    border: 1px solid var(--emerald) !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

pre, code, .stText, div[data-testid="stText"] {
    background-color: #10141a !important;
    color: #b7c2ba !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    padding: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stSlider"] > div > div > div > div {
    background-color: var(--emerald) !important;
}
.stSlider [role="slider"] {
    background-color: var(--emerald) !important;
    box-shadow: 0 0 6px rgba(34, 197, 94, 0.6) !important;
}

div[data-baseweb="select"] > div {
    background-color: var(--panel) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #2a3038; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="app-banner"><h1>🌳 Decision Tree ML Dashboard</h1><span class="tag">IRIS DATASET</span></div>""", unsafe_allow_html=True)

# =========================
# 1. LOAD DATA
# =========================

@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return iris, X, y

iris, X, y = load_data()

st.sidebar.header("⚙️ Model Hyperparameters")

max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)

# =========================
# 2. TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    max_depth=max_depth,
    criterion=criterion,
    min_samples_split=min_samples_split,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

# =========================
# 3. LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# 4. DATA VISUALIZATION
# =========================
with col1:
    st.subheader("📊 Dataset Overview")
    st.write(X.head())
    st.write("Shape:", X.shape)

    st.subheader("📈 Accuracy")
    st.markdown(f"""<div class="metric-card"><div class="label">Model Accuracy</div><div class="value">{acc:.2%}</div></div>""", unsafe_allow_html=True)

with col2:
    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

# =========================
# 5. LIVE PREDICTION (IMPORTANT)
# =========================
st.sidebar.subheader("🌸 Live Prediction Input")

sl = st.sidebar.slider("Sepal Length", float(X.iloc[:,0].min()), float(X.iloc[:,0].max()))
sw = st.sidebar.slider("Sepal Width", float(X.iloc[:,1].min()), float(X.iloc[:,1].max()))
pl = st.sidebar.slider("Petal Length", float(X.iloc[:,2].min()), float(X.iloc[:,2].max()))
pw = st.sidebar.slider("Petal Width", float(X.iloc[:,3].min()), float(X.iloc[:,3].max()))

input_data = pd.DataFrame(
    [[sl, sw, pl, pw]],
    columns=iris.feature_names
)
prediction = model.predict(input_data)
proba = model.predict_proba(input_data)

class_names = iris.target_names

st.subheader("🔮 Live Prediction Result")

st.markdown(f"""<div class="predict-card"><div class="predict-label">Predicted Class</div><div class="predict-value">🌸 {class_names[prediction][0]}</div></div>""", unsafe_allow_html=True)

conf_rows = ""
for i, prob in enumerate(proba[0]):
    pct = prob * 100
    conf_rows += f'<div class="conf-row"><div class="conf-name">{class_names[i]}</div><div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%;"></div></div><div class="conf-pct">{pct:.0f}%</div></div>'

conf_html = f'<div class="predict-card"><div class="predict-label">Confidence</div>{conf_rows}</div>'
st.markdown(conf_html, unsafe_allow_html=True)

# =========================
# 6. DECISION TREE VISUALIZATION
# =========================
st.subheader("🌳 Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0b0f14')
ax.set_facecolor('#0b0f14')
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    ax=ax
)

for text in ax.texts:
    text.set_color('#0b0f14')
    text.set_fontweight('bold')

st.pyplot(fig)