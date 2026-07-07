import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="ML Dashboard", layout="wide", page_icon="🌿")

# =========================
# 🎨 DARK THEME — SLATE + SUBTLE GREEN ACCENTS (UI ONLY)
# =========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: #12151a;
        background-image: radial-gradient(circle at 15% 0%, rgba(45, 212, 149, 0.06), transparent 40%),
                           radial-gradient(circle at 85% 100%, rgba(45, 212, 149, 0.05), transparent 40%);
        color: #d7dbe0;
    }

    section[data-testid="stSidebar"] {
        background: #171b21;
        border-right: 1px solid #232830;
    }
    section[data-testid="stSidebar"] * {
        color: #cfd6dd !important;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #2dd495 !important;
    }

    .app-banner {
        background: #171b21;
        border: 1px solid #232830;
        border-left: 4px solid #2dd495;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .app-banner h1 {
        margin: 0 !important;
        font-size: 26px !important;
        color: #f1f3f5 !important;
        font-weight: 700 !important;
    }
    .app-banner span.tag {
        color: #2dd495;
        font-size: 13px;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid #2dd495;
        padding: 4px 10px;
        border-radius: 20px;
    }

    h2, h3, h4 {
        color: #e7e9ec !important;
        font-weight: 600 !important;
    }

    div[data-testid="column"] > div {
        background: #171b21;
        border: 1px solid #232830;
        border-radius: 12px;
        padding: 20px;
    }
    div[data-testid="column"] {
        padding: 6px;
    }

    .metric-card {
        background: linear-gradient(135deg, #17332a, #171b21);
        border: 1px solid #2dd495;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-top: 8px;
    }
    .metric-card .label {
        font-size: 13px;
        color: #9fb0a8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 34px;
        font-weight: 700;
        color: #2dd495;
        font-family: 'JetBrains Mono', monospace;
    }

    .predict-card {
        background: #171b21;
        border: 1px solid #232830;
        border-left: 4px solid #2dd495;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .predict-card .predict-label {
        font-size: 13px;
        color: #9fb0a8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .predict-card .predict-value {
        font-size: 24px;
        font-weight: 700;
        color: #2dd495;
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
        background: #232830;
        border-radius: 6px;
        overflow: hidden;
        height: 10px;
    }
    .conf-bar-fill {
        background: linear-gradient(90deg, #1a7a55, #2dd495);
        height: 100%;
        border-radius: 6px;
    }
    .conf-pct {
        width: 46px;
        text-align: right;
        color: #2dd495;
    }

    div[data-testid="stAlertContainer"] {
        background: #17332a !important;
        color: #d7f7e6 !important;
        border-radius: 10px !important;
        border: 1px solid #2dd495 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #232830;
        border-radius: 10px;
        overflow: hidden;
    }

    pre, code, .stText, div[data-testid="stText"] {
        background-color: #10141a !important;
        color: #b7c2ba !important;
        border-radius: 8px !important;
        border: 1px solid #232830 !important;
        padding: 12px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    div[data-testid="stSlider"] > div > div > div > div {
        background-color: #2dd495 !important;
    }
    .stSlider [role="slider"] {
        background-color: #2dd495 !important;
        box-shadow: 0 0 6px rgba(45, 212, 149, 0.6) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #171b21 !important;
        border-radius: 8px !important;
        border: 1px solid #232830 !important;
        color: #d7dbe0 !important;
    }

    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: #9fb0a8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
        border-bottom: 1px solid #232830;
        padding-bottom: 8px;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #12151a; }
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
fig.patch.set_facecolor('#12151a')
ax.set_facecolor('#12151a')
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    ax=ax
)

st.pyplot(fig)