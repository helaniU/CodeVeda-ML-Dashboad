import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# PAGE CONFIG (must be first Streamlit call)
# ==========================================================
st.set_page_config(
    page_title="Forest Console | Random Forest House Price Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# THEME: "Forest Console" — modern dark dashboard, emerald/slate accents
# CSS-ONLY styling. No ML logic below has been changed.
# ==========================================================
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

/* Native Streamlit headings — kept as real elements so they never disappear */
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

/* Panels around dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--emerald);
    border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    background: var(--emerald);
    color: #04140a;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 8px 22px;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: var(--emerald-soft);
    color: #04140a;
    transform: translateY(-1px);
}

/* Number inputs */
div[data-testid="stNumberInput"] input {
    background-color: var(--panel-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stNumberInput"] label {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
}

/* Success alert */
.stSuccess {
    background: rgba(34, 197, 94, 0.1) !important;
    border: 1px solid var(--emerald) !important;
    border-radius: 10px !important;
}

/* Checkbox */
label[data-testid="stWidgetLabel"] p {
    color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}

/* Divider line */
hr { border-color: var(--border) !important; }

/* Card wrapper used around charts to keep them compact & framed */
.chart-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 14px 2px 14px;
    margin-bottom: 6px;
}

/* Badge row under title */
.badge-row { display: flex; gap: 10px; margin-bottom: 22px; flex-wrap: wrap; }
.badge {
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}
.badge b { color: var(--emerald-soft); }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# MATPLOTLIB THEME (cosmetic only — no data/logic changes)
# ==========================================================
plt.rcParams.update({
    "figure.facecolor": "#11161d",
    "axes.facecolor": "#151b23",
    "axes.edgecolor": "#232b36",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b98a9",
    "ytick.color": "#8b98a9",
    "text.color": "#e6edf3",
    "axes.titlecolor": "#86efac",
    "grid.color": "#232b36",
    "font.family": "sans-serif",
    "font.size": 8,
})
EMERALD = "#22c55e"
EMERALD_SOFT = "#86efac"
AMBER = "#f0a020"

def chart(fig):
    """Render a matplotlib figure inside a compact framed card, small & consistent size."""
    #st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("🌲 Random Forest Regression Console")
st.markdown("""
<div class="badge-row">
    <span class="badge">MODEL <b>RandomForestRegressor</b></span>
    <span class="badge">TASK <b>House Price Prediction</b></span>
    <span class="badge">DATA <b>Boston-style Housing Set</b></span>
</div>
""", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX",
    "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT", "MEDV"
]

df = pd.read_csv(
    r"D:\CodeVeda ML Project\datasets\4) house Prediction Data Set.csv",
    sep=r"\s+",
    header=None,
    names=columns
)

# -------------------------
# OUTLIER REMOVAL
# -------------------------
def remove_outliers(df, cols):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

df = remove_outliers(df, df.columns)

st.subheader("📊 Dataset After Cleaning")
st.dataframe(df.head(), use_container_width=True)

# -------------------------
# FEATURE ENGINEERING
# -------------------------
df["RM_LSTAT"] = df["RM"] * df["LSTAT"]
df["TAX_RAD"] = df["TAX"] * df["RAD"]
df["DIS_NOX"] = df["DIS"] * df["NOX"]

# -------------------------
# SPLIT DATA
# -------------------------
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL
# -------------------------
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# -------------------------
# METRICS
# -------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.2f}")
m2.metric("RMSE", f"{rmse:.2f}")
m3.metric("R² Score", f"{r2:.2f}")

# -------------------------
# FEATURE IMPORTANCE  +  CROSS VALIDATION  (side-by-side, compact)
# -------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔥 Feature Importance")
    importances = rf_model.feature_importances_
    features = X.columns

    fig, ax = plt.subplots(figsize=(4, 3.2), dpi=130)
    ax.barh(features, importances, color=EMERALD)
    ax.set_title("Feature Importance", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    chart(fig)

with col_right:
    st.subheader("🔁 Cross Validation (R²)")
    cv_scores = cross_val_score(rf_model, X, y, cv=10, scoring="r2")

    cv1, cv2 = st.columns(2)
    cv1.metric("CV Mean", f"{np.mean(cv_scores):.3f}")
    cv2.metric("CV Std", f"{np.std(cv_scores):.3f}")

    fig, ax = plt.subplots(figsize=(4, 3.2), dpi=130)
    ax.plot(range(1, len(cv_scores) + 1), cv_scores, marker="o", color=EMERALD_SOFT)
    ax.axhline(np.mean(cv_scores), color=AMBER, linestyle="--", linewidth=1)
    ax.set_title("CV Fold Scores", fontsize=9, fontweight="bold")
    ax.set_xlabel("Fold", fontsize=7)
    ax.set_ylabel("R²", fontsize=7)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    chart(fig)

# -------------------------
# PREDICTION UI
# -------------------------
st.header("🏠 Predict House Price")

input_data = []
columns_ui = st.columns(2, gap="small")

for i, col in enumerate(X.columns):
    with columns_ui[i % 2]:
        val = st.number_input(f"{col}", value=float(df[col].mean()), key=f"{col}_input")
        input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

if st.button("🔮 Predict Price"):
    prediction = rf_model.predict(input_array)
    st.success(f"🏡 Predicted House Price: **{prediction[0]:.2f}**")

# -------------------------
# OPTIONAL: TUNING (BUTTON)
# -------------------------
if st.checkbox("⚙️ Run Hyperparameter Tuning (Slow)"):

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    st.write("Best Params:", search.best_params_)
    st.write("Best Score:", search.best_score_)

# -------------------------
# Actual vs Predicted + Residuals (side-by-side, compact)
# -------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=130)
    ax.scatter(y_test, y_pred, alpha=0.75, color=EMERALD_SOFT, edgecolors=EMERALD, linewidths=0.3, s=18)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle='--', color=AMBER, linewidth=1
    )
    ax.set_xlabel("Actual Price", fontsize=7)
    ax.set_ylabel("Predicted Price", fontsize=7)
    ax.set_title("Actual vs Predicted", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    chart(fig)

with col_b:
    st.subheader("📉 Residual Plot")
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=130)
    ax.scatter(y_pred, residuals, alpha=0.75, color=EMERALD_SOFT, edgecolors=EMERALD, linewidths=0.3, s=18)
    ax.axhline(0, color=AMBER, linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Residual", fontsize=7)
    ax.set_title("Residuals", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    chart(fig)

# -------------------------
# Correlation Heatmap + Distribution (side-by-side, compact)
# -------------------------
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🧭 Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(4.3, 3.8), dpi=130)
    cax = ax.imshow(corr, cmap="Greens")
    cbar = plt.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#8b98a9", labelsize=6)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=5.5)
    ax.set_title("Feature Correlation", fontsize=9, fontweight="bold")
    fig.tight_layout()
    chart(fig)

with col_d:
    st.subheader("📊 House Price Distribution")
    fig, ax = plt.subplots(figsize=(4.3, 3.8), dpi=130)
    ax.hist(df["MEDV"], bins=20, color=EMERALD, edgecolor="#0b0f14")
    ax.set_title("House Price Distribution", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    chart(fig)

#Dataset Statistics Cards
st.subheader("🗂️ Dataset Overview")
c1,c2,c3,c4 = st.columns(4)
c1.metric("🏘️ Rows", len(df))
c2.metric("📑 Columns", len(df.columns))
c3.metric("🧬 Features", len(X.columns))
c4.metric("🎯 Target", "MEDV")

#Prediction Confidence
all_predictions = np.array([
    tree.predict(input_array)
    for tree in rf_model.estimators_
])

st.subheader("🎯 Prediction Confidence")
pc1, pc2 = st.columns(2)
pc1.metric("Average", f"{all_predictions.mean():.2f}")
pc2.metric("Std Dev", f"{all_predictions.std():.2f}")