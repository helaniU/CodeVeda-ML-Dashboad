import streamlit as st
import pandas as pd
import numpy as np
import os

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
    confusion_matrix
)


# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="SVM Sentiment Dashboard",
    page_icon="🧠",
    layout="wide"
)


# -----------------------
# CSS THEME
# -----------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:#0b0f14;
    --panel:#11161d;
    --panel-2:#151b23;
    --text:#e6edf3;
    --muted:#8b98a9;
    --border:#243044;
    --blue:#4aa3ff;
    --blue-soft:#a9d8ff;
}


.stApp {
    background: linear-gradient(180deg, #0b0f14 0%, #0f141b 100%);
}


html,body,[class*="css"] {
    font-family:'Manrope',sans-serif;
    color:var(--text)!important;
}


h1,h2,h3,h4 {
    font-family:'Manrope',sans-serif!important;
    color:var(--text)!important;
}

h1 {
    background:linear-gradient(90deg, #4aa3ff, #8fd3ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}


div[data-testid="stMetric"] {

    background:var(--panel);
    border:1px solid var(--border);
    border-radius:14px;
    padding:15px;
    box-shadow: 0 8px 24px rgba(74, 163, 255, 0.08);

}


div[data-testid="stMetricValue"]{
    color:var(--text)!important;
}


div[data-testid="stMetricLabel"]{
    color:var(--muted)!important;
}


textarea {

    background:var(--panel)!important;
    color:var(--text)!important;
    border:1px solid var(--border)!important;
    border-radius:12px!important;

}


.stButton button {

    background:linear-gradient(90deg, #4aa3ff, #8fd3ff);

    color:#04111f!important;
    border-radius:10px;
    border:none;
    font-weight:600;
    box-shadow: 0 8px 18px rgba(74, 163, 255, 0.22);

}


div[data-testid="stDataFrame"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
}


div[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: var(--blue-soft) !important;
}


.app-banner {
    background: linear-gradient(135deg, #11161d, #151b23);
    border: 1px solid var(--border);
    border-left: 4px solid var(--blue);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 24px rgba(74, 163, 255, 0.08);
}

.app-banner h1 {
    margin: 0 !important;
    font-size: 34px !important;
    color: #e6edf3 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid var(--blue);
    padding-bottom: 12px;
}

.app-banner span.tag {
    color: var(--blue-soft);
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid rgba(143, 211, 255, 0.65);
    padding: 4px 10px;
    border-radius: 20px;
    background: rgba(74, 163, 255, 0.08);

}

</style>

""", unsafe_allow_html=True)



# -----------------------
# MODEL TRAINING
# -----------------------
@st.cache_resource
def train_models(_X_train, _y_train):

    svm_linear = SVC(
        kernel="linear",
        probability=True,
        decision_function_shape="ovr"
    )

    svm_rbf = SVC(
        kernel="rbf",
        probability=True,
        decision_function_shape="ovr"
    )


    svm_linear.fit(
        _X_train,
        _y_train
    )

    svm_rbf.fit(
        _X_train,
        _y_train
    )


    return svm_linear, svm_rbf

# -----------------------
# HEADER
# -----------------------
st.markdown("""
<div class="app-banner">
    <h1>🧠 Sentiment Analysis Dashboard</h1>
    <span class="tag">SENTIMENT DATASET</span>
</div>
""", unsafe_allow_html=True)

# -----------------------
# LOAD DATA
# -----------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "3) Sentiment dataset.csv"
)



df = pd.read_csv(DATA_PATH)



df = df[
    [
        "Text",
        "Sentiment"
    ]
].dropna()



df["Sentiment"] = (
    df["Sentiment"]
    .astype(str)
    .str.strip()
    .str.lower()
)



# -----------------------
# EMOTION MAPPING
# -----------------------


positive_words = [

"positive",
"happy",
"happiness",
"joy",
"joyful",
"love",
"amusement",
"enjoyment",
"admiration",
"affection",
"euphoria",
"contentment",
"gratitude",
"hope",
"pride",
"excitement",
"confidence",
"serenity",
"success",
"accomplishment",
"inspiration",
"enthusiasm",
"wonder",
"adventure",
"celebration",
"triumph",
"optimism",
"curiosity"

]



negative_words = [

"negative",
"sad",
"sadness",
"anger",
"fear",
"disgust",
"despair",
"grief",
"loneliness",
"frustration",
"anxiety",
"jealousy",
"regret",
"heartbreak",
"sorrow",
"betrayal",
"suffering",
"isolation",
"hate",
"bad",
"disappointed",
"disappointment",
"bitterness"

]



neutral_words = [

"neutral",
"confusion",
"curiosity",
"indifference",
"nostalgia",
"reflection",
"contemplation",
"ambivalence",
"acceptance",
"determination",
"calmness"

]




def sentiment_converter(value):

    if value in positive_words:
        return "Positive"

    elif value in negative_words:
        return "Negative"

    elif value in neutral_words:
        return "Neutral"

    else:
        return None



df["Sentiment"] = df["Sentiment"].apply(
    sentiment_converter
)


df = df.dropna()



label_map = {

    "Negative":0,
    "Neutral":1,
    "Positive":2

}


df["Sentiment"] = df["Sentiment"].map(
    label_map
)



# -----------------------
# SIDEBAR
# -----------------------

st.sidebar.title("⚙️ Controls")

show_data = st.sidebar.checkbox(
    "📄 Show Dataset"
)


train_button = st.sidebar.button(
    "🚀 Train Models"
)



# -----------------------
# DATA PREVIEW
# -----------------------

if show_data:

    st.subheader(
        "📄 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.caption(
        f"Dataset Shape : {df.shape}"
    )



# -----------------------
# FEATURES
# -----------------------


X = df["Text"]

y = df["Sentiment"]



tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)



X_vec = tfidf.fit_transform(
    X
)



X_train,X_test,y_train,y_test = train_test_split(

    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)




# -----------------------
# TRAIN
# -----------------------


if train_button:


    with st.spinner(
        "Training SVM Models..."
    ):

        svm_linear, svm_rbf = train_models(
            X_train,
            y_train
        )



    pred_linear = svm_linear.predict(
        X_test
    )


    pred_rbf = svm_rbf.predict(
        X_test
    )



    def get_metrics(y_true,y_pred):

        return {

            "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),


            "Precision":
            precision_score(
                y_true,
                y_pred,
                average="weighted"
            ),


            "Recall":
            recall_score(
                y_true,
                y_pred,
                average="weighted"
            ),


            "F1 Score":
            f1_score(
                y_true,
                y_pred,
                average="weighted"
            )

        }



    linear_metrics = get_metrics(
        y_test,
        pred_linear
    )


    rbf_metrics = get_metrics(
        y_test,
        pred_rbf
    )



    st.subheader(
        "📊 Model Performance"
    )


    c1,c2,c3,c4 = st.columns(4)



    c1.metric(
        "Linear Accuracy",
        f"{linear_metrics['Accuracy']:.2f}"
    )


    c2.metric(
        "RBF Accuracy",
        f"{rbf_metrics['Accuracy']:.2f}"
    )


    c3.metric(
        "Linear F1",
        f"{linear_metrics['F1 Score']:.2f}"
    )


    c4.metric(
        "RBF F1",
        f"{rbf_metrics['F1 Score']:.2f}"
    )



    # -----------------------
    # CHARTS
    # -----------------------


    col1,col2 = st.columns(2)



    with col1:

        st.subheader(
            "📊 Accuracy Comparison"
        )


        fig,ax = plt.subplots(
            figsize=(4,3)
        )


        ax.bar(

            ["Linear","RBF"],

            [
                linear_metrics["Accuracy"],
                rbf_metrics["Accuracy"]
            ]

        )


        ax.set_ylim(
            0,1
        )


        st.pyplot(
            fig
        )



    with col2:


        st.subheader(
            "🔥 Confusion Matrix"
        )


        cm = confusion_matrix(
            y_test,
            pred_linear
        )


        fig,ax = plt.subplots(
            figsize=(4,3)
        )


        sns.heatmap(

            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Negative",
                "Neutral",
                "Positive"
            ],

            yticklabels=[
                "Negative",
                "Neutral",
                "Positive"
            ]

        )


        st.pyplot(
            fig
        )



# -----------------------
# LIVE PREDICTION
# -----------------------


st.divider()


st.subheader(
    "✍️ Live Sentiment Prediction"
)



text = st.text_area(
    "Enter a sentence",
    placeholder=
    "Example: This product is amazing!"
)



if st.button(
    "🔮 Predict"
):


    if text.strip()=="":

        st.warning(
            "Enter some text"
        )


    else:

        model,_ = train_models(
            X_train,
            y_train
        )


        sample = tfidf.transform(
            [text]
        )


        prediction = model.predict(
            sample
        )[0]



        result = {

            0:"😠 Negative",
            1:"😐 Neutral",
            2:"😊 Positive"

        }

        st.success(
            result[prediction]
        )