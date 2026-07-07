import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------
# 1. Load dataset
# -----------------------
df = pd.read_csv("../datasets/3) Sentiment dataset.csv")

# Keep only useful columns
df = df[['Text', 'Sentiment']].dropna()

print("Dataset shape:", df.shape)

# -----------------------
# CLEAN SENTIMENT LABELS
# -----------------------

df = df[['Text', 'Sentiment']].dropna()

# normalize text labels
df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()

# map safely
df = df[df['Sentiment'].isin(['positive', 'negative'])]

df['Sentiment'] = df['Sentiment'].map({
    'positive': 1,
    'negative': 0
})

# -----------------------
# 3. Features & Labels
# -----------------------
X = df['Text']
y = df['Sentiment']

# -----------------------
# 4. TF-IDF Vectorization
# -----------------------
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_vec = tfidf.fit_transform(X)

# -----------------------
# 5. Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# -----------------------
# 6. Train SVM Models
# -----------------------

# Linear SVM (best for text)
svm_linear = SVC(kernel='linear', probability=True)
svm_linear.fit(X_train, y_train)

# RBF SVM
svm_rbf = SVC(kernel='rbf', probability=True)
svm_rbf.fit(X_train, y_train)

# -----------------------
# 7. Predictions
# -----------------------
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

y_prob_linear = svm_linear.predict_proba(X_test)[:, 1]
y_prob_rbf = svm_rbf.predict_proba(X_test)[:, 1]

# -----------------------
# 8. Evaluation Function
# -----------------------
def evaluate(name, y_true, y_pred, y_prob):
    print("\n======================")
    print(name)
    print("======================")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("AUC      :", roc_auc_score(y_true, y_prob))

evaluate("Linear SVM", y_test, y_pred_linear, y_prob_linear)
evaluate("RBF SVM", y_test, y_pred_rbf, y_prob_rbf)

# -----------------------
# 9. Confusion Matrix (Linear SVM)
# -----------------------
cm = confusion_matrix(y_test, y_pred_linear)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Linear SVM")
plt.show()

# -----------------------
# 10. Simple Comparison Plot
# -----------------------
models = ['Linear SVM', 'RBF SVM']
accuracy = [
    accuracy_score(y_test, y_pred_linear),
    accuracy_score(y_test, y_pred_rbf)
]

plt.figure(figsize=(6,4))
plt.bar(models, accuracy)
plt.title("SVM Model Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.show()
