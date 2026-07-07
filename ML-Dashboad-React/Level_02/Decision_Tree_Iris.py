#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
#check data
X.head()

#Explore Data
print(X.shape)
print(X.info())
print(X.describe())
#Check classes
print(iris.target_names)

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#Train Decision Tree Model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)

#Evaluate Model
#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Confusion Matrix
print(confusion_matrix(y_test, y_pred))

#Full Report
print(classification_report(y_test, y_pred))

#Visualize Decision Tree
plt.figure(figsize=(12,8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.show()

#Pruning Concept
DecisionTreeClassifier(max_depth=3)
