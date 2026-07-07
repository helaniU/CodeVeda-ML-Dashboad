#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Find Best K (Elbow Method)
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

#📊 Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

#Train K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

#Add Cluster Labels
X["Cluster"] = clusters

#Visualize Clusters (2D)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='red',
    marker='X'
)

plt.title("K-Means Clusters (Iris)")
plt.show()

#Compare with Real Labels
from sklearn.metrics import confusion_matrix
print(confusion_matrix(iris.target, clusters))

