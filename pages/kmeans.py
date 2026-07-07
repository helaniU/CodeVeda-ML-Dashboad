import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

FIGSIZE = (4.5, 3)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Clustering Dashboard", layout="wide")

st.title("🟡 K-Means vs DBSCAN - Clustering Analysis Tool")

# =========================
# LOAD DATA
# =========================
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

st.subheader("📊 Dataset Preview")
st.write(X.head())

# =========================
# SCALE DATA
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Controls")

algo = st.sidebar.selectbox("Choose Algorithm", ["KMeans", "DBSCAN"])

# =========================================================
# KMEANS SECTION
# =========================================================
if algo == "KMeans":

    st.sidebar.subheader("KMeans Settings")
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)

    # =========================
    # ELBOW + SILHOUETTE
    # =========================
    st.subheader("📉 Model Selection Analysis")

    ks = range(2, 11)
    inertia = []
    sil_scores = []

    for i in ks:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(ks, inertia, marker='o')
        ax1.set_title("Elbow Method (WCSS)")
        ax1.set_xlabel("K")
        ax1.set_ylabel("WCSS")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(ks, sil_scores, marker='o', color='green')
        ax2.set_title("Silhouette Score vs K")
        ax2.set_xlabel("K")
        ax2.set_ylabel("Score")
        st.pyplot(fig2)

    # =========================
    # TRAIN MODEL
    # =========================
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, clusters)

    st.success(f"Silhouette Score (Current K): {score:.3f}")

    # =========================
    # PCA VISUALIZATION
    # =========================

    st.subheader("📌 PCA Cluster Visualization")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig3, ax3 = plt.subplots(figsize=FIGSIZE)

    ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=20)

    centers = pca.transform(model.cluster_centers_)
    ax3.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100)

    ax3.set_title("PCA Cluster View")
    st.pyplot(fig3, use_container_width=False)

    # =========================
    # DATAFRAME WITH CLUSTERS
    # =========================
    X_clustered = X.copy()
    X_clustered["Cluster"] = clusters

    # =========================
    # CLUSTER ANALYSIS TOOL
    # =========================
    st.subheader("🧠 Cluster Analysis Tool")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Cluster Counts")
        st.write(X_clustered["Cluster"].value_counts())

    with col2:
        st.write("Cluster Mean Values")
        st.dataframe(X_clustered.groupby("Cluster").mean())

    # =========================
    # FEATURE INFLUENCE (NEW)
    # =========================
    st.subheader("📊 Feature Influence (Cluster Separation Strength)")

    feature_influence = X_clustered.groupby("Cluster").mean().var().sort_values(ascending=False)

    fig4, ax4 = plt.subplots(figsize=FIGSIZE)

    ax4.bar(feature_influence.index, feature_influence.values)
    ax4.set_title("Feature Influence")
    ax4.set_ylabel("Importance")

    plt.xticks(rotation=30)

    st.pyplot(fig4, use_container_width=False)

    # =========================
    # CLUSTER EXPLORER (NEW)
    # =========================
    st.subheader("🔍 Cluster Explorer Mode")

    selected_cluster = st.selectbox("Select Cluster", sorted(X_clustered["Cluster"].unique()))

    cluster_data = X_clustered[X_clustered["Cluster"] == selected_cluster]

    st.write(f"📌 Cluster {selected_cluster} Data")
    st.dataframe(cluster_data)

    st.write("📊 Cluster Statistics")
    st.dataframe(cluster_data.describe())

    # =========================
    # DOWNLOAD CSV
    # =========================
    csv = X_clustered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Clustered Data",
        csv,
        "kmeans_clusters.csv",
        "text/csv"
    )

# =========================================================
# DBSCAN SECTION
# =========================================================
else:

    st.sidebar.subheader("DBSCAN Settings")
    eps = st.sidebar.slider("eps", 0.1, 5.0, 1.0)
    min_samples = st.sidebar.slider("min_samples", 2, 10, 5)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    st.subheader("📌 DBSCAN Cluster Visualization")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig5, ax5 = plt.subplots(figsize=(4.5, 3))

    ax5.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap='rainbow',
        s=20
    )

    ax5.set_title("DBSCAN Clusters (PCA Reduced)")
    ax5.set_xlabel("PC1")
    ax5.set_ylabel("PC2")

    st.pyplot(fig5, use_container_width=False)

# =========================
# INFO SECTION
# =========================
st.markdown("---")

st.subheader("ℹ️ What this tool does")

st.write("""
✔ Compares KMeans and DBSCAN  
✔ Elbow Method + Silhouette Score  
✔ PCA-based visualization  
✔ Cluster analysis + feature influence  
✔ Cluster explorer mode  
✔ Downloadable clustered dataset  
✔ Interactive parameter tuning  
""")