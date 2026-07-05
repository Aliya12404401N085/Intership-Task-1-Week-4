import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering

# Load Iris dataset
iris = load_iris()
X = iris.data

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# K-Means Plot
plt.scatter(X[:,0], X[:,1], c=kmeans_labels)
plt.title("K-Means Clustering")
plt.savefig("images/kmeans.png")
plt.close()

# Hierarchical Plot
plt.scatter(X[:,0], X[:,1], c=hierarchical_labels)
plt.title("Hierarchical Clustering")
plt.savefig("images/hierarchical.png")
plt.close()

# DBSCAN Plot
plt.scatter(X[:,0], X[:,1], c=dbscan_labels)
plt.title("DBSCAN Clustering")
plt.savefig("images/dbscan.png")
plt.close()

print("Clustering completed successfully.")
