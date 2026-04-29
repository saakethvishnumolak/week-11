import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from time import time
import seaborn as sns


# ===============================
# Exercise 1: KMeans function
# ===============================
def kmeans(X, k):
    """
    Perform k-means clustering.

    Parameters:
        X (np.array): numerical data
        k (int): number of clusters

    Returns:
        centroids (np.array): shape (k, n_features)
        labels (np.array): shape (n_samples,)
    """
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels


# ===============================
# Exercise 2: Diamonds dataset
# ===============================

# Load dataset globally
diamonds = sns.load_dataset("diamonds")

# Keep only numeric columns
diamonds_numeric = diamonds.select_dtypes(include=[np.number])


def kmeans_diamonds(n, k):
    """
    Run kmeans on first n rows of diamonds numeric data.
    """
    df_subset = diamonds_numeric.iloc[:n]

    X = df_subset.values

    centroids, labels = kmeans(X, k)

    return centroids, labels


# ===============================
# Exercise 3: Timing function
# ===============================
def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds multiple times and return avg runtime.
    """
    times = []

    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        elapsed = time() - start
        times.append(elapsed)

    return np.mean(times)