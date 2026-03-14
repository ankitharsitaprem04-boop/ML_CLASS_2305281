import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# KMeans
from sklearn.cluster import KMeans

model = KMeans(n_clusters=4)
labels = model.fit_predict(X)

# visualization
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("KMeans Clustering")
plt.show()