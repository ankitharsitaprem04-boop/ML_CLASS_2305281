import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# DBSCAN
model = DBSCAN(eps=0.8, min_samples=5)
labels = model.fit_predict(X)

# visualization
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("DBSCAN Clustering")
plt.show()