import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

k = 4
max_iter = 100

# initialize medians randomly
idx = np.random.choice(len(X), k, replace=False)
medians = X[idx]

for _ in range(max_iter):

    # Manhattan distance
    distances = np.sum(np.abs(X[:, None] - medians), axis=2)

    labels = np.argmin(distances, axis=1)

    new_medians = np.array([
        np.median(X[labels == i], axis=0)
        for i in range(k)
    ])

    if np.all(medians == new_medians):
        break

    medians = new_medians


plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("K-Median Clustering")
plt.show()