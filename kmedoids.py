import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

k = 4
max_iter = 100

# random medoids
idx = np.random.choice(len(X), k, replace=False)
medoids = X[idx]

for _ in range(max_iter):

    distances = np.linalg.norm(X[:, None] - medoids, axis=2)

    labels = np.argmin(distances, axis=1)

    new_medoids = []

    for i in range(k):
        cluster = X[labels == i]

        if len(cluster) == 0:
            new_medoids.append(medoids[i])
            continue

        dist = np.sum(
            np.linalg.norm(cluster[:, None] - cluster, axis=2),
            axis=1
        )

        new_medoids.append(cluster[np.argmin(dist)])

    new_medoids = np.array(new_medoids)

    if np.all(medoids == new_medoids):
        break

    medoids = new_medoids


plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("K-Medoids Clustering")
plt.show()