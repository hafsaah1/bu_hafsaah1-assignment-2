import numpy as np

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X, method='random'):
        if method == 'random':
            idx = np.random.choice(X.shape[0], self.k, replace=False)
            self.centroids = X[idx]
        elif method == 'farthest_first':
            self.centroids = self.farthest_first(X)
        elif method == 'kmeans++':
            self.centroids = self.kmeans_plus_plus(X)
        # Manual initialization will be handled on the frontend
    
    def farthest_first(self, X):
        centroids = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.k):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def kmeans_plus_plus(self, X):
        centroids = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.k):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            probabilities = distances / np.sum(distances)
            next_centroid = X[np.random.choice(range(X.shape[0]), p=probabilities)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def fit(self, X):
        for _ in range(self.max_iters):
            distances = self._compute_distances(X)
            labels = self._assign_clusters(distances)
            new_centroids = self._compute_centroids(X, labels)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        self.labels = labels

    def _compute_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

    def predict(self, X):
        distances = self._compute_distances(X)
        return self._assign_clusters(distances)
