import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class RBFNN:
    def __init__(self, num_centers, sigma=1.0):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _rbf(self, X, center):
        return np.exp(-self.sigma * cdist(X, center[np.newaxis], 'euclidean')**2).flatten()

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.num_centers)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        G = np.zeros((X.shape[0], self.num_centers), dtype=np.float32)
        for i, center in enumerate(self.centers):
            G[:, i] = self._rbf(X, center)

        self.weights = np.linalg.pinv(G).dot(y)

    def predict(self, X):
        G = np.zeros((X.shape[0], self.num_centers), dtype=np.float32)
        for i, center in enumerate(self.centers):
            G[:, i] = self._rbf(X, center)

        return G.dot(self.weights)

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.rand(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

    rbfnn = RBFNN(num_centers=10, sigma=0.1)
    rbfnn.fit(X_train, y_train)

    X_test = np.random.rand(10, 2)
    y_test = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1])
    y_pred = rbfnn.predict(X_test)

    print("Исходник:", y_test)
    print("Предсказ:", y_pred)
