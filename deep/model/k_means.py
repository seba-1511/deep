import numpy as np
import matplotlib.pyplot as plt


class KMeans(object):

    def __init__(self, n_clusters=2):

        self.n_clusters = n_clusters

    def e_step(self, X, centers):

        distances = np.zeros((X.shape[0], centers.shape[0]))

        for index, center in enumerate(centers):

            distances[:, index] = np.sum((X - center)**2, axis=1)

        labels = np.argmin(distances, axis=1)

        return labels

    def m_step(self, X, centers, labels):

        for index, center in enumerate(centers):

            centers[index] = np.sum(X[labels == index], axis=0)
            centers[index] /= np.sum(labels == index)

        return centers

    def fit(self, X=None):

        if not X:
            np.random.seed(0)
            X = np.random.random((15, 2))
            X[:5] += 1
            X[:10] += 2

        n_samples, n_features = X.shape
        centers = np.random.random((self.n_clusters, n_features))

        for i in range(5):
            self.plot(X, centers)

            labels = self.e_step(X, centers)
            centers = self.m_step(X, centers, labels)

    def plot(self, X, centers):

        for x, y in centers:
            plt.scatter(x, y, c='r', s=50)

        for x, y in X:
            plt.scatter(x, y)
        plt.show()


