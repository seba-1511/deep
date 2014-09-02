import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def multivariate_gaussian_density(X, means, covars):

    n_features = X.shape[1]

    for index, (mean, covar) in enumerate(zip(means, covars)):

        variance = np.dot(X - mean, np.dot(X - mean, covar).T)

        det = np.linalg.det(covar)

        return 1 / (2 * np.pi) ** (n_features / 2) / det ** .5 * np.exp(-.5 * variance)


class GaussianMixture(object):

    def __init__(self, n_components=3):

        self.n_components = n_components

    def e_step(self):

        raise NotImplementedError

    def m_step(self):

        raise NotImplementedError

    def fit(self, X=None):

        if not X:
            np.random.seed(0)
            X = np.random.random((15, 2))
            X[:5] += 1
            X[:10] += 2

        n_samples, n_features = X.shape

        means = np.random.random((n_features, self.n_components))
        covars = np.array([np.eye(n_features) for _ in range(self.n_components)])
        weights = np.tile(1.0 / self.n_components, self.n_components)



    def plot(self, X, centers):

        for x, y in centers:
            plt.scatter(x, y, c='r', s=50)

        for x, y in X:
            plt.scatter(x, y)
        plt.show()




np.random.seed(0)
X = np.random.random((15, 2))
X[:5] += 1
X[:10] += 2

means = np.random.random((3, 2))
covars = np.array([np.eye(2) for _ in range(3)])

print multivariate_gaussian_density(X, means, covars)

#gmm = GaussianMixture()
#gmm.fit()