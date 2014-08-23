class GaussianMixture(object):

    def __init__(self, n_clusters=2):

        self.n_clusters = n_clusters


    def e_step(self):

        raise NotImplementedError

    def m_step(self):

        raise NotImplementedError

    def fit(self, X):

        raise NotImplementedError

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM


def make_ellipses(gmm, ax):
    for n, color in enumerate('rg'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


np.random.seed(0)
X = np.random.random((10, 2))
X[:5] += 1

classifier = GMM(n_components=2)
classifier.fit(X)

h = plt.subplot(111)
make_ellipses(classifier, h)

plt.scatter(X[:, 0], X[:, 1])

plt.show()
