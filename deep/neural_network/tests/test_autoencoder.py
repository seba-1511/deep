from deep.datasets import load_iris
from deep.datasets import load_digits
from deep.neural_network import SigmoidLayer
from deep.neural_network import TheanoStackedAutoEncoder

import numpy as np


def test_fit_with_iris():
    X, y = load_iris()
    X -= X.min()
    X /= X.max()

    layer1 = SigmoidLayer(4, 40)
    clf = TheanoStackedAutoEncoder([layer1], n_iter=10, learning_rate=1)
    clf.fit(X)

    assert(np.all(np.diff(clf.costs_) < 0))


def test_fit_with_digits():
    X, y = load_digits()
    X -= X.min()
    X /= X.max()

    layer1 = SigmoidLayer(64, 32)
    clf = TheanoStackedAutoEncoder([layer1], n_iter=10, learning_rate=1)
    clf.fit(X)

    assert(np.all(np.diff(clf.costs_) < 0))


if __name__ == '__main__':
    import nose
    nose.runmodule()