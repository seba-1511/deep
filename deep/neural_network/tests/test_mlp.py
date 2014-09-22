from deep.datasets import load_iris
from deep.datasets import load_digits
from deep.neural_network import SigmoidLayer
from deep.neural_network import SoftMaxLayer
from deep.neural_network import TheanoMultiLayerPerceptron

import numpy as np


def test_fit_with_iris():
    X, y = load_iris()
    X -= X.min()
    X /= X.max()

    layer1 = SigmoidLayer(4, 30)
    layer2 = SoftMaxLayer(30, 3)
    clf = TheanoMultiLayerPerceptron([layer1, layer2])
    clf.fit(X, y)

    assert(np.all(np.diff(clf.costs_) < 0))


def test_fit_with_digits():
    X, y = load_digits()
    X -= X.min()
    X /= X.max()

    layer1 = SigmoidLayer(64, 32)
    layer2 = SoftMaxLayer(32, 10)
    clf = TheanoMultiLayerPerceptron([layer1, layer2])
    clf.fit(X, y)

    assert(np.all(np.diff(clf.costs_) < 0))


if __name__ == '__main__':
    import nose
    nose.runmodule()