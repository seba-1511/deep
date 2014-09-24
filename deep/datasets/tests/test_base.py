from deep.datasets import load_mnist

import numpy as np


def test_load_mnist():
    data = load_mnist()
    X_train, y_train = data[0]
    assert(X_train.shape == (50000, 784))
    assert(y_train.shape == (50000,))
    assert(np.unique(y_train).size == 10)
    X_valid, y_valid = data[1]
    assert(X_valid.shape == (10000, 784))
    assert(y_valid.shape == (10000,))
    assert(np.unique(y_valid).size == 10)
    X_test, y_test = data[2]
    assert(X_test.shape == (10000, 784))
    assert(y_test.shape == (10000,))
    assert(np.unique(y_test).size == 10)


if __name__ == '__main__':
    import nose
    nose.runmodule()