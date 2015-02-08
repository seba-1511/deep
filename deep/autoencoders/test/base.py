import unittest

from deep.autoencoders.tied import TiedAE

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal

import numpy as np


iris = load_iris()
X = iris.data
y = iris.target

X -= X.min()
X /= X.std()


class TestTiedAE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ae = TiedAE(learning_rate=.01, n_iter=5, n_hidden=10)

        try:
            cls.ae.fit(X)
        except Exception:
            print 'Exception while fitting'

    # test fit method

    def test_fit(self):
        assert_array_equal(np.diff(self.ae._scores) <= 0, True)

    def test_deterministic_fit(self):
        ae = TiedAE(learning_rate=.01, n_iter=5, n_hidden=10)
        ae.fit(X)
        assert_array_equal(self.ae._scores, ae._scores)

    # test input dimensions

    def test_transform_1d(self):
        assert_equal(self.ae.__call__(X[0]).shape, (self.ae.n_hidden,))
