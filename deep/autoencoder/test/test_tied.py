import unittest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal, assert_equal, assert_raises
from deep.autoencoder.tied import TiedAE

iris = load_iris()
X = iris.data


class TestTiedAE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Init and fit FeedForwardNN. """
        cls.ae = TiedAE(n_hidden=100, n_iter=5, learning_rate=1)

        try:
            cls.ae.fit(X)
        except Exception:
            print 'Exception while fitting'

    def test_fit(self):
        """ Test that score (% correct) increases each iter of fit. """
        assert_array_equal(np.diff(self.ae._scores) >= 0, True)